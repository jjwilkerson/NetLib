/**
 * @file
 * @brief Defines Network class, which represents an entire network.
 *
 */

#include "Network.h"
#include "layers/Layer.h"
#include "gpu/CudaFunc.h"
#include "gpu/CublasFunc.h"
#include <iostream>
#include <algorithm>
#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "config/Config.h"
#include "config/WeightInit.h"
#include "input/InputSource.h"
#include "layers/InputLayer.h"
#include "loss/LossFunction.h"
#include "util/FileUtil.h"

int print_s = -1;

using namespace std;

string datasetsDir = getenv("SED_DATASETS_DIR") ? getenv("SED_DATASETS_DIR") : "";

namespace netlib {

Network::Network(cublasHandle_t& handle, int batchSize, int maxSeqLength, Config config, LossFunction& lossFunction,
		WeightInit& ffWeightInit, WeightInit& recWeightInit, bool training)
		: handle(handle), batchSize(batchSize), maxSeqLength(maxSeqLength), lossFunction(lossFunction),
		  ffWeightInit(ffWeightInit), recWeightInit(recWeightInit), training(training) {
	numBatchError = config.numBatchError;
	numBatchGrad = config.numBatchGrad;
	numBatchG = config.numBatchG;
	rngSeed = config.seed;
	l2 = config.l2;
	weightsFilename = config.weightsFilename;
	curandStatesFilename = config.curandStatesFilename;

	if (config.pDropMatch > 0) {
		matchDrop = true;
	} else {
		matchDrop = false;
	}

	optHF = config.optimizer == Config::HF;

	CudaFunc::pDropMatch = config.pDropMatch;
	CudaFunc::batchSize = batchSize;
	CudaFunc::maxSentenceLength = maxSeqLength;

	lossScaleEnabled = config.lossScaleEnabled;
	lossScaleFac = config.lossScaleFac;
	lossScaleUpFac = config.lossScaleUpFac;
	lossScaleDnFac = config.lossScaleDnFac;
	lossScalePeriod = config.lossScalePeriod;
	iterNoOverflow = config.iterNoOverflow;

	if (!lossScaleEnabled) {
		lossScaleFac = 1.0;
	}

	cout << "loss scale factor: " << lossScaleFac << endl;
}

Network::~Network() {
	freeMem();

	vector<Layer*>::iterator it;
	for (it = layers.begin(); it != layers.end(); it++) {
		Layer* layer = *it;
		delete layer;
	}

	if (trainSource != NULL) {
		delete trainSource;
	}

	CudaFunc::freeCurandStates();
}

//void Network::addInputSource(InputSource* source) {
//	if (find(trainSources.begin(), trainSources.end(), source) == trainSources.end()) {
//		trainSources.push_back(source);
//	}
//}

void Network::setTrainSource(InputSource* source) {
	trainSource = source;
}

void Network::addInput(InputLayer* layer) {
	layer->net = this;
	if (find(inputLayers.begin(), inputLayers.end(), layer) == inputLayers.end()) {
		inputLayers.push_back(layer);
	}
	if (find(layers.begin(), layers.end(), layer) == layers.end()) {
		layers.push_back(layer);
	}
	layer->structDamp = false;
}

void Network::setOutput(Layer* layer) {
	layer->net = this;
	outputLayer = layer;
	if (find(layers.begin(), layers.end(), layer) == layers.end()) {
		layers.push_back(layer);
	}
	layer->structDamp = true; //TODO: should be false?
}

void Network::addHidden(Layer* layer) {
	layer->net = this;
//	if (find(hiddenLayers.begin(), hiddenLayers.end(), layer) == hiddenLayers.end()) {
//		hiddenLayers.push_back(layer);
//	}
	if (find(layers.begin(), layers.end(), layer) == layers.end()) {
		layers.push_back(layer);
	}
	layer->structDamp = true;
}

Layer* Network::getOutputLayer() {
	return outputLayer;
}

void Network::init() {
	lossFunction.setNetwork(this);

	calcOffsets();
	initMem();

	if (weightsFilename == "") {
		initWeights();
	} else {
		loadWeights(weightsFilename);
	}

	if (curandStatesFilename == "") {
		initCurand();
	} else {
		loadCurand(curandStatesFilename);
	}
}

dtype2** Network::getRevInputs() {
	if (currSource == NULL) {
		if (trainSource == NULL) {
			return NULL;
		}
		currSource = trainSource;
	}

	return currSource->getRevInputs();
}

dtype2** Network::getFwdInputs() {
	if (currSource == NULL) {
		if (trainSource == NULL) {
			return NULL;
		}
		currSource = trainSource;
	}

	return currSource->getFwdInputs();
}

unsigned int** Network::getDInputLengths() {
	if (currSource == NULL) {
		if (trainSource == NULL) {
			return NULL;
		}
		currSource = trainSource;
	}

	return currSource->getDInputLengths();
}

unsigned int* Network::getHInputLengths() {
	if (currSource == NULL) {
		if (trainSource == NULL) {
			return NULL;
		}
		currSource = trainSource;
	}

	return currSource->getHInputLengths();
}


dtype2** Network::getTargets() {
	if (currSource == NULL) {
		if (trainSource == NULL) {
			return NULL;
		}
		currSource = trainSource;
	}

	return currSource->getTargets();
}

dtype2** Network::matchMasks(int batchNum) {
	if (!matchDrop) {
		return NULL;
	}
	return trainSource->matchMasks(batchNum);
}

void Network::calcOffsets() {
	int offset = paramOffset;
	vector<Layer*>::iterator it;
	for (it = layers.begin(); it != layers.end(); it++) {
		Layer* layer = *it;
		layer->setParamOffset(offset);
		offset += layer->getNParams();
	}
	nParams = offset;

	cout << "nParams: " << nParams << endl;
}

void Network::initMem() {
	cout << endl << "Allocating GPU arrays" << endl;

	size_t arraySize = nParams * sizeof(dtype2);
    checkCudaErrors(cudaMalloc((void **)&W, arraySize));
	checkCudaErrors(cudaMallocHost((void **)&h_W, arraySize));

#ifdef PARAM_SINGLE_HALF
	size_t arraySize1 = nParams * sizeof(dtype1);
	checkCudaErrors(cudaMallocHost((void **)&h_W1, arraySize1));
	checkCudaErrors(cudaMalloc((void **)&W1, arraySize1));
#else
    h_W1 = h_W;
#endif

//	losses_d = new dtype2*[maxSeqLength];
//	for (int s = 0; s < maxSeqLength; s++) {
//		checkCudaErrors(cudaMalloc((void **)&losses_d[s], batchSize * sizeof(dtype2)));
//	}

	vector<Layer*>::iterator it;
	for (it = layers.begin(); it != layers.end(); it++) {
		Layer* layer = *it;
		layer->W = W;
		layer->initMem(training, optHF);
	}
}

void Network::freeMem() {
	cout << endl << "freeMem called" << endl;

    checkCudaErrors(cudaFree(W));
    checkCudaErrors(cudaFreeHost(h_W));
    h_W = NULL;

    if (W1 != NULL) {
        checkCudaErrors(cudaFree(W1));
    }
#ifdef PARAM_SINGLE_HALF
    checkCudaErrors(cudaFreeHost(h_W1));
#endif
//    for (int s = 0; s < maxSeqLength; s++) {
//    	checkCudaErrors(cudaFree(losses_d[s]));
//    }
//    delete [] losses_d;

    vector<Layer*>::iterator it;
	for (it = layers.begin(); it != layers.end(); it++) {
		Layer* layer = *it;
		layer->freeMem(training);
	}
}

void Network::initWeights() {
	vector<Layer*>::iterator it;
	for (it = layers.begin(); it != layers.end(); it++) {
		Layer* layer = *it;
		layer->initWeights(ffWeightInit, recWeightInit);
	}
}

void Network::loadWeights(string weightsFilename) {
	cout << endl << "loadWeights" << endl;

	ifstream file(weightsFilename.c_str(), ios::binary);

	unsigned arraySize = nParams * sizeof(dtype1);

	for (int i = 0; i < nParams; i++) {
		dtypeh p = FileUtil::readFloat(file);
		h_W1[i] = float21(p);
	}

	dtypeh min = 99999999;
	dtypeh max = -99999999;
	dtypeh sum = 0;
	dtypeh sumAbs = 0;
	int numNonZero = 0;
	for (int i = 0 ; i < nParams; i++) {
		dtypeh val = d1h(h_W1[i]);
		if (val < min) {
			min = val;
		}
		if (val > max) {
			max = val;
		}
		sum += val;
		sumAbs += abs(val);
		if (val != 0) {
			numNonZero++;
		}
	}
	cout << "min: " << min << endl;
	cout << "max: " << max << endl;
	cout << "mean: " << (sum/nParams) << endl;
	cout << "mean(abs): " << (sumAbs/nParams) << endl;
	cout << "numNonZero: " << numNonZero << endl;
	cout << endl;

#ifdef PARAM_SINGLE_HALF
	dtype1* params = W1;
#else
	dtype1* params = W;
#endif

	checkCudaErrors(cudaMemcpy(params, h_W1, arraySize, cudaMemcpyHostToDevice));

	file.close();
}

void Network::saveWeights(const char* weightsFilename) {
	ofstream wFile(weightsFilename, ios::binary | ios::trunc);

#ifdef PARAM_SINGLE_HALF
	dtype1* params = W1;
#else
	dtype1* params = W;
#endif

	unsigned arraySize = nParams * sizeof(dtype1);
	checkCudaErrors(cudaMemcpy(h_W1, params, arraySize, cudaMemcpyDeviceToHost));

	for (int i = 0; i < nParams; i++) {
		FileUtil::writeFloat(d1h(h_W1[i]), wFile);
	}

	wFile.close();
	if (!wFile) {
		cerr << "write error" << endl;
		exit(EXIT_FAILURE);
	}
}

void Network::initCurand() {
	int maxDropoutLength = 1;

	vector<Layer*>::iterator it;
	for (it = layers.begin(); it != layers.end(); it++) {
		Layer* layer = *it;
		if (layer->dropout != 0) {
			maxDropoutLength = max(maxDropoutLength, batchSize * layer->size);
		}
	}

	cout << endl << "maxDropoutLength: " << maxDropoutLength << endl;
	CudaFunc::initCurandStates(rngSeed, maxDropoutLength);
}

void Network::loadCurand(string filename) {
	ifstream file(filename.c_str(), ios::binary);

	size_t arraySize;
	file.read(reinterpret_cast<char *>(&arraySize), sizeof(size_t));

	char *a_h;
	a_h = (char*) malloc(arraySize);

	file.read(a_h, arraySize);

	CudaFunc::setCurandStates(a_h, arraySize);

	free(a_h);
}

void Network::saveCurandStates(const char* filename) {
	char *a_h;
	size_t arraySize;
	CudaFunc::getCurandStates(&a_h, &arraySize);

	ofstream sFile(filename, ios::binary | ios::trunc);

	sFile.write(reinterpret_cast<char *>(&arraySize), sizeof(size_t));
	sFile.write(a_h, arraySize);

	sFile.close();

	free(a_h);
	if (!sFile) {
		cerr << "write error" << endl;
		exit(EXIT_FAILURE);
	}
}

void Network::toFirstBatch() {
	trainSource->toFirstBatch();
//	vector<InputSource*>::iterator it;
//	for (it = trainSources.begin(); it != trainSources.end(); it++) {
//		InputSource* source = *it;
//		source->toFirstBatch();
//	}
}

void Network::toNextBatchSet() {
	trainSource->toNextBatchSet();
//	vector<InputSource*>::iterator it;
//	for (it = trainSources.begin(); it != trainSources.end(); it++) {
//		InputSource* source = *it;
//		source->toNextBatchSet();
//	}
}

void Network::nextBatch(int batchNum) {
	trainSource->nextBatch(batchNum);
//	vector<InputSource*>::iterator it;
//	for (it = trainSources.begin(); it != trainSources.end(); it++) {
//		InputSource* source = *it;
//		source->nextBatch(batchNum);
//	}
}

dtype2 Network::error(dtype2* params, bool stochasticDropout, bool matchDrop, InputSource* source) {
	if (params == NULL) {
		params = W;
	}

	dtypeh error = 0;
	toFirstBatch();
	for (int i = 0; i < numBatchError; i++) {
		nextBatch(i);
		forward(i, params, false, stochasticDropout, source, matchDrop);
		dtype2** masks = NULL;
		if (matchDrop) {
			masks = matchMasks(i);
		}
		error += lossFunction.batchLoss(outputLayer->activation, getTargets(), losses_d, false,
				masks, getDInputLengths()[i]);
	}

	return error / (batchSize * numBatchError);
}

void Network::forward(int batchNum, dtype2* params, bool deriv, bool stochasticDropout,
		InputSource* source, bool computeMatchMasks) {
	if (source == NULL) {
		currSource = trainSource;
	} else {
		currSource = source;
	}

	vector<InputLayer*>::iterator it;
	for (it = inputLayers.begin(); it != inputLayers.end(); it++) {
		(*it)->forward(batchNum, params, deriv, stochasticDropout);
	}

//	cout << "forward" << endl;
//	int print_s = 0;
//	cout << "s=" << print_s << endl;
//	vector<Layer*>::iterator it2;
//	for (it2 = layers.begin(); it2 != layers.end(); it2++) {
//		Layer* layer = *it2;
////		cout << layer->name << endl;
////		cout << layer->size << endl;
////		printAllGpu(layer->activation[print_s], layer->size*batchSize);
//		printStatsGpu(layer->name, layer->activation[print_s], layer->size*batchSize);
//	}
//	print_s = 30;
//	cout << "print_s=" << print_s << endl;
//	for (it2 = layers.begin(); it2 != layers.end(); it2++) {
//		Layer* layer = *it2;
////		int s = print_s;
////		if (layer->seqLength < print_s+1) s = 0;
////		cout << endl << "s=" << s << endl;
//		int s = 0;
//		printStatsGpu(layer->name, layer->activation[s], layer->size*batchSize);
//	}
//	exit(0);

	if (matchDrop && currSource == trainSource && computeMatchMasks) {
		trainSource->computeMatchMasks(batchNum, outputLayer->activation);
	}
}

void Network::forwardNext(vector<string>* tokensRet) {
	trainSource->nextBatch(0, tokensRet);
	forward(0, W, false, false, NULL, false);
}

dtype2* Network::calcGrad() {
	dtype2* grad;
	checkCudaErrors(cudaMalloc((void **)&grad, nParams * sizeof(dtype2)));
	checkCudaErrors(cudaMemset((void *)grad, 0, nParams * sizeof(dtype2)));

	toFirstBatch();
	for (int i = 0; i < numBatchGrad; i++) {
		nextBatch(i);
		forward(i, W, true, true);

		vector<Layer*>::iterator it;
		for (it = layers.begin(); it != layers.end(); it++) {
			Layer* layer = *it;
			layer->clearError();
		}

		outputLayer->calcGrad(grad, i);
	}

	dtypeh scale =  1.0 / (dtypeh) (batchSize * numBatchGrad);

	if (lossScaleEnabled) {
		scale /= lossScaleFac;

		bool overflow;
		CudaFunc::checkForOverflow(grad, nParams, &overflow);
		if (overflow) {
			gradOverflow = true;
			iterNoOverflow = 0;
			lossScaleFac *= lossScaleDnFac;
			cout << "grad overflow. lossScale down -> " << lossScaleFac << endl;
		} else {
			gradOverflow = false;
			iterNoOverflow++;
			if (iterNoOverflow > lossScalePeriod) {
				lossScaleFac *= lossScaleUpFac;
				cout << "lossScale up -> " << lossScaleFac << endl;
				iterNoOverflow = 0;
			}
		}
	}

	CublasFunc::scal(handle, nParams, &scale, grad, 1);

//	cout << "calcGrad" << endl;
//	vector<Layer*>::iterator it2;
//	for (it2 = layers.begin(); it2 != layers.end(); it2++) {
//		Layer* layer = *it2;
//		int s = 0;
//		printStatsGpu(layer->name, layer->error[s], layer->size*batchSize);
//	}
//	exit(0);

	if (l2 > 0) {
		vector<Layer*>::iterator it;
		for (it = layers.begin(); it != layers.end(); it++) {
			Layer* layer = *it;
			layer->addGradL2(grad, l2);
		}
	}

	return grad;
}

void Network::checkGrad(dtype2* gradCalc) {
	int arraySize = nParams * sizeof(dtype2);
	dtype2* gradCalc_h;
	checkCudaErrors(cudaMallocHost((void **)&gradCalc_h, arraySize));
	checkCudaErrors(cudaMemcpy(gradCalc_h, gradCalc, arraySize, cudaMemcpyDeviceToHost));

	if (l2 > 0) {
		checkCudaErrors(cudaMemcpy(h_W, W, arraySize, cudaMemcpyDeviceToHost));
	}

	dtype2 eps = 1e-7;

	dtype2* inc;
	checkCudaErrors(cudaMalloc((void **)&inc, sizeof(dtype2)));
	checkCudaErrors(cudaMemcpy(inc, &eps, sizeof(dtype2), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaDeviceSynchronize());

	dtype2* paramInc;
	checkCudaErrors(cudaMalloc((void **)&paramInc, arraySize));

	dtype2* gradFinDiff = (dtype2*) malloc(arraySize);

	dtype2 l2fac = 2 * l2 * eps;

	for (unsigned long i = 0; i < nParams; i++) {
		if ((i % 10) == 0) {
			cout << "i: " << i << endl;
		}

		checkCudaErrors(cudaMemcpy(paramInc, W, arraySize, cudaMemcpyDeviceToDevice));
		CublasFunc::axpy(handle, 1, &one, inc, 1, paramInc+i, 1);
		checkCudaErrors(cudaDeviceSynchronize());
		dtypeh errorInc = error(paramInc);

		checkCudaErrors(cudaMemcpy(paramInc, W, arraySize, cudaMemcpyDeviceToDevice));
		CublasFunc::axpy(handle, 1, &minus_one, inc, 1, paramInc+i, 1);
		checkCudaErrors(cudaDeviceSynchronize());
		dtypeh errorDec = error(paramInc);

		dtypeh del = errorInc - errorDec;

		if (l2 > 0) {
			if (isWeight(i)) {
				del += l2fac * h_W[i];
			}
		}
		gradFinDiff[i] = del / (2 * eps);
	}

	dtypeh sqsum = 0.0;
	for (int i = 0; i < nParams; i++) {
		sqsum += pow(gradFinDiff[i], 2);
	}
	cout << endl << "finite diff grad norm: " << sqrt(sqsum) << endl;

	bool passed = checkClose(gradCalc_h, gradFinDiff, nParams);

	if (passed) {
		cout << endl << "grad check passed" << endl;
	} else {
		cout << endl << "grad check failed" << endl;
	}

	checkCudaErrors(cudaFreeHost(gradCalc_h));
	checkCudaErrors(cudaFree(inc));
	checkCudaErrors(cudaFree(paramInc));
	free(gradFinDiff);
}

dtype2* Network::calcG(dtype2* v, dtypeh damping, dtype2* out) {
	dtype2* Gv;
	if (out == NULL) {
		checkCudaErrors(cudaMalloc((void **)&Gv, nParams * sizeof(dtype2)));
	} else {
		Gv = out;
	}

	checkCudaErrors(cudaMemset((void *)Gv, 0, nParams * sizeof(dtype2)));

	toFirstBatch();
	for (int i = 0; i < numBatchG; i++) {
		nextBatch(i);
		forward(i, W, true, true);
		Rforward(v, i);
		Rback(Gv, i);
	}

	dtypeh scale =  1.0 / (dtypeh) (batchSize * numBatchG);

	CublasFunc::scal(handle, nParams, &scale, Gv, 1);
	CublasFunc::axpy(handle, nParams, &damping, v, 1, Gv, 1);

	return Gv;
}

void Network::Rforward(dtype2* v, int batchNum) {
	vector<Layer*>::iterator it;
	for (it = layers.begin(); it != layers.end(); it++) {
		Layer* layer = *it;
		layer->clearRact();
	}

	vector<InputLayer*>::iterator iit;
	for (iit = inputLayers.begin(); iit != inputLayers.end(); iit++) {
		(*iit)->Rforward(v, batchNum);
	}
}

void Network::Rback(dtype2* Gv, int batchNum) {
	vector<Layer*>::iterator it;
	for (it = layers.begin(); it != layers.end(); it++) {
		Layer* layer = *it;
		layer->clearError();
	}

	outputLayer->Rback(Gv, batchNum);
}

bool Network::hasParam(int i) {
	vector<Layer*>::iterator it;
	for (it = layers.begin(); it != layers.end(); it++) {
		Layer* layer = *it;
		if (layer->hasParam(i)) {
			return true;
		}
	}
	return false;
}

bool Network::isWeight(int i) {
	vector<Layer*>::iterator it;
	for (it = layers.begin(); it != layers.end(); it++) {
		Layer* layer = *it;
		if (layer->isWeight(i)) {
			return true;
		}
	}
	return false;
}

void Network::checkG(dtype2* GvCalc, dtype2* v, dtypeh damping) {
	cout << endl << "checkG" << endl;

	int arraySizeG = nParams * nParams * sizeof(dtype2);

	dtype2* GDiff;
	checkCudaErrors(cudaMalloc((void **)&GDiff, arraySizeG));
	checkCudaErrors(cudaMemset((void *)GDiff, 0, arraySizeG));

	int numLayers = layers.size();

	dtype2** J = new dtype2*[numLayers];
	dtype2** intermed = new dtype2*[numLayers];
	dtype2** L = new dtype2*[numLayers];
	J[0] = intermed[0] = L[0] = NULL;
	for (int l = 1; l < numLayers; l++) {
		Layer *layer = layers[l];
		if (layer->nParams == 0) {
			J[l] = intermed[l] = L[l] = NULL;
			continue;
		}

		int layerSize = layer->size;
		int arraySizeJ = batchSize * nParams * layerSize * sizeof(dtype2);
		int arraySizeL = batchSize * layerSize * sizeof(dtype2);

		checkCudaErrors(cudaMalloc((void **)&J[l], arraySizeJ));
		checkCudaErrors(cudaMalloc((void **)&L[l], arraySizeL));
		checkCudaErrors(cudaMalloc((void **)&intermed[l], arraySizeJ));
	}

	for (int s = 0; s < maxSeqLength; s++) {
		checkJ(s, J);

		for (int l = 1; l < numLayers; l++) {
			Layer *layer = layers[l];
			if (layer->nParams == 0) {
				continue;
			}
			if (layer->seqLength == 1 && s > 0) {
				continue;
			}

			int layerSize = layer->size;
			int arraySizeJ = batchSize * nParams * layerSize * sizeof(dtype2);
			int arraySizeL = batchSize * layerSize * sizeof(dtype2);

			lossFunction.d2_loss(NULL, layer, s, L[l], false, matchMasks(0));

			CublasFunc::dgmm(handle, CUBLAS_SIDE_LEFT, batchSize * layerSize, nParams, J[l], batchSize * layerSize, L[l], 1,
					intermed[l], batchSize * layerSize);
			CublasFunc::gemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, nParams, nParams, batchSize * layerSize, &one, J[l],
					batchSize * layerSize, intermed[l], batchSize * layerSize, &one, GDiff, nParams);
		}
	}

	dtypeh scale = 1.0 / (dtypeh) batchSize;
	CublasFunc::scal(handle, nParams * nParams, &scale, GDiff, 1);

	int arraySizeGv = nParams * sizeof(dtype2);
	dtype2* GvDiff;
	checkCudaErrors(cudaMalloc((void **)&GvDiff, arraySizeGv));
	CublasFunc::gemv(handle, CUBLAS_OP_N, nParams, nParams, &one, GDiff, nParams, v, 1, &zero, GvDiff, 1);
	CublasFunc::axpy(handle, nParams, &damping, v, 1, GvDiff, 1);

	dtype2 *GvDiff_h;
	checkCudaErrors(cudaMallocHost((void **)&GvDiff_h, arraySizeGv));
	checkCudaErrors(cudaMemcpy(GvDiff_h, GvDiff, arraySizeGv, cudaMemcpyDeviceToHost));

	dtype2 *GvCalc_h;
	checkCudaErrors(cudaMallocHost((void **)&GvCalc_h, arraySizeGv));
	checkCudaErrors(cudaMemcpy(GvCalc_h, GvCalc, arraySizeGv, cudaMemcpyDeviceToHost));

	bool passed = checkClose(GvCalc_h, GvDiff_h, nParams);
	if (passed) {
		cout << endl << "G check passed" << endl;
	} else {
		cout << endl << "G check failed" << endl;
	}

	for (int l = 1; l < numLayers; l++) {
		Layer *layer = layers[l];
		if (layer->nParams == 0) {
			continue;
		}

		checkCudaErrors(cudaFree(J[l]));
		checkCudaErrors(cudaFree(L[l]));
		checkCudaErrors(cudaFree(intermed[l]));
	}

	delete [] J;
	delete [] intermed;
	delete [] L;

	checkCudaErrors(cudaFree(GDiff));
	checkCudaErrors(cudaFree(GvDiff));
	checkCudaErrors(cudaFreeHost(GvDiff_h));
	checkCudaErrors(cudaFreeHost(GvCalc_h));
}

void Network::checkJ(int s, dtype2** J) {
	cout << "checkJ, s=" << s << endl;

	int numLayers = layers.size();
	dtype2 eps = 1e-8;

	int arraySize = nParams * sizeof(dtype2);

	dtype2* inc;
	dtype2* paramInc;

	checkCudaErrors(cudaMalloc((void **)&inc, arraySize));
	checkCudaErrors(cudaMemset((void *)inc, 0, arraySize));

	checkCudaErrors(cudaMalloc((void **)&paramInc, arraySize));

	int debugNmax = 1;
	int print_n = 400;

	for (int i = 0; i < nParams; i++) {
		checkCudaErrors(cudaMemcpy(inc+i, &eps, sizeof(dtype2), cudaMemcpyHostToDevice));

		checkCudaErrors(cudaMemcpy(paramInc, W, arraySize, cudaMemcpyDeviceToDevice));
		CublasFunc::axpy(handle, nParams, &one, inc, 1, paramInc, 1);
		forward(0, paramInc);

		for (int l = 1; l < numLayers; l++) {
			Layer *layer = layers[l];
			if (layer->nParams == 0) {
				continue;
			}
			if (layer->seqLength == 1 && s > 0) {
				continue;
			}

			int layerSize = layer->size;
			int colIx = IDX2(0,i,batchSize * layerSize);

			CublasFunc::copy(handle, batchSize * layerSize, layer->activation[s], 1, J[l]+colIx, 1);
		}

		checkCudaErrors(cudaMemcpy(paramInc, W, arraySize, cudaMemcpyDeviceToDevice));
		CublasFunc::axpy(handle, nParams, &minus_one, inc, 1, paramInc, 1);
		forward(0, paramInc);

		for (int l = 1; l < numLayers; l++) {
			Layer *layer = layers[l];
			if (layer->nParams == 0) {
				continue;
			}
			if (layer->seqLength == 1 && s > 0) {
				continue;
			}

			int layerSize = layer->size;
			int colIx = IDX2(0,i,batchSize * layerSize);

			CublasFunc::axpy(handle, batchSize * layerSize, &minus_one, layer->activation[s], 1, J[l]+colIx, 1);
		}

		checkCudaErrors(cudaMemset(inc+i, 0, sizeof(dtype2)));
	}

	dtypeh scale = 1.0 / (dtype2) (2 * eps);

	for (int l = 1; l < numLayers; l++) {
		Layer *layer = layers[l];
		if (layer->nParams == 0) {
			continue;
		}
		if (layer->seqLength == 1 && s > 0) {
			continue;
		}
		int layerSize = layer->size;
		int sizeJ = batchSize * layerSize * nParams;

		CublasFunc::scal(handle, sizeJ, &scale, J[l], 1);

	}

	checkCudaErrors(cudaFree(inc));
	checkCudaErrors(cudaFree(paramInc));
}

void Network::iterInit() {
	vector<Layer*>::iterator it;
	for (it = layers.begin(); it != layers.end(); it++) {
		Layer* layer = *it;
		layer->iterInit();
	}

	copyParams();
}

void Network::copyParams() {
#ifdef PARAM_SINGLE_HALF
	CudaFunc::copy12(W1, W, nParams);
#endif
}

dtype1* Network::getMasterParams() {
#ifdef PARAM_SINGLE_HALF
	return W1;
#else
	return W;
#endif
}

bool Network::checkClose(dtype2* a, dtype2* b, unsigned int size, float rtol,
		float atol) {
	int arraySize = size * sizeof(dtype2);

	dtype2* diff = (dtype2*) malloc(arraySize);
	for (int i = 0; i < size; i++) {
		diff[i] = a[i] - b[i];
	}

	bool passing = true;
	for (int i = 0; i < size; i++) {
		dtype2 absDiff = abs(diff[i]);
		dtype2 tol = atol + rtol * abs(b[i]);
		if (absDiff > tol) {
			passing = false;
			break;
		}
	}

	if (!passing && size <= 5000) {
		cout << endl << "a:" << endl;
		printAll(a, size);

		cout << endl << "b:" << endl;
		printAll(b, size);

		cout << endl << "diff:" << endl;
		printAll(diff, size);

		dtype2* ratio = (dtype2*) malloc(arraySize);
		for (int i = 0; i < size; i++) {
			ratio[i] = a[i] / b[i];
		}
		cout << endl << "ratio:" << endl;
		printAll(ratio, size);
		free(ratio);

//		cout << "Paused. Press Enter to continue." << endl;
//		string line;
//		getline(cin, line);
//		char c;
//		cin >> c;
	}

	free(diff);

	return passing;
}

void Network::printStatsGpu(string label, half* a, unsigned int size) {
	cout << endl << label << endl;

	int arraySize = size * sizeof(half);
	half* a_h;
	checkCudaErrors(cudaMallocHost((void **)&a_h, arraySize));
	checkCudaErrors(cudaMemcpy(a_h, a, arraySize, cudaMemcpyDeviceToHost));

	dtypeh min = 99999999;
	dtypeh max = -99999999;
	dtypeh minAbs = 99999999;
	dtypeh sum = 0;
	dtypeh sumAbs = 0;
	int numNonZero = 0;
	for (int i = 0 ; i < size; i++) {
		dtypeh val =  a_h[i];
		dtypeh absval = abs(val);
		if (val < min) {
			min = val;
		}
		if (val > max) {
			max = val;
		}
		if (absval < minAbs) {
			minAbs = absval;
		}
		sum += val;
		sumAbs += absval;
		if (val != 0) {
			numNonZero++;
		}
	}
	cout << "min: " << min << endl;
	cout << "max: " << max << endl;
	cout << "min(abs): " << minAbs << endl;
	cout << "mean: " << (sum/size) << endl;
	cout << "mean(abs): " << (sumAbs/size) << endl;
	cout << "numNonZero: " << numNonZero << endl;
	cout << endl;

	checkCudaErrors(cudaFreeHost(a_h));
}

void Network::printStatsGpu(string label, float* a, unsigned int size) {
	cout << endl << label << endl;

	int arraySize = size * sizeof(float);
	float* a_h;
	checkCudaErrors(cudaMallocHost((void **)&a_h, arraySize));
	checkCudaErrors(cudaMemcpy(a_h, a, arraySize, cudaMemcpyDeviceToHost));

	dtypeh min = 99999999;
	dtypeh max = -99999999;
	dtypeh minAbs = 99999999;
	dtypeh sum = 0;
	dtypeh sumAbs = 0;
	int numNonZero = 0;
	for (int i = 0 ; i < size; i++) {
		dtypeh val =  a_h[i];
		dtypeh absval = abs(val);
		if (val < min) {
			min = val;
		}
		if (val > max) {
			max = val;
		}
		if (absval < minAbs) {
			minAbs = absval;
		}
		sum += val;
		sumAbs += absval;
		if (val != 0) {
			numNonZero++;
		}
	}
	cout << "min: " << min << endl;
	cout << "max: " << max << endl;
	cout << "min(abs): " << minAbs << endl;
	cout << "mean: " << (sum/size) << endl;
	cout << "mean(abs): " << (sumAbs/size) << endl;
	cout << "numNonZero: " << numNonZero << endl;
	cout << endl;

	checkCudaErrors(cudaFreeHost(a_h));
}

void Network::printStatsGpu(string label, double* a, unsigned int size) {
	cout << endl << label << endl;

	int arraySize = size * sizeof(double);
	dtypeh* a_h;
	checkCudaErrors(cudaMallocHost((void **)&a_h, arraySize));
	checkCudaErrors(cudaMemcpy(a_h, a, arraySize, cudaMemcpyDeviceToHost));

	dtypeh min = 99999999;
	dtypeh max = -99999999;
	dtypeh minAbs = 99999999;
	dtypeh sum = 0;
	dtypeh sumAbs = 0;
	int numNonZero = 0;
	for (int i = 0 ; i < size; i++) {
		dtypeh val =  a_h[i];
		dtypeh absval = abs(val);
		if (val < min) {
			min = val;
		}
		if (val > max) {
			max = val;
		}
		if (absval < minAbs) {
			minAbs = absval;
		}
		sum += val;
		sumAbs += absval;
		if (val != 0) {
			numNonZero++;
		}
	}
	cout << "min: " << min << endl;
	cout << "max: " << max << endl;
	cout << "min(abs): " << minAbs << endl;
	cout << "mean: " << (sum/size) << endl;
	cout << "mean(abs): " << (sumAbs/size) << endl;
	cout << "numNonZero: " << numNonZero << endl;
	cout << endl;

	checkCudaErrors(cudaFreeHost(a_h));
}

void Network::printStatsGpu(string label, unsigned int* a, unsigned int size) {
	cout << endl << label << endl;

	int arraySize = size * sizeof(unsigned int);
	unsigned int* a_h;
	checkCudaErrors(cudaMallocHost((void **)&a_h, arraySize));
	checkCudaErrors(cudaMemcpy(a_h, a, arraySize, cudaMemcpyDeviceToHost));

	unsigned int min = 99999999;
	unsigned int max = 0;
	unsigned int sum = 0;
	int numNonZero = 0;
	for (int i = 0 ; i < size; i++) {
		unsigned int val =  a_h[i];
		if (val < min) {
			min = val;
		}
		if (val > max) {
			max = val;
		}
		sum += val;
		if (val != 0) {
			numNonZero++;
		}
	}
	cout << "min: " << min << endl;
	cout << "max: " << max << endl;
	cout << "mean: " << (sum/size) << endl;
	cout << "numNonZero: " << numNonZero << endl;
	cout << endl;

	checkCudaErrors(cudaFreeHost(a_h));
}

void Network::printAll(dtype2* a, unsigned int size) {
	for (int i = 0; i < size; i++) {
		cout << a[i] << " ";
	}
	cout << endl;
}

void Network::printAll(int* a, unsigned int size) {
	for (int i = 0; i < size; i++) {
		cout << a[i] << " ";
	}
	cout << endl;
}

void Network::printAll(unsigned int* a, unsigned int size) {
	for (int i = 0; i < size; i++) {
		cout << a[i] << " ";
	}
	cout << endl;
}

void Network::printAllGpu(dtype2* a, unsigned int size) {
	int arraySize = size * sizeof(dtype2);
	dtype2* a_h;
	checkCudaErrors(cudaMallocHost((void **)&a_h, arraySize));
	checkCudaErrors(cudaMemcpy(a_h, a, arraySize, cudaMemcpyDeviceToHost));

	dtypeh sum = 0.0;

	for (int i = 0; i < size; i++) {
		dtype2 val = a_h[i];
		cout << val << " ";
		sum += val;
	}
	cout << endl << "sum: " << sum << endl;
	cout << endl;

	checkCudaErrors(cudaFreeHost(a_h));
}

void Network::printRowGpu(const char* label, dtype2* a, unsigned int rows, unsigned int cols,
		unsigned int row) {
	cout << endl << label << endl;

	int arraySize = rows * cols * sizeof(dtype2);
	dtype2* a_h;
	checkCudaErrors(cudaMallocHost((void **)&a_h, arraySize));
	checkCudaErrors(cudaMemcpy(a_h, a, arraySize, cudaMemcpyDeviceToHost));

	dtypeh sum = 0.0;

	for (int j = 0; j < cols; j++) {
		int ix = IDX2(row, j, rows);
		dtype2 val = a_h[ix];
		cout << val << " ";
		sum += val;
	}
	cout << endl << "sum: " << sum << endl;
	cout << endl;

	checkCudaErrors(cudaFreeHost(a_h));
}

void Network::printAllGpu(unsigned int* a, unsigned int size) {
	int arraySize = size * sizeof(unsigned int);
	unsigned int* a_h;
	checkCudaErrors(cudaMallocHost((void **)&a_h, arraySize));
	checkCudaErrors(cudaMemcpy(a_h, a, arraySize, cudaMemcpyDeviceToHost));

	unsigned int sum = 0;

	for (int i = 0; i < size; i++) {
		unsigned int val = a_h[i];
		cout << val << " ";
		sum += val;
	}
	cout << endl << "sum: " << sum << endl;
	cout << endl;

	checkCudaErrors(cudaFreeHost(a_h));
}

void Network::printAllGpu(int* a, unsigned int size) {
	int arraySize = size * sizeof(int);
	int* a_h;
	checkCudaErrors(cudaMallocHost((void **)&a_h, arraySize));
	checkCudaErrors(cudaMemcpy(a_h, a, arraySize, cudaMemcpyDeviceToHost));

	int sum = 0;

	for (int i = 0; i < size; i++) {
		int val = a_h[i];
		cout << val << " ";
		sum += val;
	}
	cout << endl << "sum: " << sum << endl;
	cout << endl;

	checkCudaErrors(cudaFreeHost(a_h));
}

void Network::cpuSumRows(dtype2* d_mat, dtype2* d_vec, int rows, int cols) {
	int arraySize = rows * cols * sizeof(dtype2);
	int vecArraySize = rows * sizeof(dtype2);

	dtype2* h_mat = (dtype2*) malloc(arraySize);
	dtype2* h_vec = (dtype2*) malloc(vecArraySize);

	checkCudaErrors(cudaMemcpy(h_mat, d_mat, arraySize, cudaMemcpyDeviceToHost));

	for (int i = 0; i < rows; i++) {
		dtypeh sum = 0.0f;
		for (int j = 0; j < cols; j++) {
			int ix = IDX2(i,j,rows);
			sum += h_mat[ix];
		}
		h_vec[i] = sum;
	}

	checkCudaErrors(cudaMemcpy(d_vec, h_vec, vecArraySize, cudaMemcpyHostToDevice));
	free(h_vec);
	free(h_mat);
}

bool Network::isGradOverflow() {
	return gradOverflow;
}

bool Network::isLossScaleEnabled() {
	return lossScaleEnabled;
}

dtypeh Network::getLossScaleFac() {
	return lossScaleFac;
}

int Network::getIterNoOverflow() {
	return iterNoOverflow;
}

void Network::setParamOffset(int offset) {
	paramOffset = offset;
}

void Network::clearLayers() {
	layers.clear();
}

} /* namespace netlib */
