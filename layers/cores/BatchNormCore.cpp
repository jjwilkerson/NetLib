/**
 * @file
 * @brief Defines BatchNormCore class, a batch normalization layer core.
 *
 * A network layer core that applies applies batch normalization. Used with BaseLayer.
 *
 */

#include "BatchNormCore.h"
#include "../../Network.h"
#include "../../gpu/CudaFunc.h"
#include "../../gpu/CublasFunc.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <iostream>

extern int print_s;

using namespace std;

namespace netlib {

map<string, BatchNormCore*> BatchNormCore::instances;

BatchNormCore::BatchNormCore(string name, cublasHandle_t& handle, int batchSize, int size, int seqLength)
		: Core(name, handle, batchSize, size, seqLength) {
	BatchNormCore::addInstance(this);
	epsilon = 1e-8f;
}

BatchNormCore::~BatchNormCore() {
}

int BatchNormCore::getNParams() {
	return size * 2;
}

void BatchNormCore::setParamOffset(int offset) {
	Core::setParamOffset(offset);
	stdOffset = offset + size;
	cout << name << endl;
	cout << "paramOffset: " << paramOffset << endl;
	cout << "stdOffset: " << stdOffset << endl << endl;
}

void BatchNormCore::initMem(bool training, bool optHF) {
	int arraySize = size * batchSize * sizeof(dtype2);
	int arraySizeSingle = size * sizeof(dtype2);
    checkCudaErrors(cudaMalloc((void **)&mean, arraySizeSingle));
    checkCudaErrors(cudaMalloc((void **)&var, arraySizeSingle));

    if (training) {
    	xhats = new dtype2*[seqLength];
    	inv_vars = new dtype2*[seqLength];
    	for (int s = 0; s < seqLength; s++) {
    	    checkCudaErrors(cudaMalloc((void **)&xhats[s], arraySize));
    	    checkCudaErrors(cudaMalloc((void **)&inv_vars[s], arraySizeSingle));
    	}

	    checkCudaErrors(cudaMalloc((void **)&intermed1, arraySize));
	    checkCudaErrors(cudaMalloc((void **)&intermed2, arraySize));
	    checkCudaErrors(cudaMalloc((void **)&single1, arraySizeSingle));
    }
}

void BatchNormCore::freeMem(bool training) {
	checkCudaErrors(cudaFree(mean));
	checkCudaErrors(cudaFree(var));

	if (training) {
		for (int s = 0; s < seqLength; s++) {
	        checkCudaErrors(cudaFree(xhats[s]));
	        checkCudaErrors(cudaFree(inv_vars[s]));
		}
		delete [] xhats;
		delete [] inv_vars;
	}
}

void BatchNormCore::initWeights(dtype1* params, WeightInit& ffWeightInit,
		WeightInit& recWeightInit) {
	int arraySize = size * sizeof(dtype1);
	checkCudaErrors(cudaMemset(params + paramOffset, 0, arraySize));
	CudaFunc::fill1(params + stdOffset, size, 1.0f);
}

int BatchNormCore::calcN(unsigned int* h_inputLengths, int s) {
	int n = 0;
	for (int i = 0; i < batchSize; i++) {
		if (h_inputLengths[i] > s) {
			n++;
		}
	}
	return n;
}

void BatchNormCore::forward(dtype2* prevAct, dtype2* act, dtype2* params, unsigned int* h_inputLengths,
		unsigned int* d_inputLengths, bool deriv, dtype2* priorAct, int s) {
	dtype2* targetMean = params + paramOffset;
	dtype2* targetStd = params + stdOffset;
	int arraySize = size * batchSize * sizeof(dtype2);
	int arraySizeSingle = size * sizeof(dtype2);

	bool printDebug = (s == print_s);

	int n = calcN(h_inputLengths, s);

	if (printDebug) {
		cout << name << " forward" << endl;
		cout << "s: " << s << endl;
		cout << "n: " << n << endl;
	}

	if (n < 2) {
		checkCudaErrors(cudaMemcpy(act, prevAct, arraySize, cudaMemcpyDeviceToDevice));
		return;
	}

	if (printDebug) {
		Network::printStatsGpu("prevAct", prevAct, size * batchSize);
		cout << "before" << endl;
		Network::printAllGpu(prevAct, size * batchSize);
	}

	//calc mean
	checkCudaErrors(cudaMemset(mean, 0, arraySizeSingle));
	CudaFunc::sum_cols_reduce4(prevAct, mean, batchSize, size);
	dtypeh a = 1.0f / n;
	CublasFunc::scal(handle, size, &a, mean, 1);

	if (printDebug) {
		cout << "mean" << endl;
		Network::printAllGpu(mean, size);
	}

	//update avg_mean

	//calc var
	checkCudaErrors(cudaMemset(var, 0, arraySizeSingle));
	checkCudaErrors(cudaMemcpy(act, prevAct, arraySize, cudaMemcpyDeviceToDevice));

	CublasFunc::scal(handle, size, &minus_one, mean, 1);
	CudaFunc::addRowVecMat(act, mean, batchSize, size, batchSize);
	CudaFunc::maskByLength(act, d_inputLengths, s, size, batchSize);

	CudaFunc::isquare(act, size * batchSize);
	CudaFunc::sum_cols_reduce4(act, var, batchSize, size);
	a = 1.0f / n;
	CublasFunc::scal(handle, size, &a, var, 1);

	if (printDebug) {
		cout << "var" << endl;
		Network::printAllGpu(var, size);
	}

	//update avg_var

	CudaFunc::iadd(var, size, epsilon);
	CudaFunc::isqrt(var, size);
	CudaFunc::iinvert(var, size);

	if (printDebug) {
		cout << "inv var" << endl;
		Network::printAllGpu(var, size);
	}

	if (deriv) checkCudaErrors(cudaMemcpy(inv_vars[s], var, arraySizeSingle, cudaMemcpyDeviceToDevice));

	//transform activation
	checkCudaErrors(cudaMemcpy(act, prevAct, arraySize, cudaMemcpyDeviceToDevice));
	CudaFunc::addRowVecMat(act, mean, batchSize, size, batchSize);
	CudaFunc::multRowVecMat(act, var, batchSize, size, batchSize);
	CudaFunc::maskByLength(act, d_inputLengths, s, size, batchSize);

	if (printDebug) {
		cout << "normalized" << endl;
		Network::printAllGpu(act, size * batchSize);
	}

	if (deriv) checkCudaErrors(cudaMemcpy(xhats[s], act, arraySize, cudaMemcpyDeviceToDevice));

	CudaFunc::multRowVecMat(act, targetStd, batchSize, size, batchSize);
	CudaFunc::addRowVecMat(act, targetMean, batchSize, size, batchSize);
	CudaFunc::maskByLength(act, d_inputLengths, s, size, batchSize);

	if (printDebug) {
		cout << "after" << endl;
		Network::printAllGpu(act, size * batchSize);
	}
}

void BatchNormCore::calcGrad(dtype2 *prevAct, dtype2 *act, dtype2 *prevError, dtype2 *error,
		dtype2 *grad, dtype2 *params, unsigned int *h_inputLengths,
		unsigned int *d_inputLengths, dtype2 *priorAct, dtype2 *priorError, int s) {
	int arraySize = size * batchSize * sizeof(dtype2);
	int arraySizeSingle = size * sizeof(dtype2);
	dtype2* targetStd = params + stdOffset;
	dtype2* targetMeanGrad = grad + paramOffset;
	dtype2* targetStdGrad = grad + stdOffset;

	bool printDebug = (s == print_s);

	int n = calcN(h_inputLengths, s);

	dtype2* xhat = xhats[s];

	if (printDebug) {
		cout << name << " calcGrad" << endl;
		cout << "s: " << s << endl;
		cout << "n: " << n << endl;
	}

	if (n < 2) {
		checkCudaErrors(cudaMemcpy(prevError, error, arraySize, cudaMemcpyDeviceToDevice));
		return;
	}

	if (printDebug) {
		cout << "error" << endl;
		Network::printAllGpu(error, size * batchSize);
	}

	if (printDebug) {
		cout << "xhat" << endl;
		Network::printAllGpu(xhat, size * batchSize);
	}

	//calc grad of target_mean
	CudaFunc::sum_cols_reduce4(error, targetMeanGrad, batchSize, size);

	//calc grad of target_std
	CudaFunc::multiply(xhat, error, intermed1, batchSize * size);
	CudaFunc::sum_cols_reduce4(intermed1, targetStdGrad, batchSize, size);

	// backpropagate error
	checkCudaErrors(cudaMemcpy(intermed1, error, arraySize, cudaMemcpyDeviceToDevice));
	CudaFunc::multRowVecMat(intermed1, targetStd, batchSize, size, batchSize);
	dtype2* dxhat = intermed1;

	if (printDebug) {
		cout << "dxhat" << endl;
		Network::printAllGpu(dxhat, size * batchSize);
	}

	CudaFunc::multiply(dxhat, xhat, intermed2, batchSize * size);

	if (printDebug) {
		cout << "intermed2_1" << endl;
		Network::printAllGpu(intermed2, size * batchSize);
	}

	checkCudaErrors(cudaMemset(single1, 0, arraySizeSingle));
	CudaFunc::sum_cols_reduce4(intermed2, single1, batchSize, size);

	if (printDebug) {
		cout << "single1_1" << endl;
		Network::printAllGpu(single1, size);
	}

	checkCudaErrors(cudaMemcpy(intermed2, xhat, arraySize, cudaMemcpyDeviceToDevice));
	CudaFunc::multRowVecMat(intermed2, single1, batchSize, size, batchSize);

	if (printDebug) {
		cout << "intermed2_2" << endl;
		Network::printAllGpu(intermed2, size * batchSize);
	}

	checkCudaErrors(cudaMemset(single1, 0, arraySizeSingle));
	CudaFunc::sum_cols_reduce4(dxhat, single1, batchSize, size);

	if (printDebug) {
		cout << "single1_2" << endl;
		Network::printAllGpu(single1, size);
	}

	CudaFunc::addRowVecMat(intermed2, single1, batchSize, size, batchSize);
	CudaFunc::maskByLength(intermed2, d_inputLengths, s, size, batchSize);

	if (printDebug) {
		cout << "intermed2_3" << endl;
		Network::printAllGpu(intermed2, size * batchSize);
	}

	dtypeh a = n;
	CublasFunc::scal(handle, size * batchSize, &a, dxhat, 1);
	if (printDebug) {
		cout << "N*dxhat" << endl;
		Network::printAllGpu(dxhat, size * batchSize);
	}

	CublasFunc::axpy(handle, batchSize * size, &minus_one, intermed2, 1, intermed1, 1);

	if (printDebug) {
		cout << "N*dxhat-intermed2_3" << endl;
		Network::printAllGpu(intermed1, size * batchSize);
	}


	dtype2* inv_var = inv_vars[s];

	if (printDebug) {
		cout << "inv_var" << endl;
		Network::printAllGpu(inv_var, size);
	}

	a = 1.0f / n;
	CublasFunc::scal(handle, size, &a, inv_var, 1);
	CudaFunc::multRowVecMat(intermed1, inv_var, batchSize, size, batchSize);

	if (printDebug) {
		cout << "error" << endl;
		Network::printAllGpu(intermed1, size * batchSize);
	}

	CublasFunc::axpy(handle, batchSize * size, &one, intermed1, 1, prevError, 1);
}

void BatchNormCore::saveState(ofstream& file) {
}

void BatchNormCore::loadState(ifstream& file) {
}

void BatchNormCore::saveStates(int iterNum) {
	static char buf[100];
	sprintf(buf, "encdeccu_bnStates_iter%d.bin", iterNum);
	doSaveStates(buf);
}

void BatchNormCore::saveStatesTemp() {
	const char* filename = "encdeccu_bnStates_save.bin";
	doSaveStates(filename);
}

void BatchNormCore::doSaveStates(const char* filename) {
	ofstream file(filename, ios::binary | ios::trunc);

	map<string, BatchNormCore*>::iterator iter;
	for (iter = instances.begin(); iter != instances.end(); iter++) {
		BatchNormCore* instance = iter->second;
		instance->saveState(file);
	}

	file.close();
}

void BatchNormCore::loadStates(string filename) {
	ifstream file(filename.c_str(), ios::binary);

	map<string, BatchNormCore*>::iterator iter;
	for (iter = instances.begin(); iter != instances.end(); iter++) {
		BatchNormCore* instance = iter->second;
		instance->loadState(file);
	}

	file.close();
}

void BatchNormCore::addInstance(BatchNormCore* instance) {
	string name = instance->name;

	map<string, BatchNormCore*>::iterator iter;
	iter = instances.find(name);
	if (iter != instances.end()) {
		cerr << "Duplicate BatchNameCore name: " << name << endl;
		exit(1);
	}

	instances.insert(pair<string, BatchNormCore*>(name, instance));
}

} /* namespace netlib */
