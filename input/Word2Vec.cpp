/**
 * @file
 * @brief Defines Word2Vec class, which wraps a Word2Vec dictionary and adds useful methods.
 *
 */

#include "Word2Vec.h"

#include "../gpu/CudaFunc.h"
#include "../Network.h"

#define WV_LENGTH 300
#define EPS 1e-9

extern void printAll(const char* label, dtype2* a, unsigned int size);

namespace netlib {

Word2Vec::Word2Vec(cublasHandle_t& handle, bool dictGpu1, int batchSize) : handle(handle), dictGpu1(dictGpu1), batchSize(batchSize) {
	words = new string[NUM_WORDS_MAX];
	simMatBatch = NULL;
//	simMatBatchTestLS = NULL;
//	simMatBatchTestErrs = NULL;
	ixMatch = NULL;
	useGpu = false;
	numWords = 0;
}

Word2Vec::~Word2Vec() {
	delete [] words;

	if (dictGpu1) {
		checkCudaErrors(cudaSetDevice(1));
	}

	if (useGpu) {
		checkCudaErrors(cudaFree(dictGpu));
	}

	if (wvNorms != NULL) {
		checkCudaErrors(cudaFree(wvNorms));
	}

	if (simMatBatch != NULL) {
		checkCudaErrors(cudaFree(simMatBatch));
	}

	if (hSimMat != NULL) {
		free(hSimMat);
	}

	if (ixMatch != NULL) {
		checkCudaErrors(cudaFree(ixMatch));
	}

	if (d1_output != NULL) {
		checkCudaErrors(cudaFree(d1_output));
	}

	checkCudaErrors(cudaSetDevice(0));
}

void Word2Vec::init(string modelFilename, string wordsFilename, bool labelsOnly, int numSpecial,
		dtypeh* mean, dtypeh* std) {
	set<string> corpusWords;
	bool haveCorpusWords = false;
	if (!wordsFilename.empty())
	{
		haveCorpusWords = true;

		ifstream wordsFile(wordsFilename);
		if (!wordsFile)
		{
			cerr << "Words file " << wordsFilename << " could not be opened for reading." << endl;
			exit(1);
		}

		while (wordsFile) {
			string word;
			wordsFile >> word;
			boost::trim(word);
			corpusWords.insert(word);
		}

		wordsFile.close();

		cout << "num corpus words: " << corpusWords.size() << endl;
	}

	wvMean = mean;
	wvStd = std;
	haveWordStats = wvMean != NULL && wvStd != NULL;
	if (haveWordStats) cout << "have word stats" << endl;

	ifstream modelFile(modelFilename, ios::binary);
	if (!modelFile)
	{
		cerr << "Model file " << modelFilename << " could not be opened for reading." << endl;
		exit(1);
	}

	string strNumWords, strVectorLength;

	modelFile >> strNumWords;
	modelFile >> strVectorLength;

	int numWords = atoi(strNumWords.c_str());
	int vectorLength = atoi(strVectorLength.c_str());

	cout << "Model contains " << numWords << " words." << endl;
	cout << vectorLength << " dimensions" << endl;

	size_t hMatArraySize = NUM_WORDS_MAX * vectorLength * sizeof(dtype3);

	if (!labelsOnly)
	{
		h_mat = (dtype3*) malloc(hMatArraySize);
	}

	int numLoaded = 0;
	int numAllZero = 0;

	dtypeh vec[vectorLength];

#ifdef DEBUG
	numWords = 1000000;
#endif
	for (int i = 0; i < numWords; i++)
	{
		string label;
		float fValue;

		modelFile >> label;
		modelFile.ignore(1);
		boost::trim(label);
		if ((i % 100000 == 0) || (i == numWords-1))
		{
			cout << "Loading " << label << " from row " << i << endl;
		}

		bool allZero = true;
		if (labelsOnly)
		{
			modelFile.ignore(sizeof(float) * WV_LENGTH);
		}
		else
		{
			for (int j = 0; j < vectorLength; j++)
			{
				fValue = readFloat(modelFile);
				if (allZero && (abs(fValue) > 1e-9)) {
					allZero = false;
				}
				vec[j] = float2h(fValue);
			}
		}

		if (haveCorpusWords && corpusWords.count(label) == 0)
		{
			continue;
		}

		if (allZero)
		{
			cout << "###################" << label << " has zero length" << endl;
			numAllZero++;
			continue;
		}

		if (label == "EOS") {
//			cout << "changing EOS to EOS_" << endl;
			label = "EOS_";
		} else if (label == "UNK") {
//			cout << "changing UNK to UNK_" << endl;
			label = "UNK_";
		}

		if (!labelsOnly) {
			if (numLoaded >= NUM_WORDS_MAX) {
				cout << "Too many words loaded." << endl;
				exit(1);
			}

//#ifndef DEBUG
			if (vecUnitNorm) normalize(vec);
			if (haveWordStats) whiten(vec, wvMean, wvStd);
//#endif
			for (int k = 0; k < vectorLength; k++)
			{
				h_mat[IDX2(numLoaded, k, NUM_WORDS_MAX)] = h23(vec[k]);
			}
		}

		words[numLoaded] = label;
		numLoaded++;
	}

	modelFile.close();

	this->numWords = numLoaded;

	for (int i = 0; i < numLoaded; i++)
	{
		string word = words[i];
		index.insert(pair<string,int>(word, i));
	}

	cout << numAllZero << " had zero length" << endl;
	cout << numLoaded << " loaded" << endl;
}

float Word2Vec::readFloat(ifstream &f) {
	float fval;
	f.read((char *)&fval, 4);
	return fval;
}

dtype2* Word2Vec::lookup(string word, int* ix) {
	map<string,int>::iterator it = index.find(word);
	if (it == index.end()) {
		if (ix != NULL) {
			*ix = -1;
		}
		return NULL;
	}

	int i = it->second;

	if (ix != NULL) {
		*ix = i;
	}

	return lookupGpu(i);
}

dtype2* Word2Vec::lookupGpu(int ix) {
	if (dictGpu1) {
		checkCudaErrors(cudaSetDevice(1));
	}

	dtype2 *d_vec;
	int arraySize = WV_LENGTH * sizeof(dtype2);
	checkCudaErrors(cudaMalloc((void **)&d_vec, arraySize));

	CublasFunc::copy(handle, WV_LENGTH, dictGpu+IDX2(ix,0,numWords), numWords, d_vec, 1);
	checkCudaErrors(cudaSetDevice(0));
	return d_vec;
}

void Word2Vec::freeVector(dtype2* vec) {
	checkCudaErrors(cudaFree(vec));
}

int Word2Vec::indexOf(string word) {
	map<string,int>::iterator it = index.find(word);
	if (it == index.end()) {
		return -1;
	}

	return it->second;
}

WordSim Word2Vec::nearest(dtype2* vec) {
	if (dictGpu1) {
		checkCudaErrors(cudaSetDevice(1));
	}

	initSimMatBatch();
	if (!vecUnitNorm) initWvNorms();

	CublasFunc::gemvUp(handle, CUBLAS_OP_N, numWords, WV_LENGTH, &one, dictGpu, numWords, vec, 1, &zero, simMatBatch, 1);

	if (!vecUnitNorm) CudaFunc::multiplyUp2(simMatBatch, wvNorms, simMatBatch, numWords);

	int ix;
	CublasFunc::iamaxUp(handle, numWords, simMatBatch, 1, &ix);
	ix--;

	dtypeup sim_up;
	checkCudaErrors(cudaMemcpy(&sim_up, simMatBatch+ix, sizeof(dtypeup), cudaMemcpyDeviceToHost));
	dtypeh sim = dup2h(sim_up);

	string word = words[ix];

	checkCudaErrors(cudaSetDevice(0));
	return WordSim(word, sim);
}

WordSimVec* Word2Vec::topN(string word, int n) {
	dtype2* vec = lookup(word);
	if (vec == NULL) {
		return NULL;
	}

	WordSimVec* results = topN(vec, n);
	freeVector(vec);
	return results;

//	int inputIx = indexOf(word);
//	if (inputIx == -1) {
//		return NULL;
//	}
//
//	dtype3* vec_d = dictGpu+IDX2(inputIx,0,numWords);
//
//	dtype3* vec =
//	CublasFunc::copy(handle,  WV_LENGTH, vec_d, numWords, vec, 1);
}

WordSimVec* Word2Vec::topN(dtype2* vec_d, int n) {
	initSimMat();

	if (!vecUnitNorm) initWvNorms();

	dtype2 inputNorm;
	CublasFunc::nrm2(handle,  WV_LENGTH, vec_d, 1, &inputNorm);

	CublasFunc::gemvUp(handle, CUBLAS_OP_N, numWords, WV_LENGTH, &one, dictGpu, numWords, vec_d, 1, &zero, simMat, 1);

	if (!vecUnitNorm) CudaFunc::multiplyUp2(simMat, wvNorms, simMat, numWords);

	WordSimVec* results = new WordSimVec();

	int ix;
	dtypeup sim;

	for (int i = 0; i < n; ) {
		CublasFunc::iamaxUp(handle, numWords, simMat, 1, &ix);
		ix--;

//		if (ix != inputIx) {
			checkCudaErrors(cudaMemcpy(&sim, simMat+ix, sizeof(dtypeup), cudaMemcpyDeviceToHost));
			sim /= inputNorm;

			string word = words[ix];
			results->push_back(WordSim(word, sim));
			i++;
//		}
		checkCudaErrors(cudaMemset((void *)(simMat+ix), 0, sizeof(dtypeup)));
	}

	return results;
}

void Word2Vec::replace(string word, dtype2* vec) {
	int ix = indexOf(word);
	if (ix == -1) {
		cerr << "Word2Vec::replace; not found: " << word << endl;
		return;
	}
	dtype3* target = dictGpu+IDX2(ix,0,numWords);
	CublasFunc::copy(handle, WV_LENGTH, vec, 1, target, numWords);
}

void Word2Vec::initWvNorms() {
	if (wvNorms == NULL) {
		int arraySize = numWords * sizeof(float);
		checkCudaErrors(cudaMalloc((void **)&wvNorms, arraySize));

		calcWvNorms();
	}
}

void Word2Vec::calcWvNorms() {
	cublasPointerMode_t origMode;
	CublasFunc::getPointerMode(handle, &origMode);
	CublasFunc::setPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);

	for (int i = 0; i < numWords; i++) {
		CublasFunc::nrm2(handle, WV_LENGTH, dictGpu+IDX2(i, 0, numWords), numWords, wvNorms+i);
	}

	CublasFunc::setPointerMode(handle, origMode);

	CudaFunc::iinvert(wvNorms, numWords);
}

void Word2Vec::initSimMatBatch() {
	if (simMatBatch == NULL) {
		int arraySize = numWords * batchSize * sizeof(dtypeup);
		checkCudaErrors(cudaMalloc((void **)&simMatBatch, arraySize));
	}
}

void Word2Vec::initSimMat() {
	if (simMat == NULL) {
		int arraySize = numWords * sizeof(dtypeup);
		checkCudaErrors(cudaMalloc((void **)&simMat, arraySize));
	}
}

#ifndef DEBUG
string* Word2Vec::nearestBatch(dtype2* output, int batchSize) {
	if (dictGpu1) {
		checkCudaErrors(cudaSetDevice(1));
		unsigned int arraySize = batchSize * WV_LENGTH * sizeof(dtype2);
		checkCudaErrors(cudaMemcpyPeer(d1_output, 1, output, 0, arraySize));
		output = d1_output;
	}

	initSimMatBatch();
	if (!vecUnitNorm) initWvNorms();

	dtype2* normOutput;
	int arraySize = batchSize * WV_LENGTH * sizeof(dtype2);
	checkCudaErrors(cudaMalloc((void **)&normOutput, arraySize));
	checkCudaErrors(cudaMemset((void *)normOutput, 0, arraySize));
	for (int i = 0; i < batchSize; i++) {
		dtype2 norm_d;
		CublasFunc::nrm2(handle, WV_LENGTH, output + IDX2(i,0,batchSize), batchSize, &norm_d);
		dtypeh norm = d2h(norm_d);
		dtypeh a = 1/norm;
		CublasFunc::axpy(handle, WV_LENGTH, &a, output + IDX2(i,0,batchSize), batchSize, normOutput + IDX2(i,0,batchSize),
				batchSize);
	}

	CublasFunc::gemmUp(handle, CUBLAS_OP_N, CUBLAS_OP_T, numWords, batchSize, WV_LENGTH, &one, dictGpu, numWords,
			normOutput, batchSize, &zero, simMatBatch, numWords);

	if (!vecUnitNorm) CudaFunc::multColVecMatUp(simMatBatch, wvNorms, numWords, batchSize, numWords);

	string* matches = new string[batchSize];
	for (int i = 0; i < batchSize; i++) {
		int ix;
		CublasFunc::iamaxUp(handle, numWords, simMatBatch + IDX2(0,i,numWords), 1, &ix);
		matches[i] = words[ix-1];
	}

	checkCudaErrors(cudaFree(normOutput));

	checkCudaErrors(cudaSetDevice(0));
	return matches;
}
#endif

//#ifndef DEBUG
//string* Word2Vec::nearestBatchLS(dtype2* output, int batchSize, unsigned int* h_inputLengths, int s) {
////	cout << "Word2Vec::nearestBatchLS" << endl;
//
//	int arraySize = numWords * WV_LENGTH * sizeof(float);
//	if (simMatBatchTestLS == NULL) {
//		checkCudaErrors(cudaMalloc((void **)&simMatBatchTestLS, arraySize));
//	}
//	if (simMatBatchTestErrs == NULL) {
//		checkCudaErrors(cudaMalloc((void **)&simMatBatchTestErrs, numWords * sizeof(float)));
//	}
//	dtype2* negOutput;
//	checkCudaErrors(cudaMalloc((void **)&negOutput, WV_LENGTH * sizeof(float)));
//
//	string* matches = new string[batchSize];
//	for (int i = 0; i < batchSize; i++) {
//		int sentLen = h_inputLengths[i];
//		if (s >= sentLen) {
//			matches[i] = "";
//			continue;
//		}
//
//		checkCudaErrors(cudaMemcpy(simMatBatchTestLS, dictGpu, arraySize, cudaMemcpyDeviceToDevice));
//		checkCudaErrors(cudaMemset((void *)negOutput, 0,  WV_LENGTH * sizeof(float)));
////		checkCudaErrors(cudaMemset((void *)simMatBatchTestErrs, 0,  numWords * sizeof(float)));
//
//		CublasFunc::axpy(handle, WV_LENGTH, &minus_one, output + IDX2(i,0,batchSize), batchSize, negOutput, 1);
//		CudaFunc::addRowVecMat(simMatBatchTestLS, negOutput, numWords, WV_LENGTH, numWords);
//		CudaFunc::isquare(simMatBatchTestLS, numWords*WV_LENGTH);
////		CudaFunc::sum_rows_reduce4(simMatBatchTestLS, simMatBatchTestErrs, numWords, WV_LENGTH);
//		cpuSumRows(simMatBatchTestLS, simMatBatchTestErrs, numWords, WV_LENGTH);
//
//		int ix;
//		CublasFunc::iamin(handle, numWords, simMatBatchTestErrs, 1, &ix);
//		matches[i] = words[ix-1];
//
//		if (i == 0) {
//			dtype2 err;
//			checkCudaErrors(cudaMemcpy(&err, simMatBatchTestErrs+(ix-1), sizeof(float), cudaMemcpyDeviceToHost));
//			cout << err << endl;
//		}
//	}
//
//	checkCudaErrors(cudaFree(negOutput));
//
//	return matches;
//}
//#endif
//
//#ifndef DEBUG
//void Word2Vec::cpuSumRows(dtype2* d_mat, dtype2* d_vec, int rows, int cols) {
//	int arraySize = rows * cols * sizeof(float);
//	if (hSimMat == NULL) {
//		hSimMat = (float*) malloc(arraySize);
//	}
//
//	int vecArraySize = rows * sizeof(float);
//	float* h_vec = (float*) malloc(vecArraySize);
//
//	checkCudaErrors(cudaMemcpy(hSimMat, d_mat, arraySize, cudaMemcpyDeviceToHost));
//
//	for (int i = 0; i < rows; i++) {
//		float sum = 0.0f;
//		for (int j = 0; j < cols; j++) {
//			int ix = IDX2(i,j,rows);
//			sum += hSimMat[ix];
//		}
//		h_vec[i] = sum;
//	}
//
//	checkCudaErrors(cudaMemcpy(d_vec, h_vec, vecArraySize, cudaMemcpyHostToDevice));
//	free(h_vec);
//}
//#endif

void Word2Vec::freeTemp() {
}

int* Word2Vec::nearestBatchIx(dtype2* output) {
	if (simMatBatch == NULL) {
		int arraySize = numWords * batchSize * sizeof(dtype2);
		checkCudaErrors(cudaMalloc((void **)&simMatBatch, arraySize));
	}

	dtype2* normOutput;
	int arraySize = batchSize * WV_LENGTH * sizeof(dtype2);
	checkCudaErrors(cudaMalloc((void **)&normOutput, arraySize));
	checkCudaErrors(cudaMemset((void *)normOutput, 0, arraySize));
	for (int i = 0; i < batchSize; i++) {
		dtype2 norm_d;
		CublasFunc::nrm2(handle, WV_LENGTH, output + IDX2(i,0,batchSize), batchSize, &norm_d);
		dtypeh norm = d2h(norm_d);
		dtypeh a = 1/norm;
		CublasFunc::axpy(handle, WV_LENGTH, &a, output + IDX2(i,0,batchSize), batchSize, normOutput + IDX2(i,0,batchSize),
				batchSize);
	}

	CublasFunc::gemmUp(handle, CUBLAS_OP_N, CUBLAS_OP_T, numWords, batchSize, WV_LENGTH, &one, dictGpu, numWords,
			normOutput, batchSize, &zero, simMatBatch, numWords);

	int* ixs = new int[batchSize];
	for (int i = 0; i < batchSize; i++) {
		int ix;
		CublasFunc::iamaxUp(handle, numWords, simMatBatch + IDX2(0,i,numWords), 1, &ix);
		ixs[i] = ix-1;
	}

	checkCudaErrors(cudaFree(normOutput));

	return ixs;
}

#ifndef DEBUG
void Word2Vec::nearestBatchMask(dtype2* output, int* ixTarget, unsigned int *h_inputLengths,
		int s, dtype2* mask, unsigned int *d_inputLengths) {
//	cout << "nearestBatchMask" << endl;
	if (dictGpu1) {
		checkCudaErrors(cudaSetDevice(1));
		unsigned int arraySize = batchSize * WV_LENGTH * sizeof(dtype2);
		checkCudaErrors(cudaMemcpyPeer(d1_output, 1, output, 0, arraySize));
		output = d1_output;
	}

	if (simMatBatch == NULL) {
		int arraySize = numWords * batchSize * sizeof(dtypeup);
		checkCudaErrors(cudaMalloc((void **)&simMatBatch, arraySize));
	}

	if (ixMatch == NULL) {
		int arraySize = batchSize * sizeof(int);
		checkCudaErrors(cudaMalloc((void **)&ixMatch, arraySize));
	}

	if (!vecUnitNorm) initWvNorms();

	dtype2* normOutput;
	int arraySize = batchSize * WV_LENGTH * sizeof(dtype2);
	checkCudaErrors(cudaMalloc((void **)&normOutput, arraySize));
	checkCudaErrors(cudaMemset((void *)normOutput, 0, arraySize));
	for (int i = 0; i < batchSize; i++) {
		if (s >= h_inputLengths[i]) {
			continue;
		}
		dtype2 norm_d;
		CublasFunc::nrm2(handle, WV_LENGTH, output + IDX2(i,0,batchSize), batchSize, &norm_d);
		dtypeh norm = d2h(norm_d);
		dtypeh a = 1/norm;
		CublasFunc::axpy(handle, WV_LENGTH, &a, output + IDX2(i,0,batchSize), batchSize, normOutput + IDX2(i,0,batchSize),
				batchSize);
	}

	CublasFunc::gemmUp(handle, CUBLAS_OP_N, CUBLAS_OP_T, numWords, batchSize, WV_LENGTH, &one, dictGpu, numWords,
			normOutput, batchSize, &zero, simMatBatch, numWords);

	if (!vecUnitNorm) CudaFunc::multColVecMatUp(simMatBatch, wvNorms, numWords, batchSize, numWords);

	cublasPointerMode_t origMode;
	CublasFunc::getPointerMode(handle, &origMode);
	CublasFunc::setPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);

	for (int i = 0; i < batchSize; i++) {
		if (s >= h_inputLengths[i]) {
			continue;
		}
		CublasFunc::iamaxUp(handle, numWords, simMatBatch + IDX2(0,i,numWords), 1, ixMatch+i);
	}

	CublasFunc::setPointerMode(handle, origMode);

	CudaFunc::matchMask(ixMatch, ixTarget, mask, batchSize, d_inputLengths, s);

	checkCudaErrors(cudaFree(normOutput));
	checkCudaErrors(cudaSetDevice(0));
//	cout << "nearestBatchMask end" << endl;
}
#endif

dtype2* Word2Vec::getDictGpu() {
	return dictGpu;
}

int Word2Vec::getNumWords() {
	return numWords;
}

void Word2Vec::normalize(dtypeh* in) {
	float n = norm(in);

	if (n < EPS) {
		return;
	}

	for (int i = 0; i < WV_LENGTH; i++)
	{
		in[i] /= n;
	}
}

void Word2Vec::deviceNormalize(dtype2* vec) {
	dtype2 norm_d;
	CublasFunc::nrm2(handle, WV_LENGTH, vec, 1, &norm_d);
	dtypeh norm = d2h(norm_d);
	dtypeh a = 1.0 / norm;
	CublasFunc::scal(handle, WV_LENGTH, &a, vec, 1);
}

dtypeh Word2Vec::cosineSim(dtype2* vec1, dtype2* vec2) {
	dtypeh eps = 1e-8;

	dtype2 norm1_d;
	CublasFunc::nrm2(handle, WV_LENGTH, vec1, 1, &norm1_d);
	dtypeh norm1 = d2h(norm1_d);
	norm1 = max(norm1, eps);

	dtype2 norm2_d;
	CublasFunc::nrm2(handle, WV_LENGTH, vec2, 1, &norm2_d);
	dtypeh norm2 = d2h(norm2_d);
	norm2 = max(norm2, eps);

	dtypeh sim;
	CublasFunc::dot(handle, WV_LENGTH, vec1, 1, vec2, 1, &sim);
	sim /= (norm1*norm2);

	return sim;
}

dtypeh Word2Vec::norm(dtypeh* vec) {
	dtypeh accum = 0;
	for (int i = 0; i < WV_LENGTH; i++)
	{
		dtypeh elem = vec[i];
		accum += elem*elem;
	}

	return sqrt(accum);
}

void Word2Vec::initDictGpu(cublasHandle_t& handle, map<string,float*>* specialVectors) {
	cout << endl << "initializing dict on GPU" << endl;

	this->useGpu = true;
	this->handle = handle;

	int numWordsMat = numWords;

	if (specialVectors != NULL) {
		map<string,float*>::iterator it;

		for (it = specialVectors->begin(); it != specialVectors->end(); it++) {
			string label = it->first;

			if (indexOf(label) != -1) {
				cout << "Already had entry for " << label << endl;
			}

			words[numWords] = label;
			index.insert(pair<string,int>(label, numWords));

			float* sv = it->second;
			for (int j = 0; j < WV_LENGTH; j++) {
				h_mat[IDX2(numWords,j,NUM_WORDS_MAX)] = float2d3(sv[j]);
			}

			numWords++;
		}
	}

	unsigned int arraySize = WV_LENGTH * numWords * sizeof(dtype3);
	cout << "arraySize: " << arraySize << endl;

	if (dictGpu1) {
		checkCudaErrors(cudaSetDevice(1));
		unsigned int arraySize = batchSize * WV_LENGTH * sizeof(dtype2);
		checkCudaErrors(cudaMalloc((void **)&d1_output, arraySize));
	}

	//resize
	checkCudaErrors(cudaMallocHost((void **)&h_dictGpu, arraySize));
	unsigned int colArraySize = numWords * sizeof(dtype3);
	for (int i = 0; i < WV_LENGTH; i++) {
		int matColIx = IDX2(0, i, NUM_WORDS_MAX);
		int dictColIx = IDX2(0, i, numWords);
		memcpy(h_dictGpu+dictColIx, h_mat+matColIx, colArraySize);
	}
	free(h_mat);

	checkCudaErrors(cudaMalloc((void **)&dictGpu, arraySize));
	checkCudaErrors(cudaMemcpy(dictGpu, h_dictGpu, arraySize, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaFreeHost(h_dictGpu));

	checkCudaErrors(cudaSetDevice(0));
}

void Word2Vec::whiten(dtypeh* vec, dtypeh* mean, dtypeh* std) {
	if (!vecWhiten) {
		return;
	}

	for (int i = 0; i < WV_LENGTH; i++) {
		vec[i] -= mean[i];
		vec[i] *= 0.05f/std[i];
	}
}

} /* namespace netlib */

