/**
 * @file
 * @brief Declares Word2Vec class, which wraps a Word2Vec dictionary and adds useful methods.
 *
 */

#ifndef WORD2VEC_H_
#define WORD2VEC_H_

#include "../NetLib.h"
#include "../gpu/CublasFunc.h"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <set>
#include <vector>
#include <map>
#include <utility>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <boost/algorithm/string.hpp>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <helper_cuda.h>

#define NUM_WORDS_MAX 2000000

const bool vecUnitNorm = false;
const bool vecWhiten = true;

using namespace std;

typedef pair<string,dtypeh> WordSim;
typedef vector<WordSim> WordSimVec;

namespace netlib {

/**
 * @brief Wraps a Word2Vec dictionary and adds useful methods.
 *
 */
class Word2Vec {
public:
	Word2Vec(cublasHandle_t& handle, bool dictGpu1, int batchSize);
	virtual ~Word2Vec();
	void init(string modelFilename, string wordsFilename, bool labelsOnly, int numSpecial,
			dtypeh* wvMean, dtypeh* wvStd);
	void initDictGpu(cublasHandle_t& handle, map<string,float*>* specialVectors = NULL);
	dtype2* lookup(string word, int* ix = NULL);
	dtype2* lookupGpu(int ix);
	void freeVector(dtype2* vec);
	int indexOf(string word);
	WordSim nearest(dtype2* vec);
	WordSimVec* topN(string word, int n);
	WordSimVec* topN(dtype2* vec_d, int n);
	string* nearestBatch(dtype2* output, int batchSize);
	void replace(string word, dtype2* vec);
//	string* nearestBatchLS(dtype2* output, int batchSize, unsigned int* h_inputLengths, int s);
//	void cpuSumRows(float* d_mat, float* d_vec, int rows, int cols);
	void freeTemp();
	int* nearestBatchIx(dtype2* output);
	void nearestBatchMask(dtype2* output, int* ixTarget, unsigned int *h_inputLengths, int s,
			dtype2* mask, unsigned int *d_inputLengths);
	dtype2* getDictGpu();
	int getNumWords();
	void initWvNorms();
	void calcWvNorms();
	void initSimMatBatch();
	void initSimMat();
	void deviceNormalize(dtype2* vec);
	dtypeh cosineSim(dtype2* vec1, dtype2* vec2);
	static void normalize(dtypeh* in);
	static dtypeh norm(dtypeh* vec);
	static void whiten(dtypeh* vec, dtypeh* mean, dtypeh* std);

private:
	dtype3* h_mat = NULL;
	dtype3* h_dictGpu = NULL;
	dtype3* dictGpu = NULL;
	int batchSize;
	string *words = NULL;
	dtype2* wvNorms = NULL;
	dtypeup* simMatBatch = NULL;
	dtypeup* simMat = NULL;
//	dtype2* simMatBatchTestLS;
//	dtype2* simMatBatchTestErrs;
	dtypeh* hSimMat = NULL;
	int* ixMatch = NULL;
	cublasHandle_t& handle;

	map<string, int> index;
	int numWords;
	bool useGpu;
	bool dictGpu1;

	dtype2* d1_output = NULL;

	bool haveWordStats = false;
	dtypeh* wvMean = NULL;
	dtypeh* wvStd = NULL;

	static float readFloat(ifstream &f);
};

} /* namespace netlib */

#endif /* WORD2VEC_H_ */
