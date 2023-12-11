/**
 * @file
 * @brief Declares the WordVectors class, which wraps the Word2Vec class and adds functionality useful for sentence encoding,
 * such as special vectors.
 *
 */

#ifndef WORDVECTORS_H_
#define WORDVECTORS_H_

#include <string>
#include <map>
#include "Word2Vec.h"

#define WV_LENGTH 300

namespace netlib {

/**
 * @brief Wraps the Word2Vec class and adds functionality useful for sentence encoding, such as special vectors.
 *
 */
class WordVectors {
public:
	WordVectors(cublasHandle_t& handle, bool dictGpu1, int batchSize);
	virtual ~WordVectors();
	dtype2* lookup(string label, int* ix = NULL);
	void freeVector(dtype2* vec);
	int indexOf(string label);
	pair<string,dtypeh> nearest(dtype2* vec);
	string* nearestBatch(dtype2* output, int batchSize);
//	string* nearestBatchLS(dtype2* output, int batchSize, unsigned int* h_inputLengths, int s);
	void freeTemp();
	int* nearestBatchIx(dtype2* output);
	void nearestBatchMask(dtype2* output, int* ixTarget, unsigned int *h_inputLengths, int s,
			dtype2* mask, unsigned int *d_inputLengths);
	dtypeh cosineSim(dtype2* vec1, dtype2* vec2);
	void initDictGpu(cublasHandle_t& handle);
	dtype2* getDictGpu();
	int getNumWords();
	void deviceNormalize(dtype2* vec);
	int wvLength = WV_LENGTH;
	float* UNKNOWN_VECTOR;
	float* EOS_VECTOR;
	float* COMMA_VECTOR;

private:
	Word2Vec w2v;
	map<string,float*> specialVectors;
	void initSpecialVectors();
	void saveSpecialVectors();
	void saveSpecialVectorsNew();
	void loadSpecialVectors(ifstream &svFile);
	void loadStats();
	float* randVec();
	void saveVec(string label, float* vector, ofstream &svFile);
	void writeString(string s, ofstream &svFile);
	string readString(ifstream &svFile);
	void writeFloat(float f, ofstream &svFile);
	float readFloat(ifstream &svFile);
	string modelFilename;
	string wordsFilename;
	string specialVectorsFilename;
	string specialVectorsNewFilename;
	string statsFilename;
	dtypeh *wvMean = NULL;
	dtypeh *wvStd = NULL;
};

} /* namespace netlib */

#endif /* WORDVECTORS_H_ */
