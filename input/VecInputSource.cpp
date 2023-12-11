/**
 * @file
 * @brief Defines VecInputSource class, which serves vectors as input to a network.
 *
 */

#include "VecInputSource.h"

#include "VecIterator.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>

namespace netlib {

VecInputSource::VecInputSource(VecIterator& corpus, int batchSize, int vecLength, int maxSeqLength, int maxNumBatch,
		ix_type startIx)
					: corpus(corpus), batchSize(batchSize), vecLength(vecLength), maxSeqLength(maxSeqLength),
					  maxNumBatch(maxNumBatch) {
	initMem();
	numBatchSaved = 0;
	firstBatchIx = startIx;
}

VecInputSource::~VecInputSource() {
	freeMem();
}

void VecInputSource::initMem() {
	d_inputs_rev = new dtype2*[1];
	d_targets = new dtype2*[1];
	int arraySize = batchSize * vecLength * sizeof(dtype2);
	checkCudaErrors(cudaMalloc((void **)&d_inputs_rev[0], arraySize));
	checkCudaErrors(cudaMalloc((void **)&d_targets[0], arraySize));

	h_inputLengths = new unsigned int[batchSize];
	for (int i = 0; i < batchSize; i++) {
		h_inputLengths[i] = maxSeqLength;
	}

	d_inputLengths = new unsigned int*[1];
	arraySize = batchSize * sizeof(unsigned int);
	checkCudaErrors(cudaMalloc((void **)&d_inputLengths[0], arraySize));
	checkCudaErrors(cudaMemcpy(d_inputLengths[0], h_inputLengths, arraySize, cudaMemcpyHostToDevice));
}

void VecInputSource::freeMem() {
    checkCudaErrors(cudaFree(d_inputs_rev[0]));
    delete [] d_inputs_rev;
}

void VecInputSource::toFirstBatch() {
	corpus.resetTo(firstBatchIx);
}

bool VecInputSource::hasNextBatchSet() {
	return (firstBatchIx + batchSize * maxNumBatch) <= corpus.size();
}

void VecInputSource::toNextBatchSet() {
	firstBatchIx += (batchSize * maxNumBatch);
	corpus.resetTo(firstBatchIx);
	numBatchSaved = 0;
}

bool VecInputSource::hasNext() {
	return corpus.hasNext();
}

void VecInputSource::nextBatch(int batchNum, vector<string>* tokensRet) {
	corpus.next(d_inputs_rev[0]);
	int arraySize = batchSize * vecLength * sizeof(dtype2);
	checkCudaErrors(cudaMemcpy(d_targets[0], d_inputs_rev[0], arraySize, cudaMemcpyDeviceToDevice));
}

void VecInputSource::next() {
	corpus.next(d_inputs_rev[0]);
	int arraySize = batchSize * vecLength * sizeof(dtype2);
	checkCudaErrors(cudaMemcpy(d_targets[0], d_inputs_rev[0], arraySize, cudaMemcpyDeviceToDevice));
}

dtype2** VecInputSource::getRevInputs() {
	return d_inputs_rev;
}

dtype2** VecInputSource::getFwdInputs() {
	return NULL;
}

unsigned int** VecInputSource::getDInputLengths() {
	return d_inputLengths;
}

unsigned int* VecInputSource::getHInputLengths() {
	return h_inputLengths;
}

dtype2** VecInputSource::getTargets() {
	return d_targets;
}

void VecInputSource::computeMatchMasks(int batchNum, dtype2 **outputs) {
}

dtype2** VecInputSource::matchMasks(int batchNum) {
	return NULL;
}

void VecInputSource::reset() {
	corpus.reset();
	firstBatchIx = 0;
	numBatchSaved = 0;
}

string VecInputSource::getIxFilename() {
	return corpus.getIxFilename();
}

int VecInputSource::getBatchSize() {
	return batchSize;
}

void VecInputSource::shuffle() {
	corpus.shuffle();
}

void VecInputSource::loadIxs(string filename) {
	corpus.loadIxs(filename);
}

void VecInputSource::saveIxs(string filename) {
	corpus.saveIxs(filename);
}

int VecInputSource::getVecLength() {
	return vecLength;
}

} /* namespace netlib */
