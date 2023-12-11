/**
 * @file
 * @brief Defines SedInputSource class, which serves sentences as input to a network.
 *
 */

#include "SedInputSource.h"

#include "WordVectors.h"
#include "WvCorpusIterator.h"
#include "../gpu/CudaFunc.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>

namespace netlib {

SedInputSource::SedInputSource(WvCorpusIterator& corpus, WordVectors& wv, int batchSize, int maxSeqLength,
		int maxNumBatch, bool doForward, ix_type startIx, bool dictGpu1)
			: corpus(corpus), wv(wv), batchSize(batchSize), maxSeqLength(maxSeqLength),
			  maxNumBatch(maxNumBatch), doForward(doForward), dictGpu1(dictGpu1) {
	initMem();
	numBatchSaved = 0;
	firstBatchIx = startIx;
}

SedInputSource::~SedInputSource() {
	freeMem();
}

void SedInputSource::initMem() {
	d_inputs_rev = new dtype2*[maxSeqLength];
    if (doForward) {
    	d_inputs_fwd = new dtype2*[maxSeqLength];
    }
    d_targets = new dtype2*[maxSeqLength];

	int arraySize = batchSize * wv.wvLength * sizeof(dtype2);
    for (int s = 0; s < maxSeqLength; s++) {
        checkCudaErrors(cudaMalloc((void **)&d_inputs_rev[s], arraySize));
        if (doForward) {
        	checkCudaErrors(cudaMalloc((void **)&d_inputs_fwd[s], arraySize));
        }
        checkCudaErrors(cudaMalloc((void **)&d_targets[s], arraySize));
    }

    arraySize = maxSeqLength * sizeof(dtype2*);
    checkCudaErrors(cudaMalloc((void **)&dd_inputs_rev, arraySize));
    checkCudaErrors(cudaMemcpy(dd_inputs_rev, d_inputs_rev, arraySize, cudaMemcpyHostToDevice));
    if (doForward) {
        checkCudaErrors(cudaMalloc((void **)&dd_inputs_fwd, arraySize));
        checkCudaErrors(cudaMemcpy(dd_inputs_fwd, d_inputs_fwd, arraySize, cudaMemcpyHostToDevice));
    }

	d_inputLengths = new unsigned int*[maxNumBatch];
    for (int b = 0; b < maxNumBatch; b++) {
    	checkCudaErrors(cudaMalloc((void **)&d_inputLengths[b], batchSize * sizeof(unsigned int)));
    }

	checkCudaErrors(cudaMallocHost((void **)&h_inputLengths, batchSize * sizeof(unsigned int)));

	arraySize = batchSize * sizeof(int);
    d_batchIx = new int**[maxNumBatch];
    for (int b = 0; b < maxNumBatch; b++) {
    	d_batchIx[b] = new int*[maxSeqLength];
    	for (int s = 0; s < maxSeqLength; s++) {
    		checkCudaErrors(cudaMalloc((void **)&d_batchIx[b][s], arraySize));
    	}
    }

    h_batchIx = new int*[maxSeqLength];
	for (int s = 0; s < maxSeqLength; s++) {
		checkCudaErrors(cudaMallocHost((void **)&h_batchIx[s], arraySize));
	}

	arraySize = batchSize * sizeof(dtype2);
	d_matchMasks = new dtype2**[maxNumBatch];
	for (int b = 0; b < maxNumBatch; b++) {
		d_matchMasks[b] = new dtype2*[maxSeqLength];
		for (int s = 0; s < maxSeqLength; s++) {
			checkCudaErrors(cudaMalloc((void **)&d_matchMasks[b][s], arraySize));
		}
	}

	checkCudaErrors(cudaMallocHost((void **)&h_matchMask, arraySize));

	if (dictGpu1) {
    	d1_inputs_rev = new dtype2*[maxSeqLength];
        if (doForward) {
        	d1_inputs_fwd = new dtype2*[maxSeqLength];
        }
        d1_targets = new dtype2*[maxSeqLength];

		checkCudaErrors(cudaSetDevice(1));

		arraySize = batchSize * wv.wvLength * sizeof(dtype2);
		for (int s = 0; s < maxSeqLength; s++) {
			checkCudaErrors(cudaMalloc((void **)&d1_inputs_rev[s], arraySize));
			if (doForward) {
				checkCudaErrors(cudaMalloc((void **)&d1_inputs_fwd[s], arraySize));
			}
			checkCudaErrors(cudaMalloc((void **)&d1_targets[s], arraySize));
		}

		arraySize = maxSeqLength * sizeof(dtype2*);
		checkCudaErrors(cudaMalloc((void **)&dd1_inputs_rev, arraySize));
		checkCudaErrors(cudaMemcpy(dd1_inputs_rev, d1_inputs_rev, arraySize, cudaMemcpyHostToDevice));
		if (doForward) {
			checkCudaErrors(cudaMalloc((void **)&dd1_inputs_fwd, arraySize));
			checkCudaErrors(cudaMemcpy(dd1_inputs_fwd, d1_inputs_fwd, arraySize, cudaMemcpyHostToDevice));
		}

		d1_inputLengths = new unsigned int*[maxNumBatch];
		for (int b = 0; b < maxNumBatch; b++) {
			checkCudaErrors(cudaMalloc((void **)&d1_inputLengths[b], batchSize * sizeof(unsigned int)));
		}

		arraySize = batchSize * sizeof(int);
		d1_batchIx = new int**[maxNumBatch];
		for (int b = 0; b < maxNumBatch; b++) {
			d1_batchIx[b] = new int*[maxSeqLength];
			for (int s = 0; s < maxSeqLength; s++) {
				checkCudaErrors(cudaMalloc((void **)&d1_batchIx[b][s], arraySize));
			}
		}

		arraySize = batchSize * sizeof(dtype2);
		d1_matchMasks = new dtype2**[maxNumBatch];
		for (int b = 0; b < maxNumBatch; b++) {
			d1_matchMasks[b] = new dtype2*[maxSeqLength];
			for (int s = 0; s < maxSeqLength; s++) {
				checkCudaErrors(cudaMalloc((void **)&d1_matchMasks[b][s], arraySize));
			}
		}

		checkCudaErrors(cudaSetDevice(0));
	}
}

void SedInputSource::freeMem() {
    for (int s = 0; s < maxSeqLength; s++) {
        checkCudaErrors(cudaFree(d_inputs_rev[s]));
        if (doForward) {
        	checkCudaErrors(cudaFree(d_inputs_fwd[s]));
        }
        checkCudaErrors(cudaFree(d_targets[s]));
    }
    delete [] d_inputs_rev;
    if (doForward) {
        delete [] d_inputs_fwd;
    }
    delete [] d_targets;

    checkCudaErrors(cudaFree(dd_inputs_rev));
    if (doForward) {
        checkCudaErrors(cudaFree(dd_inputs_fwd));
    }

    for (int b = 0; b < maxNumBatch; b++) {
    	checkCudaErrors(cudaFree(d_inputLengths[b]));
    }
	delete [] d_inputLengths;

    checkCudaErrors(cudaFreeHost(h_inputLengths));

	for (int b = 0; b < maxNumBatch; b++) {
		for (int s = 0; s < maxSeqLength; s++) {
			checkCudaErrors(cudaFree(d_batchIx[b][s]));
		}
		delete [] d_batchIx[b];
	}
	delete [] d_batchIx;

	for (int s = 0; s < maxSeqLength; s++) {
		checkCudaErrors(cudaFreeHost(h_batchIx[s]));
	}
	delete [] h_batchIx;

	for (int b = 0; b < maxNumBatch; b++) {
		for (int s = 0; s < maxSeqLength; s++) {
			checkCudaErrors(cudaFree(d_matchMasks[b][s]));
		}
		delete [] d_matchMasks[b];
	}
	delete [] d_matchMasks;

	checkCudaErrors(cudaFreeHost(h_matchMask));

	if (dictGpu1) {
		checkCudaErrors(cudaSetDevice(1));

		for (int s = 0; s < maxSeqLength; s++) {
	        checkCudaErrors(cudaFree(d1_inputs_rev[s]));
	        if (doForward) {
	        	checkCudaErrors(cudaFree(d1_inputs_fwd[s]));
	        }
	        checkCudaErrors(cudaFree(d1_targets[s]));
	    }
	    delete [] d1_inputs_rev;
	    if (doForward) {
	        delete [] d1_inputs_fwd;
	    }
	    delete [] d1_targets;

	    checkCudaErrors(cudaFree(dd1_inputs_rev));
	    if (doForward) {
	        checkCudaErrors(cudaFree(dd1_inputs_fwd));
	    }

	    for (int b = 0; b < maxNumBatch; b++) {
	    	checkCudaErrors(cudaFree(d1_inputLengths[b]));
	    }
		delete [] d1_inputLengths;

		for (int b = 0; b < maxNumBatch; b++) {
			for (int s = 0; s < maxSeqLength; s++) {
				checkCudaErrors(cudaFree(d1_batchIx[b][s]));
			}
			delete [] d1_batchIx[b];
		}
		delete [] d1_batchIx;

		for (int b = 0; b < maxNumBatch; b++) {
			for (int s = 0; s < maxSeqLength; s++) {
				checkCudaErrors(cudaFree(d1_matchMasks[b][s]));
			}
			delete [] d1_matchMasks[b];
		}
		delete [] d1_matchMasks;

		checkCudaErrors(cudaSetDevice(0));
	}
}

void SedInputSource::toNextBatchSet() {
	firstBatchIx += (batchSize * maxNumBatch);
	corpus.resetTo(firstBatchIx);
	numBatchSaved = 0;
}

void SedInputSource::toFirstBatch() {
	corpus.resetTo(firstBatchIx);
}

bool SedInputSource::hasNext() {
	return corpus.hasNext();
}

void SedInputSource::nextBatch(int batchNum, vector<string>* tokensRet) { //TODO: maybe this compute batchNum, return it rather than taking it
	if (batchNum >= numBatchSaved) {
		copyNextBatch(batchNum, tokensRet);
	} else {
		corpus.resetTo(corpus.currentIx()+batchSize);
	}

	fillInputs(batchNum);

	if (batchNum >= numBatchSaved) {
		numBatchSaved++;
	}
}

void SedInputSource::next(std::vector<std::string>* tokensRet) {
	copyNextBatch(0, tokensRet);
	fillInputs(0);
}

void SedInputSource::computeMatchMasks(int batchNum, dtype2** outputs) {
	if (batchNum < numBatchSaved-1) {
		return;
	}
	assert (batchNum == numBatchSaved-1);

	dtype2** d_masks;
	unsigned int** inputLengths;
	int*** batchIx;

#ifndef DEBUG
	if (dictGpu1) {
		checkCudaErrors(cudaSetDevice(1));
		d_masks = d1_matchMasks[batchNum];
		inputLengths = d1_inputLengths;
		batchIx = d1_batchIx;
	} else {
		d_masks = d_matchMasks[batchNum];
		inputLengths = d_inputLengths;
		batchIx = d_batchIx;
	}

	for (int s = 0; s < maxSeqLength; s++) {
		wv.nearestBatchMask(outputs[s], batchIx[batchNum][s], h_inputLengths, s, d_masks[s], inputLengths[batchNum]);
	}

	if (dictGpu1) {
		int arraySize = batchSize * sizeof(dtype2);
		for (int s = 0; s < maxSeqLength; s++) {
			checkCudaErrors(cudaMemcpyPeer(d_matchMasks[batchNum][s], 0, d1_matchMasks[batchNum][s], 1, arraySize));
		}
	}

	checkCudaErrors(cudaSetDevice(0));
#else
//	cout << "DEBUG mode, not computing match masks" << endl;
#endif
}

dtype2** SedInputSource::matchMasks(int batchNum) {
	assert (batchNum < numBatchSaved);

	return d_matchMasks[batchNum];
}

void SedInputSource::reset() {
	corpus.reset();
	firstBatchIx = 0;
	numBatchSaved = 0;
}

void SedInputSource::copyNextBatch(int batchNum, vector<string>* tokensRet) {
	corpus.next(h_batchIx, h_inputLengths, tokensRet);
	copyInputs(batchNum);
}

void SedInputSource::copyInputs(int batchNum) {
	int*** batchIx = NULL;
	unsigned int** inputLengths = NULL;
	if (dictGpu1) {
		checkCudaErrors(cudaSetDevice(1));
		batchIx = d1_batchIx;
		inputLengths = d1_inputLengths;
	} else {
		batchIx = d_batchIx;
		inputLengths = d_inputLengths;
	}

	int ixArraySize = batchSize * sizeof(int);
	for (int s = 0; s < maxSeqLength; s++) {
	    checkCudaErrors(cudaMemcpy(batchIx[batchNum][s], h_batchIx[s], ixArraySize, cudaMemcpyHostToDevice));
	}

	checkCudaErrors(cudaMemcpy(inputLengths[batchNum], h_inputLengths, batchSize * sizeof(unsigned int), cudaMemcpyHostToDevice));

	if (dictGpu1) {
		checkCudaErrors(cudaSetDevice(0));
		checkCudaErrors(cudaMemcpy(d_inputLengths[batchNum], h_inputLengths, batchSize * sizeof(unsigned int), cudaMemcpyHostToDevice));
	}
}

void SedInputSource::saveReplacerState(const char *stateFilename) {
	corpus.saveReplacerState(stateFilename);
}

void SedInputSource::fillInputs(int batchNum) {
	dtype2** inputs_fwd = NULL;
	dtype2** inputs_rev = NULL;
	dtype2** targets = NULL;
	int*** batchIx = NULL;
	unsigned int** inputLengths = NULL;
	if (dictGpu1) {
		checkCudaErrors(cudaSetDevice(1));
//		inputs_fwd = dd1_inputs_fwd;
//		inputs_rev = dd1_inputs_rev;
		targets = d1_targets;
		batchIx = d1_batchIx;
		inputLengths = d1_inputLengths;
	} else {
		inputs_fwd = dd_inputs_fwd;
		inputs_rev = dd_inputs_rev;
		targets = d_targets;
		batchIx = d_batchIx;
		inputLengths = d_inputLengths;
	}

	for (int s = 0; s < maxSeqLength; s++) {
		CudaFunc::fillInput(inputs_fwd, inputs_rev, targets[s], batchIx[batchNum][s], inputLengths[batchNum], s,
				wv.getDictGpu(), wv.getNumWords());
	}

	if (dictGpu1) {
		int arraySize = batchSize * wv.wvLength * sizeof(dtype2);
		for (int s = 0; s < maxSeqLength; s++) {
//			checkCudaErrors(cudaMemcpyPeer(d_inputs_rev[s], 0, d1_inputs_rev[s], 1, arraySize));
//			if (d_inputs_fwd != NULL) {
//				checkCudaErrors(cudaMemcpyPeer(d_inputs_fwd[s], 0, d1_inputs_fwd[s], 1, arraySize));
//			}
			checkCudaErrors(cudaMemcpyPeer(d_targets[s], 0, d1_targets[s], 1, arraySize));
		}

		checkCudaErrors(cudaSetDevice(0));
		for (int s = 0; s < maxSeqLength; s++) {
			CudaFunc::copyTargetsToInputs(dd_inputs_fwd, dd_inputs_rev, d_targets[s], d_inputLengths[batchNum], s);
		}
	}
}

dtype2** SedInputSource::getRevInputs() {
	return d_inputs_rev;
}

dtype2** SedInputSource::getFwdInputs() {
	return d_inputs_fwd;
}

unsigned int** SedInputSource::getDInputLengths() {
	return d_inputLengths;
}

unsigned int* SedInputSource::getHInputLengths() {
	return h_inputLengths;
}

dtype2** SedInputSource::getTargets() {
	return d_targets;
}

int SedInputSource::getBatchSize() {
	return batchSize;
}

WvCorpusIterator& SedInputSource::getCorpus() {
	return corpus;
}

ix_type SedInputSource::size() {
	return corpus.size();
}

void SedInputSource::get(ix_type selectedIx, string& sentence) {
	corpus.get(selectedIx, sentence);
}

void SedInputSource::inputFor(string* sentences, int count, bool printTokens,
		vector<string>* tokensRet) {
	corpus.inputFor(sentences, count, h_batchIx, h_inputLengths, printTokens, tokensRet);
	copyInputs(0);
	fillInputs(0);
}

bool SedInputSource::hasNextBatchSet() {
	return (firstBatchIx + batchSize * maxNumBatch) <= corpus.size();
}

void SedInputSource::shuffle() {
	corpus.shuffle();
}

void SedInputSource::loadIxs(string filename) {
	corpus.loadIxs(filename);
}

void SedInputSource::saveIxs(string filename) {
	corpus.saveIxs(filename);
}

string SedInputSource::getIxFilename() {
	return corpus.getIxFilename();
}

} /* namespace netlib */

