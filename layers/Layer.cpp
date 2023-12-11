/**
 * @file
 * @brief Defines Layer class, the base class for layers.
 *
 */

#include "Layer.h"
#include "../Network.h"
#include "../gpu/CudaFunc.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "../nonlinearity/Nonlinearity.h"

namespace netlib {

Layer::Layer(string name, cublasHandle_t& handle, int batchSize, int size, int seqLength, Nonlinearity* nonlinearity, float dropout)
			: name(name), handle(handle), batchSize(batchSize), size(size), seqLength(seqLength), nonlinearity(nonlinearity),
			  dropout(dropout) {
}

Layer::~Layer() {
	// TODO Auto-generated destructor stub
}

void Layer::setPrev(Layer* p) {
	prev = p;
	p->next = this;
}

void Layer::asOutputLayer() {
	outputLayer = true;
}

bool Layer::isOutputLayer() {
	return outputLayer;
}

void Layer::setBnAfter(bool b) {
	bnAfter = b;
}

void Layer::initMem(bool training, bool optHF) {
	activation = new dtype2*[seqLength];
	if (training) {
		d_activation = new dtype2*[seqLength];
		if (optHF) {
			R_activation = new dtype2*[seqLength];
		}
	}

	int arraySize = batchSize * size * sizeof(dtype2);
	for (int s = 0; s < seqLength; s++) {
        checkCudaErrors(cudaMalloc((void **)&activation[s], arraySize));
    	if (training) {
			checkCudaErrors(cudaMalloc((void **)&d_activation[s], arraySize));
			if (optHF) {
				checkCudaErrors(cudaMalloc((void **)&R_activation[s], arraySize));
			}
    	}
    }

	//NOTE: not needed by ActivationLayer
	checkCudaErrors(cudaMalloc((void **)&ffInput, arraySize));

	if (dropout != 0.0) {
		checkCudaErrors(cudaMalloc((void **)&dropoutMask, arraySize));
		checkCudaErrors(cudaMalloc((void **)&doActivation, arraySize));
	}

	if (training) {
		int arraySize = batchSize * size * sizeof(dtype2);

		error = new dtype2*[seqLength];
		for (int s = 0; s < seqLength; s++) {
	        checkCudaErrors(cudaMalloc((void **)&error[s], arraySize));
		}

		checkCudaErrors(cudaMalloc((void **)&singleShared1, arraySize));
		checkCudaErrors(cudaMalloc((void **)&singleShared2, arraySize));

	}
}

void Layer::freeMem(bool training) {
	for (int s = 0; s < seqLength; s++) {
        checkCudaErrors(cudaFree(activation[s]));
    	if (training) {
			checkCudaErrors(cudaFree(d_activation[s]));
			if (R_activation != NULL) {
				checkCudaErrors(cudaFree(R_activation[s]));
			}
    	}
    }
	delete [] activation;
	if (training) {
		delete [] d_activation;
		if (R_activation != NULL) {
			delete [] R_activation;
		}
	}

	if (ffInput != NULL) {
		checkCudaErrors(cudaFree(ffInput));
	}

	if (dropoutMask != NULL) {
		checkCudaErrors(cudaFree(dropoutMask));
		checkCudaErrors(cudaFree(doActivation));
	}

	if(training) {
		for (int s = 0; s < seqLength; s++) {
			checkCudaErrors(cudaFree(error[s]));
		}
		delete [] error;

		checkCudaErrors(cudaFree(singleShared1));
		checkCudaErrors(cudaFree(singleShared2));
	}

}

void Layer::initWeights(WeightInit& ffWeightInit, WeightInit& recWeightInit) {
}

void Layer::iterInit() {
	if (dropout != 0) {
		CudaFunc::fillDropoutMask(dropoutMask, batchSize * size, dropout);
	}
}

void Layer::clearError() {
	int arraySize = batchSize * size * sizeof(dtype2);
	for (int s = 0; s < seqLength; s++) {
		checkCudaErrors(cudaMemset((void *)error[s], 0, arraySize));
	}
}

void Layer::clearRact() {
	int arraySize = batchSize * size * sizeof(dtype2);
	for (int s = 0; s < seqLength; s++) {
		checkCudaErrors(cudaMemset((void *)R_activation[s], 0, arraySize));
	}
}

void Layer::applyMask(dtype2* a, int size, int s, int batchNum) {
	unsigned int** d_inputLengths = net->getDInputLengths();
	CudaFunc::maskByLength(a, d_inputLengths[batchNum], s, size, batchSize);
}

bool Layer::hasParam(int i) {
	return (i >= paramOffset) && (i < (biasOffset+size));
}

bool Layer::isWeight(int i) {
	return (i >= paramOffset) && (i < biasOffset);
}

void Layer::setStochasticOverride(bool value) {
	stochasticOverrideSet = true;
	stochasticOverrideValue = value;
}

} /* namespace netlib */
