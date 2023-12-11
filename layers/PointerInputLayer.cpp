/**
 * @file
 * @brief Defines PointerInputLayer class, an input layer that has its input directly set via a pointer.
 *
 */

#include "PointerInputLayer.h"

#include <cuda_runtime.h>
#include <helper_cuda.h>

namespace netlib {

PointerInputLayer::PointerInputLayer(string name, cublasHandle_t& handle, int batchSize, int size, int seqLength)
		: InputLayer(name, handle, batchSize, size, seqLength, NULL, 0.0) {
}

PointerInputLayer::~PointerInputLayer() {
}

void PointerInputLayer::initMem(bool training, bool optHF) {
	int arraySize = batchSize * size * sizeof(dtype2);

	if (dropout != 0.0) {
		checkCudaErrors(cudaMalloc((void **)&dropoutMask, arraySize));
		checkCudaErrors(cudaMalloc((void **)&doActivation, arraySize));
	}

	if (training) {
		error = new dtype2*[seqLength];
		for (int s = 0; s < seqLength; s++) {
	        checkCudaErrors(cudaMalloc((void **)&error[s], arraySize));
		}
	}
}

void PointerInputLayer::freeMem(bool training) {
	if (dropoutMask != NULL) {
		checkCudaErrors(cudaFree(dropoutMask));
		checkCudaErrors(cudaFree(doActivation));
	}

	if(training) {
		for (int s = 0; s < seqLength; s++) {
			checkCudaErrors(cudaFree(error[s]));
		}
		delete [] error;
	}
}

void PointerInputLayer::setActivation(dtype2 **act) {
	activation = act;
}

void PointerInputLayer::forward(int batchNum, dtype2 *params, bool deriv,
		bool stochasticDropout) {
	if (next != NULL) {
		next->forward(batchNum, params, deriv, stochasticDropout);
	}
}

void PointerInputLayer::Rforward(dtype2 *v, int batchNum) {
}

int PointerInputLayer::getNParams() {
	return 0;
}

} /* namespace netlib */
