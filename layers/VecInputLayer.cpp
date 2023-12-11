/**
 * @file
 * @brief Defines VecInputLayer class, an input layer that handles singular (not sequential) data.
 *
 */

#include "VecInputLayer.h"

#include "../Network.h"
#include "../gpu/CudaFunc.h"
#include <iostream>
#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "../nonlinearity/Nonlinearity.h"
#include "InputLayer.h"

using namespace std;

namespace netlib {

VecInputLayer::VecInputLayer(string name, cublasHandle_t& handle, int batchSize, int size, Nonlinearity* nonlinearity, float dropout)
		: InputLayer(name, handle, batchSize, size, 1, nonlinearity, dropout) {
	nParams = 0;
}

VecInputLayer::~VecInputLayer() {
}

void VecInputLayer::forward(int batchNum, dtype2 *params, bool deriv,
		bool stochasticDropout) {
	dtype2 *d_input = net->getRevInputs()[0];
	int arraySize = batchSize * size * sizeof(dtype2);

	int s = 0;
	dtype2* act = activation[s];
    checkCudaErrors(cudaMemcpy(ffInput, d_input, arraySize, cudaMemcpyDeviceToDevice));

	nonlinearity->activation(size, ffInput, act);

	if (next != NULL) {
		next->forward(batchNum, params, deriv, stochasticDropout);
	}
}

void VecInputLayer::Rforward(dtype2 *v, int batchNum) {
}

int VecInputLayer::getNParams() {
	return nParams;
}

} /* namespace netlib */
