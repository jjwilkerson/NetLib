/**
 * @file
 * @brief Defines DupOutLayer class, which converts singular data to sequential data.
 *
 * Converts singular data to sequential data by repeating its input data. The way it works is that the following layer will
 * always take the same data, regards of position in sequence.
 */

#include "DupOutLayer.h"

#include "../Network.h"
#include "../gpu/CublasFunc.h"
#include <iostream>
#include <cuda_runtime.h>
#include <helper_cuda.h>

using namespace std;

namespace netlib {

DupOutLayer::DupOutLayer(string name, cublasHandle_t& handle, int batchSize, int size, Nonlinearity* nonlinearity, float dropout)
		: Layer(name, handle, batchSize, size, 1, nonlinearity, dropout) {
	nParams = 0;
}

DupOutLayer::~DupOutLayer() {
}

int DupOutLayer::getNParams() {
	return nParams;
}

void DupOutLayer::setParamOffset(int offset) {
}

void DupOutLayer::forward(int batchNum, dtype2* params, bool deriv, bool stochasticDropout) {
	int arraySize = batchSize * size * sizeof(dtype2);
	checkCudaErrors(cudaMemcpy(activation[0], prev->activation[0], arraySize, cudaMemcpyDeviceToDevice));
	if (deriv) {
		checkCudaErrors(cudaMemcpy(d_activation[0], prev->d_activation[0], arraySize, cudaMemcpyDeviceToDevice));
	}

	next->forward(batchNum, params, deriv, stochasticDropout);
}

void DupOutLayer::calcGrad(dtype2* grad, int batchNum) {
	dtype2** prevError = prev->error;

	int arraySize = batchSize * size * sizeof(dtype2);
	checkCudaErrors(cudaMemcpy(prevError[0], error[0], arraySize, cudaMemcpyDeviceToDevice));

	prev->calcGrad(grad, batchNum);
}

void DupOutLayer::addGradL2(dtype2* grad, dtypeh l2) {
}

void DupOutLayer::Rforward(dtype2* v, int batchNum) {
	int arraySize = batchSize * size * sizeof(dtype2);
	checkCudaErrors(cudaMemcpy(R_activation[0], prev->R_activation[0], arraySize, cudaMemcpyDeviceToDevice));

	next->Rforward(v, batchNum);
}

void DupOutLayer::Rback(dtype2* Gv, int batchNum) {
	dtype2** prevError = prev->error;

	int arraySize = batchSize * size * sizeof(dtype2);
	checkCudaErrors(cudaMemcpy(prevError[0], error[0], arraySize, cudaMemcpyDeviceToDevice));

	prev->Rback(Gv, batchNum);
}

} /* namespace netlib */
