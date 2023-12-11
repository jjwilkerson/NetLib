/**
 * @file
 * @brief Defines classes that implement activation functions.
 *
 */

#include "Nonlinearity.h"

#include "../gpu/CudaFunc.h"
#include "../gpu/CublasFunc.h"
#include <cassert>
#include <cuda_runtime.h>
#include "cublas_v2.h"
//#include <thrust/device_ptr.h>
//#include <thrust/fill.h>
#include <helper_cuda.h>

using namespace std;

namespace netlib {

Nonlinearity::Nonlinearity(int batchSize) : batchSize(batchSize) {
}

Nonlinearity::~Nonlinearity() {
}

Linear::Linear(int batchSize) : Nonlinearity(batchSize) {
}

Linear::~Linear() {
}

void Linear::activation(int size, dtype2* preAct, dtype2* act) {
    checkCudaErrors(cudaMemcpy(act, preAct, batchSize * size * sizeof(dtype2), cudaMemcpyDeviceToDevice));
}

void Linear::d_activation(int size, dtype2* preAct, dtype2* act, dtype2* d_act) {
	CudaFunc::fill(d_act, batchSize * size, 1.0);
}

bool Linear::isLinear() {
	return true;
}

Tanh::Tanh(int batchSize) : Nonlinearity(batchSize) {
}

Tanh::~Tanh() {
}

void Tanh::activation(int size, dtype2* preAct, dtype2* act) {
	CudaFunc::tanh(preAct, act, batchSize * size);
}

void Tanh::d_activation(int size, dtype2* preAct, dtype2* act, dtype2* d_act) {
	CudaFunc::dTanh(act, d_act, batchSize * size);
}

bool Tanh::isLinear() {
	return false;
}

LeakyReLU::LeakyReLU(int batchSize) : Nonlinearity(batchSize) {
}

LeakyReLU::~LeakyReLU() {
}

void LeakyReLU::activation(int size, dtype2* preAct, dtype2* act) {
	CudaFunc::leakyReLU(preAct, act, batchSize * size);
}

void LeakyReLU::d_activation(int size, dtype2* preAct, dtype2* act, dtype2* d_act) {
	CudaFunc::dLeakyReLU(preAct, d_act, batchSize * size);
}

bool LeakyReLU::isLinear() {
	return false;
}

Softmax::Softmax(cublasHandle_t& handle, int batchSize) : Nonlinearity(batchSize), handle(handle) {
	int arraySize = batchSize * sizeof(dtype2);
	checkCudaErrors(cudaMalloc((void **)&tempVec, arraySize));
}

Softmax::~Softmax() {
	checkCudaErrors(cudaFree(tempVec));
}

void Softmax::activation(int size, dtype2* preAct, dtype2* act) {
	int tempArraySize = batchSize * sizeof(dtype2);

	//find max for each example
	for (int i = 0; i < batchSize; i++) {
		int ix;
		CublasFunc::iamax(handle, size, preAct+i, batchSize, &ix);
		ix--;

		checkCudaErrors(cudaMemcpy(tempVec+i, preAct+IDX2(i,ix,batchSize), sizeof(dtype2),
				cudaMemcpyDeviceToDevice));
	}

	int arraySize = batchSize * size * sizeof(dtype2);
	checkCudaErrors(cudaMemcpy(act, preAct, arraySize, cudaMemcpyDeviceToDevice));

	//subtract max for numerical stability
	CudaFunc::subtractColumnVecMat(act, tempVec, batchSize, size, batchSize);
	CudaFunc::iexp(act, batchSize * size);

	checkCudaErrors(cudaMemset((void *)tempVec, 0, tempArraySize));
	CudaFunc::sum_rows_reduce4(act, tempVec, batchSize, size);
	CudaFunc::divColVecMat(act, tempVec, batchSize, size, batchSize);
}

void Softmax::d_activation(int size, dtype2* preAct, dtype2* act, dtype2* d_act) {
	assert(false);
}

bool Softmax::isLinear() {
	return false;
}

BiReLU::BiReLU(int batchSize, dtype2 x1, dtype2 x2) : Nonlinearity(batchSize), x1(x1), x2(x2) {
}

BiReLU::~BiReLU() {
}

void BiReLU::activation(int size, dtype2* preAct, dtype2* act) {
	CudaFunc::biReLU(preAct, act, batchSize * size, x1, x2);
}

void BiReLU::d_activation(int size, dtype2* preAct, dtype2* act, dtype2* d_act) {
	CudaFunc::dBiReLU(preAct, d_act, batchSize * size, x1, x2);
}

bool BiReLU::isLinear() {
	return false;
}

LeakyBiReLU::LeakyBiReLU(int batchSize, dtype2 x1, dtype2 x2, dtype2 a) : Nonlinearity(batchSize), x1(x1), x2(x2), a(a) {
}

LeakyBiReLU::~LeakyBiReLU() {
}

void LeakyBiReLU::activation(int size, dtype2* preAct, dtype2* act) {
	CudaFunc::leakyBiReLU(preAct, act, batchSize * size, x1, x2, a);
}

void LeakyBiReLU::d_activation(int size, dtype2* preAct, dtype2* act, dtype2* d_act) {
	CudaFunc::dLeakyBiReLU(preAct, d_act, batchSize * size, x1, x2, a);
}

bool LeakyBiReLU::isLinear() {
	return false;
}

} /* namespace netlib */
