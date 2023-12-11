/**
 * @file
 * @brief Defines DenseLayer class, a densely-connected layer core.
 *
 */

#include "DenseCore.h"

#include "../../gpu/CudaFunc.h"
#include "../../gpu/CublasFunc.h"
#include <iostream>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "../../config/WeightInit.h"

using namespace std;

namespace netlib {

DenseCore::DenseCore(string name, cublasHandle_t& handle, int batchSize, int size, int seqLength)
	: Core(name, handle, batchSize, size, seqLength) {
}

DenseCore::~DenseCore() {
}

int DenseCore::getNParams() {
	if (bnAfter) {
		return preSize * size;
	} else {
		return (preSize + 1) * size;
	}
}

void DenseCore::setParamOffset(int offset) {
	Core::setParamOffset(offset);
	biasOffset = offset + preSize * size;
	cout << name << endl;
	cout << "paramOffset: " << paramOffset << endl;
	cout << "biasOffset: " << biasOffset << endl << endl;
}

void DenseCore::forward(dtype2* prevAct, dtype2* act, dtype2* params, unsigned int* h_inputLengths,
		unsigned int* d_inputLengths, bool deriv, dtype2* priorAct, int s) {
	dtype2* w = params + paramOffset;
	dtype2* b = params + biasOffset;
	CublasFunc::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, batchSize, size, preSize, &one,
			prevAct, batchSize, w, preSize, &zero, act, batchSize);
	if (!bnAfter) CudaFunc::addRowVecMat(act, b, batchSize, size, batchSize);
}

void DenseCore::initWeights(dtype1* params, WeightInit& ffWeightInit, WeightInit& recWeightInit) {
	ffWeightInit.initialize(params+paramOffset, preSize, size);
	if (!bnAfter) checkCudaErrors(cudaMemset(params+biasOffset, 0, size * sizeof(dtype1)));
}

void DenseCore::calcGrad(dtype2 *prevAct, dtype2 *act, dtype2 *prevError, dtype2 *error,
		dtype2 *grad, dtype2 *params, unsigned int *h_inputLengths,
		unsigned int *d_inputLengths, dtype2 *priorAct, dtype2 *priorError, int s) {
	dtype2* wGrad = grad + paramOffset;
	dtype2* bGrad = grad + biasOffset;
	CublasFunc::gemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, preSize, size, batchSize, &one,
			prevAct, batchSize, error, batchSize, &one, wGrad, preSize);
	if (!bnAfter) CudaFunc::sum_cols_reduce4(error, bGrad, batchSize, size);

	// backpropagate error
	dtype2* w = params + paramOffset;
	CublasFunc::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, batchSize, preSize, size, &one,
			error, batchSize, w, preSize, &one, prevError, batchSize);
}

void DenseCore::addGradL2(dtype2* params, dtype2* grad, dtypeh l2) {
	int numWeights = biasOffset - paramOffset;
	CublasFunc::axpy(handle, numWeights, &l2, params+paramOffset, 1, grad+paramOffset, 1);
}

} /* namespace netlib */
