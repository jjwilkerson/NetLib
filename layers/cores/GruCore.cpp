/**
 * @file
 * @brief Defines GruCore, a GRU layer core.
 *
 * A GRU (gated recurrent unit) layer core.
 */

#include "GruCore.h"
#include "../../gpu/CudaFunc.h"
#include "../../gpu/CublasFunc.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <iostream>
#include "../../config/WeightInit.h"

using namespace std;

namespace netlib {

GruCore::GruCore(string name, cublasHandle_t& handle, int batchSize, int size, int seqLength)
			: Core(name, handle, batchSize, size, seqLength) {
	initMem();
}

GruCore::~GruCore() {
	freeMem();
}

void GruCore::initMem() {
	zs = new dtype2*[seqLength];
	rs = new dtype2*[seqLength];
	h_candidates = new dtype2*[seqLength];

	size_t arraySize = batchSize * size * sizeof(dtype2);
	for (int s = 0; s < seqLength; s++) {
        checkCudaErrors(cudaMalloc((void **)&zs[s], arraySize));
		checkCudaErrors(cudaMalloc((void **)&rs[s], arraySize));
		checkCudaErrors(cudaMalloc((void **)&h_candidates[s], arraySize));
    }

    checkCudaErrors(cudaMalloc((void **)&dz, arraySize));
	checkCudaErrors(cudaMalloc((void **)&dr, arraySize));
	checkCudaErrors(cudaMalloc((void **)&dh_candidate, arraySize));
	checkCudaErrors(cudaMalloc((void **)&intermed1, arraySize));
	checkCudaErrors(cudaMalloc((void **)&intermed2, arraySize));
	checkCudaErrors(cudaMalloc((void **)&intermed3, arraySize));
	checkCudaErrors(cudaMalloc((void **)&recBias, arraySize));

	checkCudaErrors(cudaMemset(recBias, 0, arraySize));
}

void GruCore::freeMem() {
	for (int s = 0; s < seqLength; s++) {
        checkCudaErrors(cudaFree(zs[s]));
        checkCudaErrors(cudaFree(rs[s]));
        checkCudaErrors(cudaFree(h_candidates[s]));
    }
	delete [] zs;
	delete [] rs;
	delete [] h_candidates;

	checkCudaErrors(cudaFree(dz));
	checkCudaErrors(cudaFree(dr));
	checkCudaErrors(cudaFree(dh_candidate));
	checkCudaErrors(cudaFree(intermed1));
	checkCudaErrors(cudaFree(intermed2));
	checkCudaErrors(cudaFree(intermed3));
	checkCudaErrors(cudaFree(recBias));
}

int GruCore::getNParams() {
	int nParams = (preSize * size + size * size + size) * 3 + size;
	cout << "GruCore::getNParams: " << nParams << endl;
	return nParams;
}

void GruCore::setParamOffset(int offset) {
	Core::setParamOffset(offset);
	wzxOffset = offset;
	wzhOffset = wzxOffset + preSize * size;
	bzOffset = wzhOffset + size * size;
	wrxOffset = bzOffset + size;
	wrhOffset = wrxOffset + preSize * size;
	brOffset = wrhOffset + size * size;
	whxOffset = brOffset + size;
	whhOffset = whxOffset + preSize * size;
	bhOffset = whhOffset + size * size;
	brecOffset = bhOffset + size;
	cout << name << endl;
	cout << "paramOffset: " << paramOffset << endl;
	cout << "wzxOffset: " << wzxOffset << endl;
	cout << "wzhOffset: " << wzhOffset << endl;
	cout << "bzOffset: " << bzOffset << endl;
	cout << "wrxOffset: " << wrxOffset << endl;
	cout << "wrhOffset: " << wrhOffset << endl;
	cout << "brOffset: " << brOffset << endl;
	cout << "whxOffset: " << whxOffset << endl;
	cout << "whhOffset: " << whhOffset << endl;
	cout << "bhOffset: " << bhOffset << endl;
	cout << "brecOffset: " << brecOffset << endl;
	int np = brecOffset + size - offset;
	cout << "np: " << np << endl;
}

void GruCore::initWeights(dtype1 *params, WeightInit &ffWeightInit,
		WeightInit &recWeightInit) {
	ffWeightInit.initialize(params+wzxOffset, preSize, size);
	recWeightInit.initialize(params+wzhOffset, size, size);
	checkCudaErrors(cudaMemset(params+bzOffset, 0, size * sizeof(dtype1)));

	ffWeightInit.initialize(params+wrxOffset, preSize, size);
	recWeightInit.initialize(params+wrhOffset, size, size);
	checkCudaErrors(cudaMemset(params+brOffset, 0, size * sizeof(dtype1)));

	ffWeightInit.initialize(params+whxOffset, preSize, size);
	recWeightInit.initialize(params+whhOffset, size, size);
	checkCudaErrors(cudaMemset(params+bhOffset, 0, size * sizeof(dtype1)));

	checkCudaErrors(cudaMemset(params+brecOffset, 0, size * sizeof(dtype1)));
}

void GruCore::forward(dtype2 *prevAct, dtype2 *act, dtype2 *params,
		unsigned int *h_inputLengths, unsigned int *d_inputLengths, bool deriv,
		dtype2 *priorAct, int s) {
	dtype2* wz_x = params + wzxOffset;
	dtype2* wz_h = params + wzhOffset;
	dtype2* bz = params + bzOffset;
	dtype2* wr_x = params + wrxOffset;
	dtype2* wr_h = params + wrhOffset;
	dtype2* br = params + brOffset;
	dtype2* wh_x = params + whxOffset;
	dtype2* wh_h = params + whhOffset;
	dtype2* bh = params + bhOffset;
	dtype2* brec = params + brecOffset;

	dtype2* x_t = prevAct;
	dtype2* h_t = act;

	dtype2* z = zs[s];
	dtype2* r = rs[s];
	dtype2* h_candidate = h_candidates[s];

	dtype2* h_prev;
	if (s == 0) {
		size_t arraySize = batchSize * size * sizeof(dtype2);
		checkCudaErrors(cudaMemset(recBias, 0, arraySize));
		CudaFunc::addRowVecMat(recBias, brec, batchSize, size, batchSize);
		h_prev = recBias;
	} else {
		h_prev = priorAct;
	}


	// Step 1: Compute reset gate (r) and update gate (z) activations
	// r = sigmoid(Wr_x * x_t + Wr_h * h_prev + br)
	// z = sigmoid(Wz_x * x_t + Wz_h * h_prev + bz)
	CublasFunc::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, batchSize, size, preSize, &one,
			x_t, batchSize, wr_x, preSize, &zero, r, batchSize);
	CublasFunc::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, batchSize, size, size, &one,
			h_prev, batchSize, wr_h, size, &one, r, batchSize);
	CublasFunc::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, batchSize, size, preSize, &one,
			x_t, batchSize, wz_x, preSize, &zero, z, batchSize);
	CublasFunc::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, batchSize, size, size, &one,
			h_prev, batchSize, wz_h, size, &one, z, batchSize);
	CudaFunc::addRowVecMat(r, br, batchSize, size, batchSize);

	CudaFunc::addRowVecMat(z, bz, batchSize, size, batchSize);
	CudaFunc::sigmoid(r, r, batchSize * size);
	CudaFunc::sigmoid(z, z, batchSize * size);

	// mask for varying sentence lengths
	CudaFunc::maskByLength(r, d_inputLengths, s, size, batchSize);
	CudaFunc::maskByLength(z, d_inputLengths, s, size, batchSize);

	// Step 2: Compute candidate hidden state (h_candidate) activation
	// h_candidate = tanh(Wh_x * x_t + Wh_h * (r * h_prev) + bh)
	CudaFunc::multiply(r, h_prev, intermed1, batchSize * size);
	CublasFunc::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, batchSize, size, preSize, &one,
			x_t, batchSize, wh_x, preSize, &zero, h_candidate, batchSize);
	CublasFunc::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, batchSize, size, size, &one,
			intermed1, batchSize, wh_h, size, &one, h_candidate, batchSize);
	CudaFunc::addRowVecMat(h_candidate, bh, batchSize, size, batchSize);
	CudaFunc::tanh(h_candidate, h_candidate, batchSize * size);

	CudaFunc::maskByLength(h_candidate, d_inputLengths, s, size, batchSize);

	// Step 3: Update the hidden state (h_t)
	// h_t = (1 - z) * h_prev + z * h_candidate
	CudaFunc::multiply(h_candidate, z, h_t, batchSize * size);
	CudaFunc::subtractFromOne(z, intermed2, batchSize * size);
	CudaFunc::multiply(h_prev, intermed2, intermed1, batchSize * size);
	CudaFunc::add(intermed1, h_t, h_t, batchSize * size);

	CudaFunc::maskByLength(h_t, d_inputLengths, s, size, batchSize);
}

void GruCore::calcGrad(dtype2 *prevAct, dtype2 *act, dtype2 *prevError, dtype2 *error,
		dtype2 *grad, dtype2 *params, unsigned int *h_inputLengths,
		unsigned int *d_inputLengths, dtype2 *priorAct, dtype2 *priorError, int s) {
	dtype2* wz_x = params + wzxOffset;
	dtype2* wz_h = params + wzhOffset;
	dtype2* wr_x = params + wrxOffset;
	dtype2* wr_h = params + wrhOffset;
	dtype2* wh_x = params + whxOffset;
	dtype2* wh_h = params + whhOffset;

	dtype2* dwz_x = grad + wzxOffset;
	dtype2* dwz_h = grad + wzhOffset;
	dtype2* dbz = grad + bzOffset;
	dtype2* dwr_x = grad + wrxOffset;
	dtype2* dwr_h = grad + wrhOffset;
	dtype2* dbr = grad + brOffset;
	dtype2* dwh_x = grad + whxOffset;
	dtype2* dwh_h = grad + whhOffset;
	dtype2* dbh = grad + bhOffset;
	dtype2* dbrec = grad + brecOffset;

	dtype2* h_t = act;
	dtype2* h_prev = (s > 0)?priorAct:recBias;
	dtype2* x_t = prevAct;

	dtype2* z = zs[s];
	dtype2* r = rs[s];
	dtype2* h_candidate = h_candidates[s];

	// Calculate dh_candidate, dr, and dz
	// Compute the gradients of the candidate hidden state (dh_candidate)
	CudaFunc::multiply(error, z, dh_candidate, batchSize * size);

	// Compute the gradients of the reset gate (dr)
	// Compute the pointwise product (1 - h_candidate^2)
	CudaFunc::multiply(h_candidate, h_candidate, intermed1, size * batchSize);
	CudaFunc::subtractFromOne(intermed1, intermed2, size * batchSize);

	// Compute Dh_candidate = (dh_candidate ⊙ one_minus_h_c_2)
	CudaFunc::multiply(dh_candidate, intermed2, dh_candidate, batchSize * size);

	// Compute ((dh_candidate ⊙ one_minus_h_c_2) * Wh_h^T)
	CublasFunc::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, batchSize, size, size, &one,
	                 dh_candidate, batchSize, wh_h, size, &zero, intermed2, batchSize);

	// Compute dr = (((Dh_candidate ⊙ one_minus_h_c_2) * Wh_h^T) ⊙ h_prev)
	CudaFunc::multiply(intermed2, h_prev, dr, batchSize * size);

	// Compute the gradients of the update gate (dz)
	CudaFunc::subtract(h_candidate, h_prev, intermed2, batchSize * size); // intermed2 = h_candidate - h_prev
	CudaFunc::multiply(error, intermed2, dz, batchSize * size); // dz = error ⊙ (h_candidate - h_prev)

	// multiply dr, dz, dh_candidate with the derivative of their activation function
	//Dr = dr ⊙ r ⊙ (1-r)
	CudaFunc::subtractFromOne(r, intermed1, batchSize * size);
	CudaFunc::multiply(r, intermed1, intermed2, batchSize * size);
	CudaFunc::multiply(intermed2, dr, dr, batchSize * size);

	//Dz = dz ⊙ z ⊙ (1-z)
	CudaFunc::subtractFromOne(z, intermed1, batchSize * size);
	CudaFunc::multiply(z, intermed1, intermed2, batchSize * size);
	CudaFunc::multiply(intermed2, dz, dz, batchSize * size);

	// mask for varying sentence lengths
	CudaFunc::maskByLength(dr, d_inputLengths, s, size, batchSize);
	CudaFunc::maskByLength(dz, d_inputLengths, s, size, batchSize);
	CudaFunc::maskByLength(dh_candidate, d_inputLengths, s, size, batchSize);

	// Compute gradients for Wz_x, Wr_x, Wh_x
	CublasFunc::gemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, preSize, size, batchSize, &one,
		x_t, batchSize, dz, batchSize, &one, dwz_x, preSize);
	CublasFunc::gemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, preSize, size, batchSize, &one,
		x_t, batchSize, dr, batchSize, &one, dwr_x, preSize);
	CublasFunc::gemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, preSize, size, batchSize, &one,
		x_t, batchSize, dh_candidate, batchSize, &one, dwh_x, preSize);

	// Compute gradients for Wz_h, Wr_h, Wh_h and biases bz, br, bh
	// Compute the pointwise product r ⊙ h_prev
	CudaFunc::multiply(r, h_prev, intermed1, size * batchSize);

	CublasFunc::gemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, size, size, batchSize, &one,
		h_prev, batchSize, dz, batchSize, &one, dwz_h, size);
	CublasFunc::gemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, size, size, batchSize, &one,
		h_prev, batchSize, dr, batchSize, &one, dwr_h, size);
	CublasFunc::gemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, size, size, batchSize, &one,
		intermed1, batchSize, dh_candidate, batchSize, &one, dwh_h, size);

	CudaFunc::sum_cols_reduce4(dz, dbz, batchSize, size);
	CudaFunc::sum_cols_reduce4(dr, dbr, batchSize, size);
	CudaFunc::sum_cols_reduce4(dh_candidate, dbh, batchSize, size);

	// Backpropagate the error to the previous layer
	CublasFunc::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, batchSize, preSize, size, &one,
	    dz, batchSize, wz_x, preSize, &one, prevError, batchSize);
	CublasFunc::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, batchSize, preSize, size, &one,
	    dr, batchSize, wr_x, preSize, &one, prevError, batchSize);
	CublasFunc::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, batchSize, preSize, size, &one,
	    dh_candidate, batchSize, wh_x, preSize, &one, prevError, batchSize);

	if (s == 0) {
		size_t arraySize = batchSize * size * sizeof(dtype2);
		checkCudaErrors(cudaMemset(intermed3, 0, arraySize));
		priorError = intermed3;
	}

	// Backpropagate the error to the prior time step
	// First term: error ⊙ (1 - z)
	CudaFunc::subtractFromOne(z, intermed1, batchSize * size); // Compute (1 - z)
	CudaFunc::multiply(error, intermed1, intermed2, batchSize * size); // Compute error ⊙ (1 - z)

	// Add the first term to priorError
	CudaFunc::add(intermed2, priorError, priorError, batchSize * size);

	// Second term: Dz * Wz_h^T
	CublasFunc::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, batchSize, size, size, &one,
					 dz, batchSize, wz_h, size, &one, priorError, batchSize); // Compute (z ⊙ (1 - z)) * Wz_h^T

	// Third term: (Dhc * Wh_h^T) ⊙ r
	CublasFunc::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, batchSize, size, size, &one,
					 dh_candidate, batchSize, wh_h, size, &zero, intermed1, batchSize);
	CudaFunc::multiply(intermed1, r, intermed2, batchSize * size);
	CudaFunc::add(intermed2, priorError, priorError, batchSize * size);

	// Fourth term: Dr * Wr_h^T
	CublasFunc::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, batchSize, size, size, &one,
					 dr, batchSize, wr_h, size, &one, priorError, batchSize);

	if (s == 0) {
		CudaFunc::sum_cols_reduce4(priorError, dbrec, batchSize, size);
	}
}

void GruCore::addGradL2(dtype2 *params, dtype2 *grad, dtypeh l2) {
	int ps = preSize * size;
	int ss = size * size;
	CublasFunc::axpy(handle, ps, &l2, params+wzxOffset, 1, grad+wzxOffset, 1);
	CublasFunc::axpy(handle, ss, &l2, params+wzhOffset, 1, grad+wzhOffset, 1);

	CublasFunc::axpy(handle, ps, &l2, params+wrxOffset, 1, grad+wrxOffset, 1);
	CublasFunc::axpy(handle, ss, &l2, params+wrhOffset, 1, grad+wrhOffset, 1);

	CublasFunc::axpy(handle, ps, &l2, params+whxOffset, 1, grad+whxOffset, 1);
	CublasFunc::axpy(handle, ss, &l2, params+whhOffset, 1, grad+whhOffset, 1);
}

} /* namespace netlib */
