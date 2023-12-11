/**
 * @file
 * @brief Defines LayerNormCore class, a layer normalization layer core.
 *
 */

#include "LayerNormCore.h"

#include "../../Network.h"
#include "../../gpu/CudaFunc.h"
#include "../../gpu/CublasFunc.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <iostream>

extern int print_s;

using namespace std;

namespace netlib {

LayerNormCore::LayerNormCore(string name, cublasHandle_t& handle, int batchSize, int size, int seqLength,
		float stdInit)
		: Core(name, handle, batchSize, size, seqLength), stdInit(stdInit) {
	epsilon =1e-8f;
}

LayerNormCore::~LayerNormCore() {
}

int LayerNormCore::getNParams() {
	return size * 2;
}

void LayerNormCore::setParamOffset(int offset) {
	Core::setParamOffset(offset);
	stdOffset = offset + size;
	cout << name << endl;
	cout << "paramOffset: " << paramOffset << endl;
	cout << "stdOffset: " << stdOffset << endl << endl;
}

void LayerNormCore::initMem(bool training, bool optHF) {
	int arraySize = size * batchSize * sizeof(dtype2);
	int arraySizeSingle = size * sizeof(dtype2);
	int arraySizeCol = batchSize * sizeof(dtype2);
    checkCudaErrors(cudaMalloc((void **)&mean, arraySizeCol));
    checkCudaErrors(cudaMalloc((void **)&var, arraySizeCol));

    if (training) {
    	xhats = new dtype2*[seqLength];
    	inv_vars = new dtype2*[seqLength];
    	for (int s = 0; s < seqLength; s++) {
    	    checkCudaErrors(cudaMalloc((void **)&xhats[s], arraySize));
    	    checkCudaErrors(cudaMalloc((void **)&inv_vars[s], arraySizeCol));
    	}

	    checkCudaErrors(cudaMalloc((void **)&intermed1, arraySize));
	    checkCudaErrors(cudaMalloc((void **)&intermed2, arraySize));
	    checkCudaErrors(cudaMalloc((void **)&singleCol1, arraySizeCol));
    }
}

void LayerNormCore::freeMem(bool training) {
	checkCudaErrors(cudaFree(mean));
	checkCudaErrors(cudaFree(var));

	if (training) {
		for (int s = 0; s < seqLength; s++) {
	        checkCudaErrors(cudaFree(xhats[s]));
	        checkCudaErrors(cudaFree(inv_vars[s]));
		}
		delete [] xhats;
		delete [] inv_vars;
	}
}

void LayerNormCore::initWeights(dtype1* params, WeightInit& ffWeightInit,
		WeightInit& recWeightInit) {
	int arraySize = size * sizeof(dtype1);
	checkCudaErrors(cudaMemset(params + paramOffset, 0, arraySize));
	CudaFunc::fill1(params + stdOffset, size, stdInit);
}

void LayerNormCore::forward(dtype2* prevAct, dtype2* act, dtype2* params, unsigned int* h_inputLengths,
		unsigned int* d_inputLengths, bool deriv, dtype2* priorAct, int s) {
	dtype2* targetMean = params + paramOffset;
	dtype2* targetStd = params + stdOffset;
	int arraySize = size * batchSize * sizeof(dtype2);
	int arraySizeSingle = size * sizeof(dtype2);
	int arraySizeCol = batchSize * sizeof(dtype2);

	bool printDebug = (s == print_s);

	if (printDebug) {
		cout << name << " forward" << endl;
		cout << "s: " << s << endl;
	}

	if (printDebug) {
		Network::printStatsGpu("prevAct", prevAct, size * batchSize);
		cout << "before" << endl;
		Network::printAllGpu(prevAct, size * batchSize);
	}

	//calc mean
	checkCudaErrors(cudaMemset(mean, 0, arraySizeCol));
	CudaFunc::sum_rows_reduce4(prevAct, mean, batchSize, size);
	dtypeh a = 1.0f / size;
	CublasFunc::scal(handle, batchSize, &a, mean, 1);

	if (printDebug) {
		cout << "mean" << endl;
		Network::printAllGpu(mean, batchSize);
	}

	//calc var
	checkCudaErrors(cudaMemset(var, 0, arraySizeCol));
	checkCudaErrors(cudaMemcpy(act, prevAct, arraySize, cudaMemcpyDeviceToDevice));

	CublasFunc::scal(handle, batchSize, &minus_one, mean, 1);
	CudaFunc::addColumnVecMat(act, mean, batchSize, size, batchSize);
	CudaFunc::maskByLength(act, d_inputLengths, s, size, batchSize);

	CudaFunc::isquare(act, size * batchSize);
	CudaFunc::sum_rows_reduce4(act, var, batchSize, size);
	a = 1.0f / size;
	CublasFunc::scal(handle, batchSize, &a, var, 1);

	if (printDebug) {
		cout << "var" << endl;
		Network::printAllGpu(var, batchSize);
	}

	CudaFunc::iadd(var, batchSize, epsilon);
	CudaFunc::isqrt(var, batchSize);
	CudaFunc::iinvert(var, batchSize);

	if (printDebug) {
		cout << "inv var" << endl;
		Network::printAllGpu(var, batchSize);
	}

	if (deriv) checkCudaErrors(cudaMemcpy(inv_vars[s], var, arraySizeCol, cudaMemcpyDeviceToDevice));

	//transform activation
	checkCudaErrors(cudaMemcpy(act, prevAct, arraySize, cudaMemcpyDeviceToDevice));
	CudaFunc::addColumnVecMat(act, mean, batchSize, size, batchSize);
	CudaFunc::multColVecMat(act, var, batchSize, size, batchSize);
	CudaFunc::maskByLength(act, d_inputLengths, s, size, batchSize);

	if (printDebug) {
		cout << "normalized" << endl;
		Network::printAllGpu(act, size * batchSize);
	}

	if (deriv) checkCudaErrors(cudaMemcpy(xhats[s], act, arraySize, cudaMemcpyDeviceToDevice));

	CudaFunc::multRowVecMat(act, targetStd, batchSize, size, batchSize);
	CudaFunc::addRowVecMat(act, targetMean, batchSize, size, batchSize);
	CudaFunc::maskByLength(act, d_inputLengths, s, size, batchSize);

	if (printDebug) {
		cout << "after" << endl;
		Network::printAllGpu(act, size * batchSize);
	}
}

void LayerNormCore::calcGrad(dtype2 *prevAct, dtype2 *act, dtype2 *prevError, dtype2 *error,
		dtype2 *grad, dtype2 *params, unsigned int *h_inputLengths,
		unsigned int *d_inputLengths, dtype2 *priorAct, dtype2 *priorError, int s) {
	int arraySize = size * batchSize * sizeof(dtype2);
	int arraySizeSingle = size * sizeof(dtype2);
	int arraySizeCol = batchSize * sizeof(dtype2);
	dtype2* targetStd = params + stdOffset;
	dtype2* targetMeanGrad = grad + paramOffset;
	dtype2* targetStdGrad = grad + stdOffset;

	bool printDebug = (s == print_s);

	dtype2* xhat = xhats[s];

	if (printDebug) {
		cout << name << " calcGrad" << endl;
		cout << "s: " << s << endl;
	}

	if (printDebug) {
		cout << "error" << endl;
		Network::printAllGpu(error, size * batchSize);
	}

	if (printDebug) {
		cout << "xhat" << endl;
		Network::printAllGpu(xhat, size * batchSize);
	}

	//calc grad of target_mean
	CudaFunc::sum_cols_reduce4(error, targetMeanGrad, batchSize, size);

	//calc grad of target_std
	CudaFunc::multiply(xhat, error, intermed1, batchSize * size);
	CudaFunc::sum_cols_reduce4(intermed1, targetStdGrad, batchSize, size);

	// backpropagate error
	checkCudaErrors(cudaMemcpy(intermed1, error, arraySize, cudaMemcpyDeviceToDevice));
	CudaFunc::multRowVecMat(intermed1, targetStd, batchSize, size, batchSize);
	dtype2* dxhat = intermed1;

	if (printDebug) {
		cout << "dxhat" << endl;
		Network::printAllGpu(dxhat, size * batchSize);
	}

	CudaFunc::multiply(dxhat, xhat, intermed2, batchSize * size);

	if (printDebug) {
		cout << "intermed2_1" << endl;
		Network::printAllGpu(intermed2, size * batchSize);
	}

	checkCudaErrors(cudaMemset(singleCol1, 0, arraySizeCol));
	CudaFunc::sum_rows_reduce4(intermed2, singleCol1, batchSize, size);

	if (printDebug) {
		cout << "single1_1" << endl;
		Network::printAllGpu(singleCol1, batchSize);
	}

	checkCudaErrors(cudaMemcpy(intermed2, xhat, arraySize, cudaMemcpyDeviceToDevice));
	CudaFunc::multColVecMat(intermed2, singleCol1, batchSize, size, batchSize);

	if (printDebug) {
		cout << "intermed2_2" << endl;
		Network::printAllGpu(intermed2, size * batchSize);
	}

	checkCudaErrors(cudaMemset(singleCol1, 0, arraySizeCol));
	CudaFunc::sum_rows_reduce4(dxhat, singleCol1, batchSize, size);

	if (printDebug) {
		cout << "single1_2" << endl;
		Network::printAllGpu(singleCol1, batchSize);
	}

	CudaFunc::addColumnVecMat(intermed2, singleCol1, batchSize, size, batchSize);
	CudaFunc::maskByLength(intermed2, d_inputLengths, s, size, batchSize);

	if (printDebug) {
		cout << "intermed2_3" << endl;
		Network::printAllGpu(intermed2, size * batchSize);
	}

	dtypeh a = size;
	CublasFunc::scal(handle, size * batchSize, &a, dxhat, 1);
	if (printDebug) {
		cout << "N*dxhat" << endl;
		Network::printAllGpu(dxhat, size * batchSize);
	}

	CublasFunc::axpy(handle, batchSize * size, &minus_one, intermed2, 1, intermed1, 1);

	if (printDebug) {
		cout << "N*dxhat-intermed2_3" << endl;
		Network::printAllGpu(intermed1, size * batchSize);
	}


	dtype2* inv_var = inv_vars[s];

	if (printDebug) {
		cout << "inv_var" << endl;
		Network::printAllGpu(inv_var, batchSize);
	}

	a = 1.0f / size;
	CublasFunc::scal(handle, batchSize, &a, inv_var, 1);
	CudaFunc::multColVecMat(intermed1, inv_var, batchSize, size, batchSize);

	if (printDebug) {
		cout << "error" << endl;
		Network::printAllGpu(intermed1, size * batchSize);
	}

	CublasFunc::axpy(handle, batchSize * size, &one, intermed1, 1, prevError, 1);
}

} /* namespace netlib */
