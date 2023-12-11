/**
 * @file
 * @brief Defines RecLayer class, a recurrent layer.
 *
 */

#include "RecLayer.h"

#include "../Network.h"
#include "../gpu/CudaFunc.h"
#include "../gpu/CublasFunc.h"
#include <iostream>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "../config/WeightInit.h"
#include "../loss/LossFunction.h"
#include "../nonlinearity/Nonlinearity.h"

using namespace std;

namespace netlib {

RecLayer::RecLayer(string name, cublasHandle_t& handle, int batchSize, int size, int seqLength, Nonlinearity* nonlinearity, float dropout)
			: Layer(name, handle, batchSize, size, seqLength, nonlinearity, dropout) {
}

RecLayer::~RecLayer() {
	// TODO Auto-generated destructor stub
}

int RecLayer::getNParams() {
	if (nParams == 0) {
		int ffParams = (prev->size + 1) * size;
		int recParams = (size + 1) * size;
		nParams = ffParams + recParams;
	}
	return nParams;
}

void RecLayer::setParamOffset(int offset) {
	paramOffset = offset;
	biasOffset = paramOffset + prev->size * size;
	recParamOffset = biasOffset + size;
	recBiasOffset = recParamOffset + size * size;
	cout << name << endl;
	cout << "paramOffset: " << paramOffset << endl;
	cout << "biasOffset: " << biasOffset << endl;
	cout << "recParamOffset: " << recParamOffset << endl;
	cout << "recBiasOffset: " << recBiasOffset << endl << endl;
}

void RecLayer::initMem(bool training, bool optHF) {
	Layer::initMem(training);

	if (training) {
		int arraySize = batchSize * size * sizeof(dtype2);
		checkCudaErrors(cudaMalloc((void **)&recShared1, arraySize));
		checkCudaErrors(cudaMalloc((void **)&recShared2, arraySize));
	}
}

void RecLayer::freeMem(bool training) {
	Layer::freeMem(training);

	if (training) {
		checkCudaErrors(cudaFree(recShared1));
		checkCudaErrors(cudaFree(recShared2));
	}
}

void RecLayer::initWeights(WeightInit& ffWeightInit, WeightInit& recWeightInit) {
	dtype1* params = net->getMasterParams();
	ffWeightInit.initialize(params+paramOffset, prev->size, size);
	checkCudaErrors(cudaMemset(params+biasOffset, 0, size * sizeof(dtype1)));
	recWeightInit.initialize(params+recParamOffset, size, size);
	checkCudaErrors(cudaMemset(params+recBiasOffset, 0, size * sizeof(dtype1)));
}

void RecLayer::forward(int batchNum, dtype2* params, bool deriv, bool stochasticDropout) {
	if (params == NULL) {
		params = W;
	}

	dtype2* w = params + paramOffset;
	dtype2* b = params + biasOffset;
	int preSize = prev->size;

	for (int s = 0; s < seqLength; s++) {
		dtype2* act = activation[s];

		int seqIx;
		if (prev->seqLength == 1) {
			seqIx = 0;
		} else {
			seqIx = s;
		}

		dtype2* doact;
		if (prev->dropout == 0) {
			doact = prev->activation[seqIx];
		} else {
			doact = prev->doActivation;
			int arraySize = batchSize * preSize * sizeof(dtype2);
			checkCudaErrors(cudaMemcpy(doact,  prev->activation[seqIx], arraySize,
					cudaMemcpyDeviceToDevice));

			if (stochasticDropout) {
				CudaFunc::multiply(doact, prev->dropoutMask, doact, batchSize * preSize);
			}
		}

		CublasFunc::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, batchSize, size, preSize, &one,
				doact, batchSize, w, preSize, &zero, ffInput, batchSize);
		CudaFunc::addRowVecMat(ffInput, b, batchSize, size, batchSize);

		//compute recurrent input
		if (s > 0) {
			dtype2* w = params + recParamOffset;
			CublasFunc::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, batchSize, size, size, &one,
					activation[s-1], batchSize, w, size, &one, ffInput, batchSize);
		} else {
			dtype2* b = params + recBiasOffset;
			CudaFunc::addRowVecMat(ffInput, b, batchSize, size, batchSize);
		}

		nonlinearity->activation(size, ffInput, act);

		applyMask(act, size, s, batchNum);

		if (deriv) {
			dtype2* d_act = d_activation[s];
			nonlinearity->d_activation(size, ffInput, act, d_act);
			applyMask(d_act, size, s, batchNum);
		}
	}

	if (next != NULL) {
		next->forward(batchNum, params, deriv, stochasticDropout);
	}
}

void RecLayer::calcGrad(dtype2* grad, int batchNum) {
	dtype2* delta = singleShared1;
	dtype2** prevError = prev->error;

	int seqEnd = seqLength - 1;

	for (int s = seqEnd; s >= 0; s--) {
		if (isOutputLayer()) {
			net->lossFunction.d_loss(net->getTargets(), this, s, error[s], true, net->matchMasks(batchNum));
		}

//		applyMask(currError, size, s, batchNum); //should apply mask to delta instead?

		// compute deltas
		CudaFunc::multiply(d_activation[s], error[s], delta, batchSize * size);
		applyMask(delta, size, s, batchNum); //?

		// gradient for recurrent weights
		if (s > 0) {
			dtype2* wGrad = grad + recParamOffset;
			CublasFunc::gemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, size, size, batchSize, &one,
					activation[s-1], batchSize, delta, batchSize, &one, wGrad, size);
		} else {
			//put remaining gradient into initial bias
			dtype2* bGrad = grad + recBiasOffset;
		    CudaFunc::sum_cols_reduce4(delta, bGrad, batchSize, size);
		}

		// feedforward connections
		dtype2* wGrad = grad + paramOffset;
		dtype2* bGrad = grad + biasOffset;

		int seqIx;
		if (prev->seqLength == 1) {
			seqIx = 0;
		} else {
			seqIx = s;
		}

		int preSize = prev->size;

		dtype2* doact;
		if (prev->dropout == 0) {
			doact = prev->activation[seqIx];
		} else {
			doact = prev->doActivation;
			int arraySize = batchSize * preSize * sizeof(dtype2);
			checkCudaErrors(cudaMemcpy(doact,  prev->activation[seqIx], arraySize,
					cudaMemcpyDeviceToDevice));

			CudaFunc::multiply(doact, prev->dropoutMask, doact, batchSize * preSize);
		}

		CublasFunc::gemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, preSize, size, batchSize, &one,
				doact, batchSize, delta, batchSize, &one, wGrad, preSize);

		CudaFunc::sum_cols_reduce4(delta, bGrad, batchSize, size);

		// backpropagate error
		dtype2* w = W + paramOffset;

		if (prev->dropout == 0) {
			CublasFunc::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, batchSize, preSize, size, &one,
					delta, batchSize, w, preSize, &one, prevError[seqIx], batchSize);
		} else {
			dtype2* doerror = prev->doActivation;
			int arraySize = batchSize * preSize * sizeof(dtype2);
			CublasFunc::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, batchSize, preSize, size, &one,
					delta, batchSize, w, preSize, &zero, doerror, batchSize);
			CudaFunc::multiply(doerror, prev->dropoutMask, doerror, batchSize * preSize);
			CublasFunc::axpy(handle, batchSize * preSize, &one, doerror, 1, prevError[seqIx], 1);
		}

		// add recurrent error for step before
		if (s > 0) {
			dtype2* w = W + recParamOffset;
			CublasFunc::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, batchSize, size, size, &one,
					delta, batchSize, w, size, &one, error[s-1], batchSize);
		}
	}

	prev->calcGrad(grad, batchNum);
}

void RecLayer::addGradL2(dtype2* grad, dtypeh l2) {
	int numWeights = biasOffset - paramOffset;
	CublasFunc::axpy(handle, numWeights, &l2, W+paramOffset, 1, grad+paramOffset, 1);

	numWeights = recBiasOffset - recParamOffset;
	CublasFunc::axpy(handle, numWeights, &l2, W+recParamOffset, 1, grad+recParamOffset, 1);
}

void RecLayer::Rforward(dtype2* v, int batchNum) {
	dtype2* w = W + paramOffset;
	dtype2* vw = v + paramOffset;
	dtype2* vb = v + biasOffset;
	int preSize = prev->size;

	for (int s = 0; s < seqLength; s++) {
		dtype2* R_act = R_activation[s];

		int seqIx;
		if (prev->seqLength == 1) {
			seqIx = 0;
		} else {
			seqIx = s;
		}

		dtype2* doact;
		if (prev->dropout == 0) {
			doact = prev->activation[seqIx];
		} else {
			doact = prev->doActivation;
			int arraySize = batchSize * preSize * sizeof(dtype2);
			checkCudaErrors(cudaMemcpy(doact,  prev->activation[seqIx], arraySize,
					cudaMemcpyDeviceToDevice));

			CudaFunc::multiply(doact, prev->dropoutMask, doact, batchSize * preSize);
		}

		CublasFunc::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, batchSize, size, preSize, &one,
				doact, batchSize, vw, preSize, &one, R_act, batchSize);
		CudaFunc::addRowVecMat(R_act, vb, batchSize, size, batchSize);

		if (prev->dropout == 0) {
			doact = prev->R_activation[seqIx];
		} else {
			doact = prev->doActivation;
			int arraySize = batchSize * preSize * sizeof(dtype2);
			checkCudaErrors(cudaMemcpy(doact,  prev->R_activation[seqIx], arraySize,
					cudaMemcpyDeviceToDevice));

			CudaFunc::multiply(doact, prev->dropoutMask, doact, batchSize * preSize);
		}

		CublasFunc::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, batchSize, size, preSize, &one,
				doact, batchSize, w, preSize, &one, R_act, batchSize);

		//compute recurrent input
		if (s > 0) {
			dtype2* wrec = W + recParamOffset;
			dtype2* vrec = v + recParamOffset;
			CublasFunc::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, batchSize, size, size, &one,
					activation[s-1], batchSize, vrec, size, &one, R_act, batchSize);
			CublasFunc::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, batchSize, size, size, &one,
					R_activation[s-1], batchSize, wrec, size, &one, R_act, batchSize);
		} else {
			dtype2* vb = v + recBiasOffset;
			CudaFunc::addRowVecMat(R_act, vb, batchSize, size, batchSize);
		}

		if (!nonlinearity->isLinear()) {
			CudaFunc::multiply(d_activation[s], R_act, R_act, batchSize * size);
		}

		if (seqLength > 1) {
			applyMask(R_act, size, s, batchNum);
		}
	}

	if (next != NULL) {
		next->Rforward(v, batchNum);
	}
}

void RecLayer::Rback(dtype2* Gv, int batchNum) {
	dtype2* delta = singleShared1;
	dtype2* d2_loss = singleShared2;
	dtype2** prevError = prev->error;

	int seqEnd = seqLength - 1;

	for (int s = seqEnd; s >= 0; s--) {
		net->lossFunction.d2_loss(NULL, this, s, d2_loss, false, net->matchMasks(batchNum));
		CudaFunc::multiply(d2_loss, R_activation[s], d2_loss, batchSize * size);
		CublasFunc::axpy(handle, batchSize * size, &one, d2_loss, 1, error[s], 1);

//		applyMask(currError, size, s, batchNum); //should apply mask to delta instead?

		// compute deltas
		CudaFunc::multiply(d_activation[s], error[s], delta, batchSize * size);
		applyMask(delta, size, s, batchNum); //?

		// gradient for recurrent weights
		if (s > 0) {
			dtype2* wGv = Gv + recParamOffset;
			CublasFunc::gemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, size, size, batchSize, &one,
					activation[s-1], batchSize, delta, batchSize, &one, wGv, size);
		} else {
			//put remaining gradient into initial bias
			dtype2* bGv = Gv + recBiasOffset;
		    CudaFunc::sum_cols_reduce4(delta, bGv, batchSize, size);
		}

		// feedforward connections
		dtype2* wGv = Gv + paramOffset;
		dtype2* bGv = Gv + biasOffset;

		int seqIx;
		if (prev->seqLength == 1) {
			seqIx = 0;
		} else {
			seqIx = s;
		}

		int preSize = prev->size;

		dtype2* doact;
		if (prev->dropout == 0) {
			doact = prev->activation[seqIx];
		} else {
			doact = prev->doActivation;
			int arraySize = batchSize * preSize * sizeof(dtype2);
			checkCudaErrors(cudaMemcpy(doact,  prev->activation[seqIx], arraySize,
					cudaMemcpyDeviceToDevice));

			CudaFunc::multiply(doact, prev->dropoutMask, doact, batchSize * preSize);
		}

		CublasFunc::gemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, preSize, size, batchSize, &one,
				doact, batchSize, delta, batchSize, &one, wGv, preSize);

		CudaFunc::sum_cols_reduce4(delta, bGv, batchSize, size);

		// backpropagate error
		dtype2* w = W + paramOffset;

		if (prev->dropout == 0) {
			CublasFunc::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, batchSize, preSize, size, &one,
					delta, batchSize, w, preSize, &one, prevError[seqIx], batchSize);
		} else {
			dtype2* doerror = prev->doActivation;
			int arraySize = batchSize * preSize * sizeof(dtype2);
			CublasFunc::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, batchSize, preSize, size, &one,
					delta, batchSize, w, preSize, &zero, doerror, batchSize);
			CudaFunc::multiply(doerror, prev->dropoutMask, doerror, batchSize * preSize);
			CublasFunc::axpy(handle, batchSize * preSize, &one, doerror, 1, prevError[seqIx], 1);
		}

		// add recurrent error for step before
		if (s > 0) {
			dtype2* w = W + recParamOffset;
			CublasFunc::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, batchSize, size, size, &one,
					delta, batchSize, w, size, &one, error[s-1], batchSize);
		}
	}

	prev->Rback(Gv, batchNum);
}

bool RecLayer::hasParam(int i) {
	if (Layer::hasParam(i)) {
		return true;
	}

	return (i >= recParamOffset) && (i < (recBiasOffset+size));
}

bool RecLayer::isWeight(int i) {
	if (Layer::isWeight(i)) {
		return true;
	}

	return (i >= recParamOffset) && (i < recBiasOffset);
}

} /* namespace netlib */

