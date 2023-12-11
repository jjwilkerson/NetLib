/**
 * @file
 * @brief Defines DenseLayer class, a densely-connected layer.
 *
 *
 */

#include "DenseLayer.h"

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

DenseLayer::DenseLayer(string name, cublasHandle_t& handle, int batchSize, int size, int seqLength,
		Nonlinearity* nonlinearity, float dropout, WeightInit* weightInit)
		: Layer(name, handle, batchSize, size, seqLength, nonlinearity, dropout), weightInit(weightInit) {
}

DenseLayer::~DenseLayer() {
}

int DenseLayer::getNParams() {
	if (nParams == 0) {
		if (bnAfter) {
			nParams = prev->size * size;
		} else {
			nParams = (prev->size + 1) * size;
		}
	}
	return nParams;
}

void DenseLayer::setParamOffset(int offset) {
	paramOffset = offset;
	biasOffset = paramOffset + prev->size * size;
	cout << name << endl;
	cout << "paramOffset: " << paramOffset << endl;
	cout << "biasOffset: " << biasOffset << endl << endl;
}

void DenseLayer::initWeights(WeightInit& ffWeightInit, WeightInit& recWeightInit) {
	dtype1* params = net->getMasterParams();
	if (weightInit == NULL) {
		ffWeightInit.initialize(params+paramOffset, prev->size, size);
	} else {
		weightInit->initialize(params+paramOffset, prev->size, size);
	}
	if (!bnAfter) checkCudaErrors(cudaMemset(params+biasOffset, 0, size * sizeof(dtype1)));
}

void DenseLayer::forward(int batchNum, dtype2* params, bool deriv, bool stochasticDropout) {
//	cout << "DenseLayer::forward" << endl;
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
		if (prev->dropout == 0 || !stochasticDropout) { //TODO: move stochasticDropout here for other layer classes
			doact = prev->activation[seqIx];
		} else {
			doact = prev->doActivation;
			int arraySize = batchSize * preSize * sizeof(dtype2);
			checkCudaErrors(cudaMemcpy(doact, prev->activation[seqIx], arraySize,
					cudaMemcpyDeviceToDevice));

			CudaFunc::multiply(doact, prev->dropoutMask, doact, batchSize * preSize);
		}

		CublasFunc::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, batchSize, size, preSize, &one,
				doact, batchSize, w, preSize, &zero, ffInput, batchSize);
		if (!bnAfter) CudaFunc::addRowVecMat(ffInput, b, batchSize, size, batchSize);

		nonlinearity->activation(size, ffInput, act);

		if (seqLength > 1) {
			applyMask(act, size, s, batchNum);
		}

		if (deriv && (!isOutputLayer() || !net->lossFunction.derivCombo())) {
			dtype2* d_act = d_activation[s];
			nonlinearity->d_activation(size, ffInput, act, d_act);
			if (seqLength > 1) {
				applyMask(d_act, size, s, batchNum);
			}
		}
	}

	if (next != NULL) {
		next->forward(batchNum, params, deriv, stochasticDropout);
	}
}

void DenseLayer::calcGrad(dtype2* grad, int batchNum) {
	dtype2* delta = singleShared1;
	dtype2** prevError = prev->error;

	int seqEnd = seqLength - 1;

	for (int s = seqEnd; s >= 0; s--) {
		if (isOutputLayer()) {
			net->lossFunction.d_loss(net->getTargets(), this, s, error[s], true, net->matchMasks(batchNum),
					net->getDInputLengths()[batchNum]);
		}

//		applyMask(currError, size, s, batchNum); //should apply mask to delta instead?

		// compute deltas
		if (isOutputLayer() && net->lossFunction.derivCombo()) {
//			delta = error[s];
			int arraySize = batchSize * size * sizeof(dtype2);
			checkCudaErrors(cudaMemcpy(delta,  error[s], arraySize, cudaMemcpyDeviceToDevice));
		} else {
			CudaFunc::multiply(d_activation[s], error[s], delta, batchSize * size);
			applyMask(delta, size, s, batchNum); //?
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

		if (!bnAfter) CudaFunc::sum_cols_reduce4(delta, bGrad, batchSize, size);

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
	}

	prev->calcGrad(grad, batchNum);
}

void DenseLayer::addGradL2(dtype2* grad, dtypeh l2) {
	int numWeights = biasOffset - paramOffset;
	CublasFunc::axpy(handle, numWeights, &l2, W+paramOffset, 1, grad+paramOffset, 1);
}

void DenseLayer::Rforward(dtype2* v, int batchNum) {
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
		if (!bnAfter) CudaFunc::addRowVecMat(R_act, vb, batchSize, size, batchSize);

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

void DenseLayer::Rback(dtype2* Gv, int batchNum) {
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

		if (!bnAfter) CudaFunc::sum_cols_reduce4(delta, bGv, batchSize, size);

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
	}

	prev->Rback(Gv, batchNum);
}

} /* namespace netlib */
