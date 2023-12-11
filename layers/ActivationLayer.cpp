/**
 * @file
 * @brief Defines ActivationLayer class, an activation layer.
 *
 * A network layer core that simply applies a nonlinear activation to its input.
 */

#include "ActivationLayer.h"
#include "../Network.h"
#include "../gpu/CudaFunc.h"
#include "../gpu/CublasFunc.h"
#include <cassert>
#include <iostream>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "../loss/LossFunction.h"
#include "../nonlinearity/Nonlinearity.h"

using namespace std;

namespace netlib {

ActivationLayer::ActivationLayer(string name, cublasHandle_t& handle, int batchSize, int size, int seqLength,
		Nonlinearity* nonlinearity)
		: Layer(name, handle, batchSize, size, seqLength, nonlinearity, 0.0f) {
	nParams = 0;
}

ActivationLayer::~ActivationLayer() {
	// TODO Auto-generated destructor stub
}

int ActivationLayer::getNParams() {
	return nParams;
}

void ActivationLayer::setParamOffset(int offset) {
}

void ActivationLayer::forward(int batchNum, dtype2* params, bool deriv,
		bool stochasticDropout) {
//	cout << "ActivationLayer::forward" << endl;
	if (stochasticOverrideSet) stochasticDropout = stochasticOverrideValue;

	int preSize = prev->size;
	assert (size == prev->size);

	int arraySize = batchSize * size * sizeof(dtype2);
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
			checkCudaErrors(cudaMemcpy(doact,  prev->activation[seqIx], arraySize,
					cudaMemcpyDeviceToDevice));

			CudaFunc::multiply(doact, prev->dropoutMask, doact, batchSize * preSize);
		}

		nonlinearity->activation(size, doact, act);

		if (seqLength > 1) {
			applyMask(act, size, s, batchNum);
		}

		if (deriv) {
			dtype2* d_act = d_activation[s];
			nonlinearity->d_activation(size, prev->activation[s], act, d_act);
			if (seqLength > 1) {
				applyMask(d_act, size, s, batchNum);
			}
		}
	}

	if (next != NULL) {
		next->forward(batchNum, params, deriv, stochasticDropout);
	}
}

void ActivationLayer::calcGrad(dtype2* grad, int batchNum) {
	dtype2* delta = singleShared1;
	dtype2** prevError = prev->error;
	int preSize = prev->size;

	int seqEnd = seqLength - 1;

	for (int s = seqEnd; s >= 0; s--) {
		if (isOutputLayer()) {
			net->lossFunction.d_loss(net->getTargets(), this, s, error[s], true, net->matchMasks(batchNum));
		}

		// compute deltas
		CudaFunc::multiply(d_activation[s], error[s], delta, batchSize * size);
		applyMask(delta, size, s, batchNum);

		if (prev->dropout == 0) {
			CublasFunc::axpy(handle, batchSize * preSize, &one, delta, 1, prevError[s], 1);
		} else {
			dtype2* doerror = prev->doActivation;
			int arraySize = batchSize * preSize * sizeof(dtype2);
			checkCudaErrors(cudaMemcpy(doerror,  delta, arraySize, cudaMemcpyDeviceToDevice));
			CudaFunc::multiply(doerror, prev->dropoutMask, doerror, batchSize * preSize);
			CublasFunc::axpy(handle, batchSize * preSize, &one, doerror, 1, prevError[s], 1);
		}
	}

	prev->calcGrad(grad, batchNum);
}

void ActivationLayer::addGradL2(dtype2* grad, dtypeh l2) {
}

void ActivationLayer::Rforward(dtype2* v, int batchNum) {
	int preSize = prev->size;
	assert (size == prev->size);

	int arraySize = batchSize * size * sizeof(dtype2);
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
			doact = prev->R_activation[seqIx];
		} else {
			doact = prev->doActivation;
			int arraySize = batchSize * preSize * sizeof(dtype2);
			checkCudaErrors(cudaMemcpy(doact,  prev->R_activation[seqIx], arraySize,
					cudaMemcpyDeviceToDevice));

			CudaFunc::multiply(doact, prev->dropoutMask, doact, batchSize * preSize);
		}

		checkCudaErrors(cudaMemcpy(R_act,  doact, arraySize, cudaMemcpyDeviceToDevice));

		CudaFunc::multiply(d_activation[s], R_act, R_act, batchSize * size);

		if (seqLength > 1) {
			applyMask(R_act, size, s, batchNum);
		}
	}

	if (next != NULL) {
		next->Rforward(v, batchNum);
	}
}

void ActivationLayer::Rback(dtype2* Gv, int batchNum) {
	dtype2* delta = singleShared1;
	dtype2* d2_loss = singleShared2;
	dtype2** prevError = prev->error;
	int preSize = prev->size;

	int seqEnd = seqLength - 1;

	for (int s = seqEnd; s >= 0; s--) {
		net->lossFunction.d2_loss(NULL, this, s, d2_loss, false, net->matchMasks(batchNum));
		CudaFunc::multiply(d2_loss, R_activation[s], d2_loss, batchSize * size);
		CublasFunc::axpy(handle, batchSize * size, &one, d2_loss, 1, error[s], 1);

		// compute deltas
		CudaFunc::multiply(d_activation[s], error[s], delta, batchSize * size);
		applyMask(delta, size, s, batchNum);

		if (prev->dropout == 0) {
			CublasFunc::axpy(handle, batchSize * preSize, &one, delta, 1, prevError[s], 1);
		} else {
			dtype2* doerror = prev->doActivation;
			int arraySize = batchSize * preSize * sizeof(dtype2);
			checkCudaErrors(cudaMemcpy(doerror,  delta, arraySize, cudaMemcpyDeviceToDevice));
			CudaFunc::multiply(doerror, prev->dropoutMask, doerror, batchSize * preSize);
			CublasFunc::axpy(handle, batchSize * preSize, &one, doerror, 1, prevError[s], 1);
		}
	}

	prev->Rback(Gv, batchNum);
}

} /* namespace netlib */
