/**
 * @file
 * @brief Defines ConcatLayer class, a concatenation layer.
 *
 * A network layer that applies a concatenation operation to its two inputs.
 *
 */

#include "Layer.h"
#include "ConcatLayer.h"
#include "../gpu/CublasFunc.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>

namespace netlib {

ConcatLayer::ConcatLayer(string name, cublasHandle_t& handle, int batchSize, int size, int seqLength,
		Nonlinearity* nonlinearity, float dropout)
		: Layer(name, handle, batchSize, size, seqLength, nonlinearity, dropout) {
	nParams = 0;
}

ConcatLayer::~ConcatLayer() {
}

int ConcatLayer::getNParams() {
	return nParams;
}

void ConcatLayer::setParamOffset(int offset) {
}

void ConcatLayer::setPrev2(Layer* p) {
	prev2 = p;
	p->next = this;
}


void ConcatLayer::forward(int batchNum, dtype2* params, bool deriv, bool stochasticDropout) {
	twoForward = !twoForward;
	if (!twoForward) {
		return;
	}

	int preSize1 = prev->size;
	int preSize2 = prev2->size;

	for (int s = 0; s < seqLength; s++) {
		dtype2* act = activation[s];
		dtype2* act2 = act + (batchSize * preSize1);

		int seq1 = (prev->seqLength == 1)? 0 : s;
		int seq2 = (prev2->seqLength == 1)? 0 : s;

		int arraySize1 = batchSize * preSize1 * sizeof(dtype2);
		int arraySize2 = batchSize * preSize2 * sizeof(dtype2);
		checkCudaErrors(cudaMemcpy(act,  prev->activation[seq1], arraySize1, cudaMemcpyDeviceToDevice));
		checkCudaErrors(cudaMemcpy(act2,  prev2->activation[seq2], arraySize2, cudaMemcpyDeviceToDevice));

		if (deriv) {
			dtype2* d_act = d_activation[s];
			dtype2* d_act2 = d_act + (batchSize * preSize1);
			checkCudaErrors(cudaMemcpy(d_act,  prev->d_activation[seq1], arraySize1, cudaMemcpyDeviceToDevice));
			checkCudaErrors(cudaMemcpy(d_act2,  prev2->d_activation[seq2], arraySize2, cudaMemcpyDeviceToDevice));
		}
	}

	if (next != NULL) {
		next->forward(batchNum, params, deriv, stochasticDropout);
	}
}

void ConcatLayer::calcGrad(dtype2* grad, int batchNum) {
	int preSize1 = prev->size;
	int preSize2 = prev2->size;
	int arraySize1 = batchSize * preSize1 * sizeof(dtype2);
	int arraySize2 = batchSize * preSize2 * sizeof(dtype2);

	for (int s = 0; s < seqLength; s++) {
		int seq1 = (prev->seqLength == 1)? 0 : s;
		int seq2 = (prev2->seqLength == 1)? 0 : s;

		dtype2* prevError = prev->error[seq1];
		dtype2* prev2Error = prev2->error[seq2];

		dtype2* errorRev = error[s] + (batchSize * preSize1);

		CublasFunc::axpy(handle, batchSize * prev->size, &one, error[s], 1, prevError, 1);
		CublasFunc::axpy(handle, batchSize * prev2->size, &one, errorRev, 1, prev2Error, 1);
	}

	prev->calcGrad(grad, batchNum);
	prev2->calcGrad(grad, batchNum);
}

void ConcatLayer::addGradL2(dtype2* grad, dtypeh l2) {
}

void ConcatLayer::Rforward(dtype2* v, int batchNum) {
	twoRforward = !twoRforward;
	if (!twoRforward) {
		return;
	}

	int preSize1 = prev->size;
	int preSize2 = prev2->size;

	int s = 0;
	dtype2* Ract = R_activation[s];
	dtype2* Ract2 = Ract + (batchSize * preSize1);

	int arraySize1 = batchSize * preSize1 * sizeof(dtype2);
	int arraySize2 = batchSize * preSize2 * sizeof(dtype2);
	checkCudaErrors(cudaMemcpy(Ract,  prev->R_activation[s], arraySize1, cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpy(Ract2,  prev2->R_activation[s], arraySize2, cudaMemcpyDeviceToDevice));

	if (next != NULL) {
		next->Rforward(v, batchNum);
	}
}

void ConcatLayer::Rback(dtype2* Gv, int batchNum) {
	dtype2* prevError = prev->error[0];
	dtype2* prev2Error = prev2->error[0];

	int preSize1 = prev->size;
	int preSize2 = prev2->size;
	int arraySize1 = batchSize * preSize1 * sizeof(dtype2);
	int arraySize2 = batchSize * preSize2 * sizeof(dtype2);

	dtype2* errorRev = error[0] + (batchSize * preSize1);

	checkCudaErrors(cudaMemcpy(prevError, error[0], arraySize1, cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpy(prev2Error, errorRev, arraySize2, cudaMemcpyDeviceToDevice));

	prev->Rback(Gv, batchNum);
	prev2->Rback(Gv, batchNum);
}

} /* namespace netlib */
