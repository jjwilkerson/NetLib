/**
 * @file
 * @brief Defines DupInLayer class, which converts sequential data to singular data.
 *
 * Converts sequential data to singular data by using the input lengths to pick the last vector in the sequence.
 */

#include "DupInLayer.h"

#include "../Network.h"
#include "../gpu/CudaFunc.h"
#include "../gpu/CublasFunc.h"
#include <iostream>

using namespace std;
namespace netlib {

DupInLayer::DupInLayer(string name, cublasHandle_t& handle, int batchSize, int size, Nonlinearity* nonlinearity, float dropout)
		: Layer(name, handle, batchSize, size, 1, nonlinearity, dropout) {
	nParams = 0;
}

DupInLayer::~DupInLayer() {
}

int DupInLayer::getNParams() {
	return nParams;
}

void DupInLayer::setParamOffset(int offset) {
}

void DupInLayer::forward(int batchNum, dtype2* params, bool deriv, bool stochasticDropout) {
	unsigned int** d_inputLengths = net->getDInputLengths();
	for (int s = 0; s < prev->seqLength; s++) { //TODO: eliminate loop
		CudaFunc::dupIn(activation[0], prev->activation[s], d_inputLengths[batchNum], s, size, batchSize);
		if (deriv) {
			CudaFunc::dupIn(d_activation[0], prev->d_activation[s], d_inputLengths[batchNum], s, size, batchSize);
		}
	}

	if (next != NULL) {
		next->forward(batchNum, params, deriv, stochasticDropout);
	}
}

void DupInLayer::calcGrad(dtype2* grad, int batchNum) {
	unsigned int** d_inputLengths = net->getDInputLengths();
	dtype2** prevError = prev->error;
	int seqEnd = prev->seqLength - 1;

	for (int s = seqEnd; s >= 0; s--) {
		CudaFunc::dupIn(prevError[s], error[0], d_inputLengths[batchNum], s, size, batchSize);
	}

	prev->calcGrad(grad, batchNum);
}

void DupInLayer::addGradL2(dtype2* grad, dtypeh l2) {
}

void DupInLayer::Rforward(dtype2* v, int batchNum) {
	unsigned int** d_inputLengths = net->getDInputLengths();
	for (int s = 0; s < prev->seqLength; s++) { //TODO: eliminate loop
		CudaFunc::dupIn(R_activation[0], prev->R_activation[s], d_inputLengths[batchNum], s, size, batchSize);
	}

	next->Rforward(v, batchNum);
}

void DupInLayer::Rback(dtype2* Gv, int batchNum) {
	unsigned int** d_inputLengths = net->getDInputLengths();
	dtype2** prevError = prev->error;
	int seqEnd = prev->seqLength - 1;

	for (int s = seqEnd; s >= 0; s--) {
		CudaFunc::dupIn(prevError[s], error[0], d_inputLengths[batchNum], s, size, batchSize);
	}

	prev->Rback(Gv, batchNum);
}

} /* namespace netlib */
