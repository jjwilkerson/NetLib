/**
 * @file
 * @brief Defines SeqInputLayer class, an input layer that handles sequential data.
 *
 */

#include "SeqInputLayer.h"

#include "../Network.h"
#include "../gpu/CudaFunc.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "../nonlinearity/Nonlinearity.h"
#include "InputLayer.h"

namespace netlib {

SeqInputLayer::SeqInputLayer(string name, cublasHandle_t& handle, int batchSize, int size, int seqLength, Nonlinearity* nonlinearity, float dropout,
		bool dirFwd)
		: InputLayer(name, handle, batchSize, size, seqLength, nonlinearity, dropout), dirFwd(dirFwd) {
	nParams = 0;
}

SeqInputLayer::~SeqInputLayer() {
	// TODO Auto-generated destructor stub
}

int SeqInputLayer::getNParams() {
	return nParams;
}

void SeqInputLayer::forward(int batchNum, dtype2* params, bool deriv, bool stochasticDropout) {
	dtype2 **d_inputs;
	if (dirFwd) {
		d_inputs = net->getFwdInputs();
	} else {
		d_inputs = net->getRevInputs();
	}

	int arraySize = batchSize * size * sizeof(dtype2);
	for (int s = 0; s < seqLength; s++) {
		dtype2* act = activation[s];
        checkCudaErrors(cudaMemcpy(ffInput, d_inputs[s], arraySize, cudaMemcpyDeviceToDevice));

		nonlinearity->activation(size, ffInput, act);
		applyMask(act, size, s, batchNum);
	}

	if (next != NULL) {
		next->forward(batchNum, params, deriv, stochasticDropout);
	}
}

void SeqInputLayer::Rforward(dtype2* v, int batchNum) {
	for (int s = 0; s < seqLength; s++) {
		dtype2* R_act = R_activation[s];

		if (!nonlinearity->isLinear()) {
			CudaFunc::multiply(d_activation[s], R_act, R_act, batchSize * size);
		}

		if (seqLength > 1) {
			applyMask(R_act, size, s, batchNum);
		}
	}

	next->Rforward(v, batchNum);
}

} /* namespace netlib */
