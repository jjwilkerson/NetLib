/**
 * @file
 * @brief Defines ActivationCore class, an activation layer core.
 *
 * A network layer core that simply applies a nonlinear activation to its input. Used with BaseLayer.
 *
 */

#include "ActivationCore.h"
#include "../../gpu/CublasFunc.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "../../nonlinearity/Nonlinearity.h"

namespace netlib {

ActivationCore::ActivationCore(string name, cublasHandle_t& handle, int batchSize, int size, int seqLength)
	: Core(name, handle, batchSize, size, seqLength) {
}

ActivationCore::~ActivationCore() {
}

int ActivationCore::getNParams() {
	return 0;
}

void ActivationCore::forward(dtype2* prevAct, dtype2* act, dtype2* params, unsigned int* h_inputLengths,
		unsigned int* d_inputLengths, bool deriv, dtype2* priorAct, int s) {
	int arraySize = batchSize * size * sizeof(dtype2);
	checkCudaErrors(cudaMemcpy(act, prevAct, arraySize, cudaMemcpyDeviceToDevice));
}

void ActivationCore::calcGrad(dtype2 *prevAct, dtype2 *act, dtype2 *prevError, dtype2 *error,
		dtype2 *grad, dtype2 *params, unsigned int *h_inputLengths,
		unsigned int *d_inputLengths, dtype2 *priorAct, dtype2 *priorError, int s) {
	CublasFunc::axpy(handle, batchSize * size, &one, error, 1, prevError, 1);
}

} /* namespace netlib */
