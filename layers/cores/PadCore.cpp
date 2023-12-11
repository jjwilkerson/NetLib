/**
 * @file
 * @brief Defines PadLayer class, a layer core that pads its input to a given size.
 *
 */

#include "PadCore.h"

#include "../../gpu/CublasFunc.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>

namespace netlib {

PadCore::PadCore(string name, cublasHandle_t &handle, int batchSize, int size,
		int seqLength) : Core(name, handle, batchSize, size, seqLength) {
}

PadCore::~PadCore() {
}

int PadCore::getNParams() {
	return 0;
}

void PadCore::forward(dtype2 *prevAct, dtype2 *act, dtype2 *params,
		unsigned int *h_inputLengths, unsigned int *d_inputLengths, bool deriv,
		dtype2 *priorAct, int s) {
	int arraySize = batchSize * preSize * sizeof(dtype2);
	checkCudaErrors(cudaMemcpy(act, prevAct, arraySize, cudaMemcpyDeviceToDevice));
}

void PadCore::calcGrad(dtype2 *prevAct, dtype2 *act, dtype2 *prevError, dtype2 *error,
		dtype2 *grad, dtype2 *params, unsigned int *h_inputLengths,
		unsigned int *d_inputLengths, dtype2 *priorAct, dtype2 *priorError, int s) {
	CublasFunc::axpy(handle, batchSize * preSize, &one, error, 1, prevError, 1);
}

} /* namespace netlib */
