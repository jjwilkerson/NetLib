/**
 * @file
 * @brief Defines SplitCore class, a layer core that receives an input and passes it to two separate downstream classes.
 *
 */

#include "SplitCore.h"

#include "../../gpu/CublasFunc.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>

namespace netlib {

SplitCore::SplitCore(string name, cublasHandle_t& handle, int batchSize, int size, int seqLength)
					: Core(name, handle, batchSize, size, seqLength) {
}

SplitCore::~SplitCore() {
}

int SplitCore::getNParams() {
	return 0;
}

void SplitCore::forward(dtype2 *prevAct, dtype2 *act, dtype2 *params,
		unsigned int *h_inputLengths, unsigned int *d_inputLengths, bool deriv,
		dtype2 *priorAct, int s) {
	int arraySize = batchSize * size * sizeof(dtype2);
	checkCudaErrors(cudaMemcpy(act, prevAct, arraySize, cudaMemcpyDeviceToDevice));
}

void SplitCore::calcGrad(dtype2 *prevAct, dtype2 *act, dtype2 *prevError, dtype2 *error,
		dtype2 *grad, dtype2 *params, unsigned int *h_inputLengths,
		unsigned int *d_inputLengths, dtype2 *priorAct, dtype2 *priorError, int s) {
	CublasFunc::axpy(handle, batchSize * size, &one, error, 1, prevError, 1);
}

} /* namespace netlib */
