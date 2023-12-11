/**
 * @file
 * @brief Defines AddCore class, an addition layer core.
 *
 * A network layer core that applies an addition operation to its two inputs. Used with BaseLayer.
 *
 */

#include "AddCore.h"

#include "../Layer.h"
#include "../../gpu/CudaFunc.h"
#include "../../gpu/CublasFunc.h"

namespace netlib {

AddCore::AddCore(string name, cublasHandle_t& handle, int batchSize, int size, int seqLength)
			: Core(name, handle, batchSize, size, seqLength) {
}

AddCore::~AddCore() {
}

int AddCore::getNParams() {
	return 0;
}

void AddCore::forward(dtype2 *prevAct, dtype2 *act, dtype2 *params,
		unsigned int *h_inputLengths, unsigned int *d_inputLengths, bool deriv,
		dtype2 *priorAct, int s) {
	dtype2 *prev2Act = prev2->activation[s];
	CudaFunc::add(prevAct, prev2Act, act, batchSize * size);
}

void AddCore::calcGrad(dtype2 *prevAct, dtype2 *act, dtype2 *prevError, dtype2 *error,
		dtype2 *grad, dtype2 *params, unsigned int *h_inputLengths,
		unsigned int *d_inputLengths, dtype2 *priorAct, dtype2 *priorError, int s) {
	dtype2* prev2Error = prev2->error[s];
	CublasFunc::axpy(handle, batchSize * size, &one, error, 1, prevError, 1);
	CublasFunc::axpy(handle, batchSize * size, &one, error, 1, prev2Error, 1);
}

void AddCore::setPrev2(Layer *p) {
	prev2 = p;
}

} /* namespace netlib */
