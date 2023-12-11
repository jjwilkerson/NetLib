/**
 * @file
 * @brief Declares ActivationCore class, an activation layer core.
 *
 * A network layer core that simply applies a nonlinear activation to its input. Used with BaseLayer.
 *
 */

#ifndef ACTIVATIONCORE_H_
#define ACTIVATIONCORE_H_

#include "Core.h"

namespace netlib {

class Nonlinearity;

/**
 * @brief An activation layer core.
 *
 * A network layer core that simply applies a nonlinear activation to its input. Used with BaseLayer.
 *
 */
class ActivationCore: public Core {
public:
	ActivationCore(string name, cublasHandle_t& handle, int batchSize, int size, int seqLength);
	virtual ~ActivationCore();
	int getNParams();
	void forward(dtype2* prevAct, dtype2* act, dtype2* params, unsigned int* h_inputLengths,
			unsigned int* d_inputLengths, bool deriv = false, dtype2* priorAct = NULL, int s = 0);
	void calcGrad(dtype2* prevAct, dtype2 *act, dtype2* prevError, dtype2* error, dtype2* grad, dtype2* params,
			unsigned int* h_inputLengths, unsigned int* d_inputLengths, dtype2 *priorAct = NULL, dtype2* priorError = NULL,
			int s = 0);
};

} /* namespace netlib */

#endif /* ACTIVATIONCORE_H_ */
