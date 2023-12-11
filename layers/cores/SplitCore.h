/**
 * @file
 * @brief Declares SplitCore class, a layer core that receives an input and passes it to two separate downstream classes.
 *
 */

#ifndef SPLITCORE_H_
#define SPLITCORE_H_

#include "Core.h"

namespace netlib {

class Layer;

/**
 * @brief A layer core that receives an input and passes it to two separate downstream classes.
 *
 */
class SplitCore: public Core {
public:
	SplitCore(string name, cublasHandle_t& handle, int batchSize, int size, int seqLength);
	virtual ~SplitCore();
	int getNParams();
	void forward(dtype2* prevAct, dtype2* act, dtype2* params, unsigned int* h_inputLengths,
			unsigned int* d_inputLengths, bool deriv = false, dtype2* priorAct = NULL, int s = 0);
	void calcGrad(dtype2* prevAct, dtype2 *act, dtype2* prevError, dtype2* error, dtype2* grad, dtype2* params,
			unsigned int* h_inputLengths, unsigned int* d_inputLengths, dtype2 *priorAct = NULL, dtype2* priorError = NULL,
			int s = 0);
};

} /* namespace netlib */

#endif /* SPLITCORE_H_ */
