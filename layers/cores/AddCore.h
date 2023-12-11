/**
 * @file
 * @brief Declares AddCore class, an addition layer core.
 *
 * A network layer core that applies an addition operation to its two inputs. Used with BaseLayer.
 *
 */

#ifndef ADDCORE_H_
#define ADDCORE_H_

#include "Core.h"

namespace netlib {

class Layer;

/**
 * @brief An addition layer core.
 *
 * A network layer core that applies an addition operation to its two inputs. Used with BaseLayer.
 *
 */
class AddCore: public Core {
public:
	AddCore(string name, cublasHandle_t& handle, int batchSize, int size, int seqLength);
	virtual ~AddCore();
	int getNParams();
	void forward(dtype2* prevAct, dtype2* act, dtype2* params, unsigned int* h_inputLengths,
			unsigned int* d_inputLengths, bool deriv = false, dtype2* priorAct = NULL, int s = 0);
	void calcGrad(dtype2* prevAct, dtype2 *act, dtype2* prevError, dtype2* error, dtype2* grad, dtype2* params,
			unsigned int* h_inputLengths, unsigned int* d_inputLengths, dtype2 *priorAct = NULL, dtype2* priorError = NULL,
			int s = 0);
	void setPrev2(Layer* p);

	Layer *prev2 = NULL;
};

} /* namespace netlib */

#endif /* ADDCORE_H_ */
