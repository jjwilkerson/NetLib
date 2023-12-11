/**
 * @file
 * @brief Declares PadLayer class, a layer core that pads its input to a given size.
 *
 */

#ifndef PADCORE_H_
#define PADCORE_H_

#include "Core.h"

namespace netlib {

/**
 * @brief A layer core that pads its input to a given size.
 *
 */
class PadCore: public Core {
public:
	PadCore(string name, cublasHandle_t& handle, int batchSize, int size, int seqLength);
	virtual ~PadCore();
	int getNParams();
	void forward(dtype2* prevAct, dtype2* act, dtype2* params, unsigned int* h_inputLengths,
			unsigned int* d_inputLengths, bool deriv = false, dtype2* priorAct = NULL, int s = 0);
	void calcGrad(dtype2* prevAct, dtype2 *act, dtype2* prevError, dtype2* error, dtype2* grad, dtype2* params,
			unsigned int* h_inputLengths, unsigned int* d_inputLengths, dtype2 *priorAct = NULL, dtype2* priorError = NULL,
			int s = 0);
};

} /* namespace netlib */

#endif /* PADCORE_H_ */
