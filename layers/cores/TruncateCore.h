/**
 * @file
 * @brief Declares TruncateCore class, a layer core that truncates its input to a given size.
 *
 */

#ifndef TRUNCATECORE_H_
#define TRUNCATECORE_H_

#include "Core.h"

namespace netlib {

/**
 * @brief A layer core that truncates its input to a given size.
 *
 */
class TruncateCore: public Core {
public:
	TruncateCore(string name, cublasHandle_t& handle, int batchSize, int size, int seqLength);
	virtual ~TruncateCore();
	int getNParams();
	void forward(dtype2* prevAct, dtype2* act, dtype2* params, unsigned int* h_inputLengths,
			unsigned int* d_inputLengths, bool deriv = false, dtype2* priorAct = NULL, int s = 0);
	void calcGrad(dtype2* prevAct, dtype2 *act, dtype2* prevError, dtype2* error, dtype2* grad, dtype2* params,
			unsigned int* h_inputLengths, unsigned int* d_inputLengths, dtype2 *priorAct = NULL, dtype2* priorError = NULL,
			int s = 0);
};

} /* namespace netlib */

#endif /* TRUNCATECORE_H_ */
