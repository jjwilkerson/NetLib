/**
 * @file
 * @brief Declares DenseLayer class, a densely-connected layer core.
 *
 */

#ifndef DENSECORE_H_
#define DENSECORE_H_

#include "Core.h"

namespace netlib {

/**
 * @brief A densely-connected layer core.
 *
 */
class DenseCore: public Core {
public:
	DenseCore(string name, cublasHandle_t& handle, int batchSize, int size, int seqLength);
	virtual ~DenseCore();
	int getNParams();
	void setParamOffset(int offset);
	void initWeights(dtype1* params, WeightInit& ffWeightInit, WeightInit& recWeightInit);
	void forward(dtype2* prevAct, dtype2* act, dtype2* params, unsigned int* h_inputLengths,
			unsigned int* d_inputLengths, bool deriv = false, dtype2* priorAct = NULL, int s = 0);
	void calcGrad(dtype2* prevAct, dtype2 *act, dtype2* prevError, dtype2* error, dtype2* grad, dtype2* params,
			unsigned int* h_inputLengths, unsigned int* d_inputLengths, dtype2 *priorAct = NULL, dtype2* priorError = NULL,
			int s = 0);
	void addGradL2(dtype2* params, dtype2* grad, dtypeh l2);
protected:
	unsigned long biasOffset = 0;
};

} /* namespace netlib */

#endif /* DENSECORE_H_ */
