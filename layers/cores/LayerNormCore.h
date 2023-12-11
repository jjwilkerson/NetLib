/**
 * @file
 * @brief Declares LayerNormCore class, a layer normalization layer core.
 *
 */

#ifndef LAYERNORMCORE_H_
#define LAYERNORMCORE_H_

#include "Core.h"
#include <string>
#include <map>

using namespace std;

namespace netlib {

/**
 * @brief A layer normalization layer core.
 *
 */
class LayerNormCore: public Core {
public:
	LayerNormCore(string name, cublasHandle_t& handle, int batchSize, int size, int seqLength,
			float stdInit);
	virtual ~LayerNormCore();
	int getNParams();
	void setParamOffset(int offset);
	void initMem(bool training, bool optHF = true);
	void freeMem(bool training);
	void initWeights(dtype1* params, WeightInit& ffWeightInit, WeightInit& recWeightInit);
	void forward(dtype2* prevAct, dtype2* act, dtype2* params, unsigned int* h_inputLengths,
			unsigned int* d_inputLengths, bool deriv = false, dtype2* priorAct = NULL, int s = 0);
	void calcGrad(dtype2* prevAct, dtype2 *act, dtype2* prevError, dtype2* error, dtype2* grad, dtype2* params,
			unsigned int* h_inputLengths, unsigned int* d_inputLengths, dtype2 *priorAct = NULL, dtype2* priorError = NULL,
			int s = 0);

protected:
	float stdInit;
	unsigned long stdOffset = 0;
	dtype2* mean = NULL;
	dtype2* var = NULL;
	dtype2** xhats = NULL;
	dtype2** inv_vars = NULL;
	dtype2* intermed1 = NULL;
	dtype2* intermed2 = NULL;
	dtype2* singleCol1 = NULL;
	float epsilon;
};

} /* namespace netlib */

#endif /* LAYERNORMCORE_H_ */
