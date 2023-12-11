/**
 * @file
 * @brief Declares GruCore, a GRU layer core.
 *
 * A GRU (gated recurrent unit) layer core.
 */

#ifndef GRUCORE_H_
#define GRUCORE_H_

#include "Core.h"
#include <string>

namespace netlib {

/**
 * @brief A GRU layer core.
 *
 * A GRU (gated recurrent unit) layer core.
 */
class GruCore: public Core {
public:
	GruCore(string name, cublasHandle_t& handle, int batchSize, int size, int seqLength);
	virtual ~GruCore();
	int getNParams();
	void setParamOffset(int offset);
	void initWeights(dtype1* params, WeightInit& ffWeightInit, WeightInit& recWeightInit);
	void forward(dtype2* prevAct, dtype2* act, dtype2* params, unsigned int* h_inputLengths,
			unsigned int* d_inputLengths, bool deriv = false, dtype2* priorAct = NULL, int s = 0);
	void calcGrad(dtype2* prevAct, dtype2 *act, dtype2* prevError, dtype2* error, dtype2* grad, dtype2* params,
			unsigned int* h_inputLengths, unsigned int* d_inputLengths, dtype2 *priorAct = NULL, dtype2* priorError = NULL,
			int s = 0);
	void addGradL2(dtype2* params, dtype2* grad, dtypeh l2);
private:
	unsigned long wzxOffset = 0;
	unsigned long wzhOffset = 0;
	unsigned long bzOffset = 0;
	unsigned long wrxOffset = 0;
	unsigned long wrhOffset = 0;
	unsigned long brOffset = 0;
	unsigned long whxOffset = 0;
	unsigned long whhOffset = 0;
	unsigned long bhOffset = 0;
	unsigned long brecOffset = 0;
	dtype2** zs = NULL;
	dtype2** rs = NULL;
	dtype2** h_candidates = NULL;
	dtype2* dz = NULL;
	dtype2* dr = NULL;
	dtype2* dh_candidate = NULL;
	dtype2* intermed1 = NULL;
	dtype2* intermed2 = NULL;
	dtype2* intermed3 = NULL;
	dtype2* recBias = NULL;
	void initMem();
	void freeMem();
};

} /* namespace netlib */

#endif /* GRUCORE_H_ */
