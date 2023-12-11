/**
 * @file
 * @brief Declares Core class, the base class for layer cores.
 *
 */

#ifndef CORE_H_
#define CORE_H_

#include "../../NetLib.h"
#include "cublas_v2.h"
#include <string>

using namespace std;

namespace netlib {

class WeightInit;

/**
 * @brief The base class for layer cores.
 *
 */
class Core {
public:
	virtual ~Core();
	virtual int getNParams() = 0;
	virtual void setParamOffset(int offset);
	virtual void initMem(bool training, bool optHF = true);
	virtual void freeMem(bool training);
	virtual void initWeights(dtype1* params, WeightInit& ffWeightInit, WeightInit& recWeightInit);
	virtual void forward(dtype2* prevAct, dtype2* act, dtype2* params, unsigned int* h_inputLengths,
			unsigned int* d_inputLengths, bool deriv = false, dtype2* priorAct = NULL, int s = 0) = 0;
	virtual void calcGrad(dtype2* prevAct, dtype2 *act, dtype2* prevError, dtype2* error, dtype2* grad, dtype2* params,
			unsigned int* h_inputLengths, unsigned int* d_inputLengths, dtype2 *priorAct = NULL, dtype2* priorError = NULL,
			int s = 0) = 0;
	virtual void addGradL2(dtype2* params, dtype2* grad, dtypeh l2);
	void setPreSize(int p);
	void setBnAfter(bool b);
protected:
	Core(string name, cublasHandle_t& handle, int batchSize, int size, int seqLength);
	cublasHandle_t& handle;
	int batchSize;
	int size;
	int seqLength;
	int preSize;
	unsigned long paramOffset = 0;
	bool bnAfter = false;
	string name;
};

} /* namespace netlib */

#endif /* CORE_H_ */
