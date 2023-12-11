/**
 * @file
 * @brief Declares BatchNormCore class, a batch normalization layer core.
 *
 * A network layer core that applies applies batch normalization. Used with BaseLayer.
 *
 */

#ifndef BATCHNORMCORE_H_
#define BATCHNORMCORE_H_

#include "Core.h"
#include <string>
#include <map>

using namespace std;

namespace netlib {

/**
 * @brief A batch normalization layer core.
 *
 * A network layer core that applies applies batch normalization. Used with BaseLayer.
 *
 */
class BatchNormCore: public Core {
public:
	BatchNormCore(string name, cublasHandle_t& handle, int batchSize, int size, int seqLength);
	virtual ~BatchNormCore();
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
	void saveState(ofstream& file);
	void loadState(ifstream& file);
	static void saveStates(int iterNum);
	static void saveStatesTemp();
	static void loadStates(string filename);

protected:
	unsigned long stdOffset = 0;
	dtype2* mean = NULL;
	dtype2* var = NULL;
	dtype2** xhats = NULL;
	dtype2** inv_vars = NULL;
	dtype2* intermed1 = NULL;
	dtype2* intermed2 = NULL;
	dtype2* single1 = NULL;
	dtype2* avg_mean = NULL;
	dtype2* avg_var = NULL;
	float epsilon;

private:
	int calcN(unsigned int* h_inputLengths, int s);
	static void addInstance(BatchNormCore* instance);
	static map<string, BatchNormCore*> instances;
	static void doSaveStates(const char*  filename);
};

} /* namespace netlib */

#endif /* BATCHNORMCORE_H_ */
