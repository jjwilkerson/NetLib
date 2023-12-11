/**
 * @file
 * @brief Declares Layer class, the base class for layers.
 *
 */

#ifndef LAYER_H_
#define LAYER_H_

#include "../NetLib.h"
//#include "Network.h"
//#include "Nonlinearity.h"
#include <string>
#include "cublas_v2.h"

using namespace std;

namespace netlib {

class Network;
class Nonlinearity;
class WeightInit;

/**
 * @brief The base class for layers.
 *
 */
class Layer {
public:
	virtual ~Layer();
	virtual int getNParams() = 0;
	virtual void setParamOffset(int offset) = 0;
	virtual void setPrev(Layer* p);
	virtual void initMem(bool training, bool optHF = true);
	virtual void freeMem(bool training);
	virtual void initWeights(WeightInit& ffWeightInit, WeightInit& recWeightInit);
	void asOutputLayer();
	bool isOutputLayer();
	virtual void setBnAfter(bool b);
	virtual void iterInit();
	virtual void forward(int batchNum, dtype2* params = NULL, bool deriv = false,
			bool stochasticDropout = true) = 0;
	virtual void calcGrad(dtype2* grad, int batchNum) = 0;
	virtual void addGradL2(dtype2* grad, dtypeh l2) = 0;
	virtual void Rforward(dtype2* v, int batchNum) = 0;
	virtual void Rback(dtype2* Gv, int batchNum) = 0;
	virtual void clearError();
	virtual void clearRact();
	virtual bool hasParam(int i);
	virtual bool isWeight(int i);
	void setStochasticOverride(bool value);
	string name;
	int size;
	int seqLength;
	int nParams = 0;
	int batchSize;
	Network *net = NULL;
	cublasHandle_t& handle;
	Layer *prev = NULL;
	Layer *next = NULL;
	dtype2 *W = NULL;

	float dropout;
	dtype2 **activation = NULL;
	dtype2 **d_activation = NULL;
	dtype2 **R_activation = NULL;
	dtype2 *doActivation = NULL;
	dtype2 *dropoutMask = NULL;
	dtype2 **error = NULL;
	dtype2 *singleShared1 = NULL;
	bool structDamp = false;

protected:
	Layer(string name, cublasHandle_t& handle, int batchSize, int size, int seqLength, Nonlinearity* nonlinearity, float dropout);
	void applyMask(dtype2* a, int size, int s, int batchNum);
	Nonlinearity* nonlinearity;
	unsigned long paramOffset = 0;
	unsigned long biasOffset = 0;
	dtype2 *singleShared2 = NULL;
	dtype2 *ffInput = NULL;
	bool bnAfter = false;
	bool stochasticOverrideSet = false;
	bool stochasticOverrideValue = false;

private:
	bool outputLayer = false;
};

} /* namespace netlib */

#endif /* LAYER_H_ */
