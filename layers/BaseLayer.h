/**
 * @file
 * @brief Declares BaseLayer class, a base layer.
 *
 * A base network layer that is meant to be used with a subclass of Core. Implements functionality common to all layers, while
 * the Core subclass implements the specific operations of a layer.
 *
 */

#ifndef BASELAYER_H_
#define BASELAYER_H_

#include "Layer.h"

namespace netlib {

class Core;

/**
 * @brief A base layer.
 *
 * A base network layer that is meant to be used with a subclass of Core. Implements functionality common to all layers, while
 * the Core subclass implements the specific operations of a layer.
 *
 */
class BaseLayer: public Layer {
public:
	virtual ~BaseLayer();
	int getNParams();
	void setParamOffset(int offset);
	void initMem(bool training, bool optHF = true);
	void freeMem(bool training);
	void initWeights(WeightInit& ffWeightInit, WeightInit& recWeightInit);
	void forward(int batchNum, dtype2* params = NULL, bool deriv = false,
			bool stochasticDropout = true);
	void calcGrad(dtype2* grad, int batchNum);
	void addGradL2(dtype2* grad, dtypeh l2);
	void Rforward(dtype2* v, int batchNum);
	void Rback(dtype2* Gv, int batchNum);
	void setPrev(Layer* p);
	void setBnAfter(bool b);
protected:
	BaseLayer(Core* core, string name, cublasHandle_t& handle, int batchSize,
			int size, int seqLength, Nonlinearity* nonlinearity, float dropout);
	Core* core;
};

} /* namespace netlib */

#endif /* BASELAYER_H_ */
