/**
 * @file
 * @brief Declares ActivationLayer class, an activation layer.
 *
 * A network layer core that simply applies a nonlinear activation to its input.
 */

#ifndef ACTIVATIONLAYER_H_
#define ACTIVATIONLAYER_H_

#include "Layer.h"

namespace netlib {

/**
 * @brief An activation layer.
 *
 * A network layer core that simply applies a nonlinear activation to its input.
 */
class ActivationLayer: public Layer {
public:
	ActivationLayer(string name, cublasHandle_t& handle, int batchSize, int size, int seqLength, Nonlinearity* nonlinearity);
	virtual ~ActivationLayer();
	int getNParams();
	void setParamOffset(int offset);
	void forward(int batchNum, dtype2* params = NULL, bool deriv = false, bool stochasticDropout = true);
	void calcGrad(dtype2* grad, int batchNum);
	void addGradL2(dtype2* grad, dtypeh l2);
	void Rforward(dtype2* v, int batchNum);
	void Rback(dtype2* Gv, int batchNum);
};

} /* namespace netlib */

#endif /* ACTIVATIONLAYER_H_ */
