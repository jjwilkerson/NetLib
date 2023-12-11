/**
 * @file
 * @brief Declares InputLayer class, the base class for input layers.
 *
 */

#ifndef INPUTLAYER_H_
#define INPUTLAYER_H_

#include "Layer.h"
#include "cublas_v2.h"

namespace netlib {

//class Layer;
class Nonlinearity;

/**
 * @brief The base class for input layers.
 *
 */
class InputLayer: public Layer {
public:
	virtual ~InputLayer();
//	void forward(int batchNum, dtype2* params = NULL, bool deriv = false,
//			bool stochasticDropout = true);
	int getNParams();
	void setParamOffset(int offset);
	void calcGrad(dtype2* grad, int batchNum);
	void addGradL2(dtype2* grad, dtypeh l2);
	void Rback(dtype2* Gv, int batchNum);

protected:
	InputLayer(string name, cublasHandle_t& handle, int batchSize, int size, int seqLength, Nonlinearity* nonlinearity, float dropout);
};

} /* namespace netlib */

#endif /* INPUTLAYER_H_ */
