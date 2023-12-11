/**
 * @file
 * @brief Declares AddLayer class, an addition layer.
 *
 * A network layer that applies an addition operation to its two inputs. Uses AddCore and BaseLayer.
 *
 */

#ifndef ADDLAYER_H_
#define ADDLAYER_H_

#include "BaseLayer.h"

namespace netlib {

/**
 * @brief An addition layer.
 *
 * A network layer that applies an addition operation to its two inputs. Uses AddCore and BaseLayer.
 *
 */
class AddLayer: public BaseLayer {
public:
	AddLayer(string name, cublasHandle_t& handle, int batchSize, int size, int seqLength, Nonlinearity* nonlinearity, float dropout);
	virtual ~AddLayer();
	void forward(int batchNum, dtype2* params, bool deriv, bool stochasticDropout);
	void calcGrad(dtype2* grad, int batchNum);
	void setPrev2(Layer* p);

	Layer *prev2 = NULL;
private:
	bool twoForward = true;
};

} /* namespace netlib */

#endif /* ADDLAYER_H_ */
