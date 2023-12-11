/**
 * @file
 * @brief Declares ActivationLayer2 class, an activation layer.
 *
 * A network layer that simply applies a nonlinear activation to its input. Uses ActivationCore and BaseLayer.
 */

#ifndef ACTIVATIONLAYER2_H_
#define ACTIVATIONLAYER2_H_

#include "BaseLayer.h"

namespace netlib {

/**
 * @brief An activation layer.
 *
 * A network layer that simply applies a nonlinear activation to its input. Uses ActivationCore and BaseLayer.
 */
class ActivationLayer2: public BaseLayer {
public:
	ActivationLayer2(string name, cublasHandle_t& handle, int batchSize, int size, int seqLength, Nonlinearity* nonlinearity);
	virtual ~ActivationLayer2();
};

} /* namespace netlib */

#endif /* ACTIVATIONLAYER2_H_ */
