/**
 * @file
 * @brief Defines ActivationLayer2 class, an activation layer.
 *
 * A network layer that simply applies a nonlinear activation to its input. Uses ActivationCore and BaseLayer.
 */

#include "ActivationLayer2.h"
#include "cores/ActivationCore.h"

namespace netlib {

ActivationLayer2::ActivationLayer2(string name, cublasHandle_t& handle, int batchSize, int size, int seqLength, Nonlinearity* nonlinearity)
		: BaseLayer(new ActivationCore(name + "_core", handle, batchSize, size, seqLength), name, handle, batchSize, size, seqLength, nonlinearity, 0.0) {
}

ActivationLayer2::~ActivationLayer2() {
}

} /* namespace netlib */
