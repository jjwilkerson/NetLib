/**
 * @file
 * @brief Defines the Optimizer class, the base class for optimizers.
 *
 */

#include "Optimizer.h"
#include "../Network.h"
#include "../state/IterInfo.h"

namespace netlib {

Optimizer::Optimizer(Network& net) : net(net) {
}

Optimizer::~Optimizer() {
}

} /* namespace netlib */
