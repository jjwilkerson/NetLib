/**
 * @file
 * @brief Declares the Optimizer class, the base class for optimizers.
 *
 */

#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_

#include "../NetLib.h"
#include "../Network.h"

namespace netlib {

class Network;
class IterInfo;

/**
 * @brief The base class for optimizers.
 *
 */
class Optimizer {
public:
	virtual ~Optimizer();
	virtual void computeUpdate(IterInfo& iterInfo, bool printing) = 0;
	virtual dtypeh getDamping() = 0;
	virtual dtypeh getDeltaDecay() = 0;
	virtual int getMaxIterCG() = 0;
	virtual float getLearningRate() = 0;
	virtual void saveInitDelta(const char* filename) = 0;
	Network& net;
protected:
	Optimizer(Network& net);
};

} /* namespace netlib */

#endif /* OPTIMIZER_H_ */
