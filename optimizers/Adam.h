/**
 * @file
 * @brief Declares Adam class, an ADAM optimizer.
 *
 * An optimizer that uses the ADAM algorithm.
 */

#ifndef ADAM_H_
#define ADAM_H_

#include "../NetLib.h"
#include "Optimizer.h"
#include <string>

namespace netlib {

class Network;
class IterInfo;

/**
 * @brief An ADAM optimizer.
 *
 * An optimizer that uses the ADAM algorithm.
 */
class Adam: public Optimizer {
public:
	Adam(cublasHandle_t& handle, Network& net, Config config, int startIter, bool debug = false, std::string initDeltaFilename = "");
	virtual ~Adam();
	void computeUpdate(IterInfo& iterInfo, bool printing);
	dtypeh getDamping();
	dtypeh getDeltaDecay();
	int getMaxIterCG();
	float getLearningRate();
	void saveInitDelta(const char* filename);
private:
	cublasHandle_t& handle;
	bool debug;
	dtypea* m1;
	dtypea* m2;
	int t;
	dtypeh lr;
	dtypeh neg_lr;
	dtypeh lrDecay;
	dtypeh gradClipThresh;
	dtypeh gradClipMax;
	dtypeh p1;
	dtypeh p2;
	dtypeh p1_inv;
	dtypeh p2_inv;
	void loadInitDelta(std::string filename);
};

} /* namespace netlib */

#endif /* ADAM_H_ */
