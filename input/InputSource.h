/**
 * @file
 * @brief Declares InputSource, a base class for sources of input for networks.
 *
 */

#ifndef INPUTSOURCE_H_
#define INPUTSOURCE_H_

#include "../NetLib.h"
#include <string>
#include <vector>

using namespace std;

namespace netlib {

/**
 * @brief A base class for sources of input for networks.
 *
 */
class InputSource {
public:
	virtual ~InputSource();
	virtual void toFirstBatch() = 0;
	virtual bool hasNextBatchSet() = 0;
	virtual void toNextBatchSet() = 0;
	virtual bool hasNext() = 0;
	virtual void nextBatch(int batchNum = 0, vector<string>* tokensRet = NULL) = 0;
	virtual dtype2** getRevInputs() = 0;
	virtual dtype2** getFwdInputs() = 0;
	virtual unsigned int** getDInputLengths() = 0;
	virtual unsigned int* getHInputLengths() = 0;
	virtual dtype2** getTargets() = 0;
	virtual void computeMatchMasks(int batchNum, dtype2** outputs) = 0;
	virtual dtype2** matchMasks(int batchNum) = 0;
	virtual void reset() = 0;
protected:
	InputSource();
};

} /* namespace netlib */

#endif /* INPUTSOURCE_H_ */
