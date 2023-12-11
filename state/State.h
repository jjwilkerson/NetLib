/**
 * @file
 * @brief Declares the State class, which encapsulates, saves, and loads program state.
 *
 */

#ifndef STATE_H_
#define STATE_H_

#include "../NetLib.h"
#include <string>

using namespace std;

namespace netlib {

/**
 * @brief Encapsulates, saves, and loads program state.
 *
 */
class State {
public:
	State();
	virtual ~State();
	void save(const char* filename);
	void load(const char* filename);
	State* randomize();

	int epoch;
	int iter;
	ix_type sentenceIx;
	ix_type docPos;

	string ixFilename;

	string weightsFilename;
	string curandStatesFilename;
	string replacerStateFilename;
	string questionFactoryStateFilename;
	string initDeltaFilename;

	dtypeh damping;
	dtypeh deltaDecay;
	dtypeh l2;

	int numBatchGrad;
	int numBatchG;
	int numBatchError;

	int maxIterCG;

	long clock;

	float learningRate;
	float lossScaleFac;
	int iterNoOverflow;
};

} /* namespace netlib */

#endif /* STATE_H_ */
