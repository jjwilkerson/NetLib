/**
 * @file
 * @brief Declares the IterInfo class, which encapsulates and saves iteration information.
 *
 */

#ifndef ITERINFO_H_
#define ITERINFO_H_

#include "../NetLib.h"

namespace netlib {

/**
 * @brief Encapsulates and saves iteration information.
 *
 */
class IterInfo {
public:
	IterInfo(int iterNum);
	virtual ~IterInfo();
	void save(const char* filename);
	static void saveHeader(const char* filename);
	int iterNum;
	long clock;
	ix_type sentenceIx;
	dtypeh initialErr;
	dtypeh gradNorm;
	int cgLastStep;
	int usedCgStep;
	dtypeh backtrackedErr;
	dtypeh quad;
	dtypeh improvementRatio;
	dtypeh damping;
	dtypeh minImprov;
	dtypeh lRate;
	dtypeh lRateErr;
	dtypeh improvement;
};

} /* namespace netlib */

#endif /* ITERINFO_H_ */
