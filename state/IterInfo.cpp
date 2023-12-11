/**
 * @file
 * @brief Defines the IterInfo class, which encapsulates and saves iteration information.
 *
 */

#include "IterInfo.h"
#include <fstream>
#include <iostream>
#include <iomanip>

using namespace std;

namespace netlib {

IterInfo::IterInfo(int iterNum) : iterNum(iterNum) {
	clock = 0l;
	sentenceIx = 0l;
	initialErr = 0.0f;
	gradNorm = 0.0f;
	cgLastStep = 0;
	usedCgStep = 0;
	backtrackedErr = 0.0f;
	quad = 0.0f;
	improvementRatio = 0.0f;
	damping = 0.0f;
	minImprov = 0.0f;
	lRate = 0.0f;
	lRateErr = 0.0f;
	improvement = 0.0f;
}

IterInfo::~IterInfo() {
}

void IterInfo::save(const char* filename) {
	std::ofstream file(filename, ios::app);

	file << fixed << setprecision(6);
	file << iterNum << '\t' << clock << '\t' << sentenceIx << '\t' << initialErr << '\t';
	file << gradNorm << '\t' << cgLastStep << '\t' << usedCgStep << '\t';
	file << backtrackedErr << '\t' << quad << '\t' << improvementRatio << '\t';
	file << defaultfloat;
	file << damping << '\t';
	file << fixed << setprecision(6);
	file << minImprov << '\t';
	file << defaultfloat << lRate << '\t';
	file << fixed << setprecision(6);
	file << lRateErr << '\t' << improvement << endl;

	file.close();
}

void IterInfo::saveHeader(const char* filename) {
	std::ofstream file(filename, ios::app);

	file << "iterNum" << '\t' << "clock" << '\t' << "sentenceIx" << '\t' << "initialErr" << '\t';
	file << "gradNorm" << '\t' << "cgLastStep" << '\t' << "usedCgStep" << '\t';
	file << "backtrackedErr" << '\t' << "quad" << '\t' << "improvementRatio" << '\t';
	file << "damping" << '\t' << "minImprov" << '\t' << "lRate" << '\t' << "lRateErr" << '\t';
	file << "improvement" << endl;

	file.close();
}

} /* namespace netlib */
