/**
 * @file
 * @brief Defines Core class, the base class for layer cores.
 *
 */

#include "Core.h"
#include <iostream>

using namespace std;

namespace netlib {

Core::Core(string name, cublasHandle_t& handle, int batchSize, int size, int seqLength)
		: name(name), handle(handle), batchSize(batchSize), size(size), seqLength(seqLength) {
	preSize = 0;
}

Core::~Core() {
}

void Core::setParamOffset(int offset) {
	paramOffset = offset;
}

void Core::initMem(bool training, bool optHF) {
}

void Core::freeMem(bool training) {
}

void Core::initWeights(dtype1* params, WeightInit& ffWeightInit,
		WeightInit& recWeightInit) {
}

void Core::addGradL2(dtype2* params, dtype2* grad, dtypeh l2) {
}

void Core::setPreSize(int p) {
	preSize = p;
}

void Core::setBnAfter(bool b) {
	bnAfter = b;
}

} /* namespace netlib */

