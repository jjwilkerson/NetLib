/**
 * @file
 * @brief Defines InputLayer class, the base class for input layers.
 *
 */

#include "InputLayer.h"

#include "Layer.h"

namespace netlib {

InputLayer::InputLayer(string name, cublasHandle_t& handle, int batchSize, int size, int seqLength, Nonlinearity* nonlinearity, float dropout)
		: Layer(name, handle, batchSize, size, seqLength, nonlinearity, dropout) {
	// TODO Auto-generated constructor stub

}

InputLayer::~InputLayer() {
	// TODO Auto-generated destructor stub
}

int InputLayer::getNParams() {
	return 0;
}

void InputLayer::setParamOffset(int offset) {
}

void InputLayer::calcGrad(dtype2* grad, int batchNum) {
}

void InputLayer::addGradL2(dtype2* grad, dtypeh l2) {
}

void InputLayer::Rback(dtype2* Gv, int batchNum) {
}

} /* namespace netlib */
