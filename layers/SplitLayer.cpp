/**
 * @file
 * @brief Defines SplitLayer class, which receives an input and passes it to two separate downstream classes.
 *
 * Receives an input and passes it to two separate downstream classes. Uses SplitCore and BaseLayer.
 */

#include "SplitLayer.h"

#include "cores/SplitCore.h"

namespace netlib {

SplitLayer::SplitLayer(string name, cublasHandle_t& handle, int batchSize, int size, int seqLength, Nonlinearity* nonlinearity, float dropout)
						: BaseLayer(new SplitCore(name + "_core", handle, batchSize, size, seqLength), name, handle, batchSize, size, seqLength,
								nonlinearity, dropout) {
}

SplitLayer::~SplitLayer() {
}

void SplitLayer::forward(int batchNum, dtype2 *params, bool deriv,
		bool stochasticDropout) {
	BaseLayer::forward(batchNum, params, deriv, stochasticDropout);

	if (next2 != NULL) {
		next2->forward(batchNum, params, deriv, stochasticDropout);
	}
}

void SplitLayer::calcGrad(dtype2 *grad, int batchNum) {
	twoBack = !twoBack;
	if (!twoBack) {
		return;
	}

	BaseLayer::calcGrad(grad, batchNum);
}

void SplitLayer::setNext2(Layer *n) {
	next2 = n;
}

} /* namespace netlib */
