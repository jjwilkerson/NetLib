/**
 * @file
 * @brief Defines AddLayer class, an addition layer.
 *
 * A network layer that applies an addition operation to its two inputs. Uses AddCore and BaseLayer.
 *
 */

#include "AddLayer.h"

#include "cores/AddCore.h"
#include "../Network.h"

namespace netlib {

AddLayer::AddLayer(string name, cublasHandle_t& handle, int batchSize, int size, int seqLength, Nonlinearity* nonlinearity, float dropout)
				: BaseLayer(new AddCore(name + "_core", handle, batchSize, size, seqLength), name, handle, batchSize, size, seqLength,
						nonlinearity, dropout) {
}

AddLayer::~AddLayer() {
}

void AddLayer::forward(int batchNum, dtype2 *params, bool deriv,
		bool stochasticDropout) {
	twoForward = !twoForward;
	if (!twoForward) {
		return;
	}
	BaseLayer::forward(batchNum, params, deriv, stochasticDropout);
}

void AddLayer::calcGrad(dtype2 *grad, int batchNum) {
	BaseLayer::calcGrad(grad, batchNum);
//	Network::printStatsGpu(name + " error[0]", error[0], batchSize * size);
	prev2->calcGrad(grad, batchNum);
}

void AddLayer::setPrev2(Layer *p) {
	prev2 = p;
	((AddCore*) core)->setPrev2(p);
}

} /* namespace netlib */
