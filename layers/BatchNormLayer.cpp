/**
 * @file
 * @brief Defines BatchNormLayer class, a batch normalization layer.
 *
 * A network layer that applies applies batch normalization. Uses BatchNormCore and BaseLayer.
 *
 */

#include "BatchNormLayer.h"
#include "cores/BatchNormCore.h"

namespace netlib {

BatchNormLayer::BatchNormLayer(string name, cublasHandle_t& handle, int batchSize, int size,
		int seqLength, Nonlinearity* nonlinearity, float dropout)
	: BaseLayer(new BatchNormCore(name + "_core", handle, batchSize, size, seqLength), name, handle, batchSize,
			size, seqLength, nonlinearity, dropout) {
}

BatchNormLayer::~BatchNormLayer() {
}

void BatchNormLayer::setPrev(Layer* p) {
	BaseLayer::setPrev(p);
	p->setBnAfter(true);
}

} /* namespace netlib */

