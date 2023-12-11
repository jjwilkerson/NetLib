/**
 * @file
 * @brief Defines LayerNormLayer class, a layer normalization layer.
 *
 * A Layer normalization layer. Uses LayerNormCore and BaseLayer.
 */

#include "LayerNormLayer.h"

#include "cores/LayerNormCore.h"

namespace netlib {

LayerNormLayer::LayerNormLayer(string name, cublasHandle_t& handle, int batchSize, int size,
		int seqLength, Nonlinearity* nonlinearity, float dropout, float stdInit)
	: BaseLayer(new LayerNormCore(name + "_core", handle, batchSize, size, seqLength, stdInit), name, handle, batchSize,
			size, seqLength, nonlinearity, dropout) {
}

LayerNormLayer::~LayerNormLayer() {
}

void LayerNormLayer::setPrev(Layer* p) {
	BaseLayer::setPrev(p);
	p->setBnAfter(true);
}

} /* namespace netlib */

