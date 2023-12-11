/**
 * @file
 * @brief Declares LayerNormLayer class, a layer normalization layer.
 *
 * A Layer normalization layer. Uses LayerNormCore and BaseLayer.
 */

#ifndef LAYERNORMLAYER_H_
#define LAYERNORMLAYER_H_

#include "BaseLayer.h"

namespace netlib {

class Nonlinearity;

/**
 * @brief A layer normalization layer.
 *
 * A Layer normalization layer. Uses LayerNormCore and BaseLayer.
 */
class LayerNormLayer: public BaseLayer {
public:
	LayerNormLayer(string name, cublasHandle_t& handle, int batchSize, int size,
			int seqLength, Nonlinearity* nonlinearity, float dropout, float stdInit);
	virtual ~LayerNormLayer();
	void setPrev(Layer* p);
};

} /* namespace netlib */

#endif /* LAYERNORMLAYER_H_ */
