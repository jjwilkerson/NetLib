/**
 * @file
 * @brief Declares GruLayer, a GRU layer.
 *
 * A GRU (gated recurrent unit) layer. Uses GruCore and BaseLayer.
 */

#ifndef GRULAYER_H_
#define GRULAYER_H_

#include "BaseLayer.h"

namespace netlib {

/**
 * @brief Declares GruLayer, a GRU layer.
 *
 * A GRU (gated recurrent unit) layer. Uses GruCore and BaseLayer.
 */
class GruLayer: public BaseLayer {
public:
	GruLayer(string name, cublasHandle_t& handle, int batchSize, int size, int seqLength, Nonlinearity* nonlinearity, float dropout);
	virtual ~GruLayer();
};

} /* namespace netlib */

#endif /* GRULAYER_H_ */
