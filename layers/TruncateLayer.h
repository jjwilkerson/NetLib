/**
 * @file
 * @brief Declares TruncateLayer class, which truncates its input to a given size.
 *
 * Truncates its input to a given size. Uses TruncateCore and BaseLayer.
 */

#ifndef TRUNCATELAYER_H_
#define TRUNCATELAYER_H_

#include "BaseLayer.h"

namespace netlib {

/**
 * @brief Truncates its input to a given size.
 *
 * Truncates its input to a given size. Uses TruncateCore and BaseLayer.
 */
class TruncateLayer: public BaseLayer {
public:
	TruncateLayer(string name, cublasHandle_t& handle, int batchSize, int size, int seqLength, Nonlinearity* nonlinearity, float dropout);
	virtual ~TruncateLayer();
};

} /* namespace netlib */

#endif /* TRUNCATELAYER_H_ */
