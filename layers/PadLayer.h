/**
 * @file
 * @brief Declares PadLayer class, which pads its input to a given size.
 *
 * Pads its input to a given size. Uses PadCore and BaseLayer.
 */

#ifndef PADLAYER_H_
#define PADLAYER_H_

#include "BaseLayer.h"

namespace netlib {

/**
 * @brief Pads its input to a given size.
 *
 * Pads its input to a given size. Uses PadCore and BaseLayer.
 */
class PadLayer: public BaseLayer {
public:
	PadLayer(string name, cublasHandle_t& handle, int batchSize, int size, int seqLength, Nonlinearity* nonlinearity, float dropout);
	virtual ~PadLayer();
};

} /* namespace netlib */

#endif /* PADLAYER_H_ */
