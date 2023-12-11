/**
 * @file
 * @brief Declares BatchNormLayer class, a batch normalization layer.
 *
 * A network layer that applies applies batch normalization. Uses BatchNormCore and BaseLayer.
 *
 */

#ifndef BATCHNORMLAYER_H_
#define BATCHNORMLAYER_H_

#include "BaseLayer.h"

namespace netlib {

class Nonlinearity;

/**
 * @brief A batch normalization layer.
 *
 * A network layer that applies applies batch normalization. Uses BatchNormCore and BaseLayer.
 *
 */
class BatchNormLayer: public BaseLayer {
public:
	BatchNormLayer(string name, cublasHandle_t& handle, int batchSize, int size,
			int seqLength, Nonlinearity* nonlinearity, float dropout);
	virtual ~BatchNormLayer();
	void setPrev(Layer* p);
};

} /* namespace netlib */

#endif /* BATCHNORMLAYER_H_ */
