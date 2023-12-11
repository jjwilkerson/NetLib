/**
 * @file
 * @brief Declares DenseLayer class, a densely-connected layer.
 *
 * A densely-connected layer. Uses DenseCore and BaseLayer.
 */

#ifndef DENSELAYER2_H_
#define DENSELAYER2_H_

#include "BaseLayer.h"

namespace netlib {

/**
 * @brief A densely-connected layer.
 *
 * A densely-connected layer. Uses DenseCore and BaseLayer.
 */
class DenseLayer2: public BaseLayer {
public:
	DenseLayer2(string name, cublasHandle_t& handle, int batchSize, int size, int seqLength, Nonlinearity* nonlinearity, float dropout);
	virtual ~DenseLayer2();
};

} /* namespace netlib */

#endif /* DENSELAYER2_H_ */
