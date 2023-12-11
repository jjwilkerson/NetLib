/**
 * @file
 * @brief Defines DenseLayer class, a densely-connected layer.
 *
 * A densely-connected layer. Uses DenseCore and BaseLayer.
 */

#include "DenseLayer2.h"

#include "cores/DenseCore.h"

namespace netlib {

DenseLayer2::DenseLayer2(string name, cublasHandle_t& handle, int batchSize, int size, int seqLength, Nonlinearity* nonlinearity, float dropout)
		: BaseLayer(new DenseCore(name + "_core", handle, batchSize, size, seqLength), name, handle, batchSize, size, seqLength, nonlinearity, dropout) {
}

DenseLayer2::~DenseLayer2() {
}

} /* namespace netlib */
