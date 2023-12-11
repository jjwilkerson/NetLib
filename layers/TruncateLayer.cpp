/**
 * @file
 * @brief Defines TruncateLayer class, which truncates its input to a given size.
 *
 * Truncates its input to a given size. Uses TruncateCore and BaseLayer.
 */

#include "TruncateLayer.h"

#include "cores/TruncateCore.h"

namespace netlib {

TruncateLayer::TruncateLayer(string name, cublasHandle_t& handle, int batchSize, int size, int seqLength, Nonlinearity* nonlinearity, float dropout)
								: BaseLayer(new TruncateCore(name + "_core", handle, batchSize, size, seqLength), name, handle, batchSize, size, seqLength,
										nonlinearity, dropout) {
}

TruncateLayer::~TruncateLayer() {
}

} /* namespace netlib */
