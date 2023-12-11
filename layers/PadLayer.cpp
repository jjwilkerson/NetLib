/**
 * @file
 * @brief Declares PadLayer class, which pads its input to a given size.
 *
 * Pads its input to a given size. Uses PadCore and BaseLayer.
 */

#include "PadLayer.h"

#include "cores/PadCore.h"

namespace netlib {

PadLayer::PadLayer(string name, cublasHandle_t& handle, int batchSize, int size, int seqLength, Nonlinearity* nonlinearity, float dropout)
								: BaseLayer(new PadCore(name + "_core", handle, batchSize, size, seqLength), name, handle, batchSize, size, seqLength,
										nonlinearity, dropout) {
}

PadLayer::~PadLayer() {
}

} /* namespace netlib */
