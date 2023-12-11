/**
 * @file
 * @brief Defines GruLayer, a GRU layer.
 *
 * A GRU (gated recurrent unit) layer. Uses GruCore and BaseLayer.
 */

#include "GruLayer.h"
#include "cores/GruCore.h"

namespace netlib {

GruLayer::GruLayer(string name, cublasHandle_t& handle, int batchSize, int size, int seqLength, Nonlinearity* nonlinearity, float dropout)
		: BaseLayer(new GruCore(name + "_core", handle, batchSize, size, seqLength), name, handle, batchSize, size, seqLength, nonlinearity, dropout){

}

GruLayer::~GruLayer() {
}

} /* namespace netlib */
