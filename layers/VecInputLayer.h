/**
 * @file
 * @brief Declares VecInputLayer class, an input layer that handles singular (not sequential) data.
 *
 */

#ifndef VECINPUTLAYER_H_
#define VECINPUTLAYER_H_

#include "InputLayer.h"

namespace netlib {

/**
 * @brief An input layer that handles singular (not sequential) data.
 *
 */
class VecInputLayer: public InputLayer {
public:
	VecInputLayer(string name, cublasHandle_t& handle, int batchSize, int size, Nonlinearity* nonlinearity, float dropout);
	virtual ~VecInputLayer();
	void forward(int batchNum, dtype2* params, bool deriv, bool stochasticDropout);
	void Rforward(dtype2* v, int batchNum);
	int getNParams();
};

} /* namespace netlib */

#endif /* VECINPUTLAYER_H_ */
