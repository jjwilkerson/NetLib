/**
 * @file
 * @brief Declares SplitLayer class, which receives an input and passes it to two separate downstream classes.
 *
 * Receives an input and passes it to two separate downstream classes. Uses SplitCore and BaseLayer.
 */

#ifndef SPLITLAYER_H_
#define SPLITLAYER_H_

#include "BaseLayer.h"

namespace netlib {

/**
 * @brief Receives an input and passes it to two separate downstream classes.
 *
 * Receives an input and passes it to two separate downstream classes. Uses SplitCore and BaseLayer.
 */
class SplitLayer: public BaseLayer {
public:
	SplitLayer(string name, cublasHandle_t& handle, int batchSize, int size, int seqLength, Nonlinearity* nonlinearity, float dropout);
	virtual ~SplitLayer();
	void forward(int batchNum, dtype2* params = NULL, bool deriv = false,
			bool stochasticDropout = true);
	void calcGrad(dtype2* grad, int batchNum);
	void setNext2(Layer* n);

	Layer *next2 = NULL;
private:
	bool twoBack = true;
};

} /* namespace netlib */

#endif /* SPLITLAYER_H_ */
