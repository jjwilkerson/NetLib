/**
 * @file
 * @brief Declares PointerInputLayer class, an input layer that has its input directly set via a pointer.
 *
 */

#ifndef POINTERINPUTLAYER_H_
#define POINTERINPUTLAYER_H_

#include "InputLayer.h"

namespace netlib {

/**
 * @brief An input layer that has its input directly set via a pointer.
 *
 */
class PointerInputLayer: public InputLayer {
public:
	PointerInputLayer(string name, cublasHandle_t& handle, int batchSize, int size, int seqLength);
	virtual ~PointerInputLayer();
	void initMem(bool training, bool optHF = true);
	void freeMem(bool training);
	void setActivation(dtype2** act);
	void forward(int batchNum, dtype2* params, bool deriv, bool stochasticDropout);
	void Rforward(dtype2* v, int batchNum);
	int getNParams();
};

} /* namespace netlib */

#endif /* POINTERINPUTLAYER_H_ */
