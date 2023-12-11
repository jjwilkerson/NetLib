/**
 * @file
 * @brief Declares ConcatLayer class, a concatenation layer.
 *
 * A network layer that applies a concatenation operation to its two inputs.
 *
 */

#ifndef CONCATLAYER_H_
#define CONCATLAYER_H_

#include "../NetLib.h"
#include "Layer.h"

namespace netlib {

/**
 * @brief A concatenation layer.
 *
 * A network layer that applies a concatenation operation to its two inputs.
 *
 */
class ConcatLayer: public Layer {
public:
	ConcatLayer(string name, cublasHandle_t& handle, int batchSize, int size, int seqLength,
			Nonlinearity* nonlinearity, float dropout);
	virtual ~ConcatLayer();
	void forward(int batchNum, dtype2* params, bool deriv, bool stochasticDropout);
	void calcGrad(dtype2* grad, int batchNum);
	void addGradL2(dtype2* grad, dtypeh l2);
	void Rforward(dtype2* v, int batchNum);
	void Rback(dtype2* Gv, int batchNum);
	int getNParams();
	void setParamOffset(int offset);
	void setPrev2(Layer* p);

private:
	bool twoForward = true;
	bool twoRforward = true;
	Layer *prev2 = NULL;
};

} /* namespace netlib */

#endif /* CONCATLAYER_H_ */
