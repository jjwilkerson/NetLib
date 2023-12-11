/**
 * @file
 * @brief Declares DupInLayer class, which converts sequential data to singular data.
 *
 * Converts sequential data to singular data by using the input lengths to pick the last vector in the sequence.
 */

#ifndef DUPINLAYER_H_
#define DUPINLAYER_H_

#include "Layer.h"

namespace netlib {

/**
 * @brief Declares DupInLayer class, which converts sequential data to singular data.
 *
 * Converts sequential data to singular data by using the input lengths to pick the last vector in the sequence.
 */
class DupInLayer: public Layer {
public:
	DupInLayer(string name, cublasHandle_t& handle, int batchSize, int size, Nonlinearity* nonlinearity, float dropout);
	virtual ~DupInLayer();
	void forward(int batchNum, dtype2* params, bool deriv, bool stochasticDropout);
	void calcGrad(dtype2* grad, int batchNum);
	void addGradL2(dtype2* grad, dtypeh l2);
	void Rforward(dtype2* v, int batchNum);
	void Rback(dtype2* Gv, int batchNum);
	int getNParams();
	void setParamOffset(int offset);
};

} /* namespace netlib */

#endif /* DUPINLAYER_H_ */
