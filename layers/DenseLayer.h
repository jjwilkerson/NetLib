/**
 * @file
 * @brief Declares DenseLayer class, a densely-connected layer.
 *
 *
 */

#ifndef DENSELAYER_H_
#define DENSELAYER_H_

#include "Layer.h"

namespace netlib {

class WeightInit;

/**
 * @brief A densely-connected layer.
 *
 */
class DenseLayer: public Layer {
public:
	DenseLayer(string name, cublasHandle_t& handle, int batchSize, int size, int seqLength, Nonlinearity* nonlinearity,
			float dropout, WeightInit* weightInit = NULL);
	virtual ~DenseLayer();
	void forward(int batchNum, dtype2* params, bool deriv, bool stochasticDropout);
	void calcGrad(dtype2* grad, int batchNum);
	void addGradL2(dtype2* grad, dtypeh l2);
	void Rforward(dtype2* v, int batchNum);
	void Rback(dtype2* Gv, int batchNum);
	int getNParams();
	void setParamOffset(int offset);
	void initWeights(WeightInit& ffWeightInit, WeightInit& recWeightInit);
private:
	WeightInit* weightInit;
};

} /* namespace netlib */

#endif /* DENSELAYER_H_ */
