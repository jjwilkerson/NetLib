/**
 * @file
 * @brief Declares RecLayer class, a recurrent layer.
 *
 */

#ifndef RECLAYER_H_
#define RECLAYER_H_

#include "Layer.h"

namespace netlib {


/**
 * @brief A recurrent layer.
 *
 */
class RecLayer: public Layer {
public:
	RecLayer(string name, cublasHandle_t& handle, int batchSize, int size, int seqLength, Nonlinearity* nonlinearity, float dropout);
	virtual ~RecLayer();
	void forward(int batchNum, dtype2* params, bool deriv, bool stochasticDropout);
	void calcGrad(dtype2* grad, int batchNum);
	void addGradL2(dtype2* grad, dtypeh l2);
	void Rforward(dtype2* v, int batchNum);
	void Rback(dtype2* Gv, int batchNum);
	int getNParams();
	void setParamOffset(int offset);
	void initMem(bool training, bool optHF = true);
	void freeMem(bool training);
	void initWeights(WeightInit& ffWeightInit, WeightInit& recWeightInit);
	bool hasParam(int i);
	bool isWeight(int i);

private:
	unsigned long recParamOffset = 0;
	unsigned long recBiasOffset = 0;
	dtype2 *recShared1 = NULL;
	dtype2 *recShared2 = NULL;
};

} /* namespace netlib */

#endif /* RECLAYER_H_ */
