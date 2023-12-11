/**
 * @file
 * @brief Declares RecDTR2nLayer class, a residual-recurrent layer.
 *
 * A residual-recurrent layer. Has 1 or more residual modules.
 */

#ifndef RECDTR2NLAYER_H_
#define RECDTR2NLAYER_H_

#include "Layer.h"

namespace netlib {

/**
 * @brief A residual-recurrent layer.
 *
 * A residual-recurrent layer. Has 1 or more residual modules.
 */

class RecDTR2nLayer: public Layer {
public:
	RecDTR2nLayer(string name, cublasHandle_t& handle, int batchSize, int size, int seqLength, float dropout,
			Nonlinearity* recNonlinearity, Nonlinearity* transNonlinearity,
			float transA1Dropout, float transB11Dropout, float transB1Dropout, int numMod);
	virtual ~RecDTR2nLayer();
	void iterInit();
	void forward(int batchNum, dtype2* params, bool deriv, bool stochasticDropout);
	void calcGrad(dtype2* grad, int batchNum);
	void addGradL2(dtype2* grad, dtypeh l2);
	void Rforward(dtype2* v, int batchNum);
	void Rback(dtype2* Gv, int batchNum);
	void clearRact();
	int getNParams();
	void setParamOffset(int offset);
	void initMem(bool training, bool optHF = true);
	void freeMem(bool training);
	void initWeights(WeightInit& ffWeightInit, WeightInit& recWeightInit);
	bool hasParam(int i);
	bool isWeight(int i);

private:
	int numAdditional;
	unsigned long recParamOffset = 0;
	unsigned long recBiasOffset = 0;
	unsigned long transA1ParamOffset = 0;
	unsigned long transA1BiasOffset = 0;
	unsigned long transA2ParamOffset = 0;
	unsigned long transA2BiasOffset = 0;

	unsigned long* transB1ParamOffset = NULL;
	unsigned long* transB10ParamOffset = NULL;
	unsigned long* transB1BiasOffset = NULL;
	unsigned long* transB2ParamOffset = NULL;
	unsigned long* transB2BiasOffset = NULL;
	float transA1Dropout;
	float transB11Dropout;
	float transB1Dropout;
	dtype2 **recActivation = NULL;
	dtype2 **transA1Activation = NULL;
	dtype2 **transAActivation = NULL;
	dtype2 ***transB11Activation = NULL;
	dtype2 ***transB1Activation = NULL;
	dtype2 ***transBActivation = NULL;
	dtype2 **d_recActivation = NULL;
	dtype2 **d_transA1Activation = NULL;
	dtype2 ***d_transB11Activation = NULL;
	dtype2 ***d_transB1Activation = NULL;
	dtype2 *transA1DropoutMask = NULL;
	dtype2 **transB11DropoutMask = NULL;
	dtype2 **transB1DropoutMask = NULL;
	dtype2 *transDoActivation = NULL;
	dtype2 *singleShared3 = NULL;
	Nonlinearity* recNonlinearity;
	Nonlinearity* transNonlinearity;
};

} /* namespace netlib */

#endif /* RECDTR2NLAYER_H_ */
