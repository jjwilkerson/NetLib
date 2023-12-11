/**
 * @file
 * @brief Declares classes that implement loss functions.
 *
 */

#ifndef LOSSFUNCTION_H_
#define LOSSFUNCTION_H_

#include "../NetLib.h"
//#include <cuda_runtime.h>
#include "cublas_v2.h"

namespace netlib {

class Network;
class Layer;

/**
 * @brief
 *
 * Base class for loss functions.
 *
 */
class LossFunction {
public:
	virtual ~LossFunction();
	virtual dtypeh loss(dtype2** outputs, dtype2** targets, int s, dtype2** masks = NULL,
			unsigned int* d_inputLengths = NULL) = 0;
	virtual void d_loss(dtype2** targets, Layer *layer, int s,
			dtype2* out, bool increment, dtype2** masks = NULL,
			unsigned int* d_inputLengths = NULL) = 0;
	virtual void d2_loss(dtype2** targets, Layer *layer, int s,
			dtype2* out, bool increment, dtype2** masks = NULL) = 0;
	dtypeh batchLoss(dtype2** outputs, dtype2** targets, dtype2** losses_d, bool average = true,
			dtype2** masks = NULL, unsigned int* d_inputLengths = NULL);
	virtual void setNetwork(Network *n);
	virtual bool derivCombo();

protected:
	LossFunction(int batchSize, int maxSeqLength);
	Network *net = NULL;
	int batchSize;
	int maxSeqLength;
};

/**
 * @brief Implements the squared error loss function.
 *
 */
class SquaredError : public LossFunction {
public:
	SquaredError(int batchSize, int maxSeqLength, cublasHandle_t& handle);
	virtual ~SquaredError();
	dtypeh loss(dtype2** outputs, dtype2** targets, int s, dtype2** masks = NULL,
			unsigned int* d_inputLengths = NULL);
	void d_loss(dtype2** targets, Layer *layer, int s,
			dtype2* out, bool increment, dtype2** masks = NULL,
			unsigned int* d_inputLengths = NULL);
	void d2_loss(dtype2** targets, Layer *layer, int s,
			dtype2* out, bool increment, dtype2** masks = NULL);
	void setNetwork(Network *n);

private:
	cublasHandle_t& handle;
	int outputLength = 0;
	dtype2* intermed = NULL;
	dtype2* d_sum = NULL;
};

/**
 * @brief Implements structural damping as a loss function.
 *
 */
class StructuralDamping : public LossFunction {
public:
	StructuralDamping(int batchSize, int maxSeqLength, dtype2 weight);
	virtual ~StructuralDamping();
	dtypeh loss(dtype2** outputs, dtype2** targets, int s, dtype2** masks = NULL,
			unsigned int* d_inputLengths = NULL);
	void d_loss(dtype2** targets, Layer *layer, int s,
			dtype2* out, bool increment, dtype2** masks = NULL,
			unsigned int* d_inputLengths = NULL);
	void d2_loss(dtype2** targets, Layer *layer, int s,
			dtype2* out, bool increment, dtype2** masks = NULL);
	dtype2 damping;
private:
	dtype2 weight;
};

/**
 * @brief Implements the cross-entropy loss function.
 *
 */
class CrossEntropy : public LossFunction {
public:
	CrossEntropy(int batchSize, int maxSeqLength, cublasHandle_t& handle);
	virtual ~CrossEntropy();
	dtypeh loss(dtype2** outputs, dtype2** targets, int s, dtype2** masks = NULL,
			unsigned int* d_inputLengths = NULL);
	void d_loss(dtype2** targets, Layer *layer, int s,
			dtype2* out, bool increment, dtype2** masks = NULL,
			unsigned int* d_inputLengths = NULL);
	void d2_loss(dtype2** targets, Layer *layer, int s,
			dtype2* out, bool increment, dtype2** masks = NULL);
	void setNetwork(Network *n);
	bool derivCombo();

private:
	cublasHandle_t& handle;
	int outputLength = 0;
	dtype2* intermedVec1 = NULL;
	dtype2* intermedMat1 = NULL;
	dtype2* d_sum = NULL;
};

/**
 * @brief Implements the cosine sim loss function.
 *
 */
class CosineSim : public LossFunction {
public:
	CosineSim(int batchSize, int maxSeqLength, cublasHandle_t& handle);
	virtual ~CosineSim();
	dtypeh loss(dtype2** outputs, dtype2** targets, int s, dtype2** masks = NULL,
			unsigned int* d_inputLengths = NULL);
	void d_loss(dtype2** targets, Layer *layer, int s,
			dtype2* out, bool increment, dtype2** masks = NULL,
			unsigned int* d_inputLengths = NULL);
	void d2_loss(dtype2** targets, Layer *layer, int s,
			dtype2* out, bool increment, dtype2** masks = NULL);
	void setNetwork(Network *n);

private:
	void calcBatchErrors(dtype2* output, dtype2* target);
	cublasHandle_t& handle;
	int outputLength = 0;
	dtype2* intermedMat1 = NULL;
	dtype2* intermedMat2 = NULL;
	dtype2* intermedVec1 = NULL;
	dtype2* intermedVec2 = NULL;
	dtype2* intermedVec3 = NULL;
};

/**
 * @brief Wrapper for a set of loss functions.
 *
 */
class LossSet : public LossFunction {
public:
	LossSet(int batchSize, int maxSeqLength, int numLoss, LossFunction** lossFunctions);
	virtual ~LossSet();
	dtypeh loss(dtype2** outputs, dtype2** targets, int s, dtype2** masks = NULL,
			unsigned int* d_inputLengths = NULL);
	void d_loss(dtype2** targets, Layer *layer, int s,
			dtype2* out, bool increment, dtype2** masks = NULL,
			unsigned int* d_inputLengths = NULL);
	void d2_loss(dtype2** targets, Layer *layer, int s,
			dtype2* out, bool increment, dtype2** masks = NULL);
	void setNetwork(Network *n);

private:
	int numLoss;
	LossFunction** lossFunctions;
};

} /* namespace netlib */

#endif /* LOSSFUNCTION_H_ */
