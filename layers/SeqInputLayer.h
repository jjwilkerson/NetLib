/**
 * @file
 * @brief Declares SeqInputLayer class, an input layer that handles sequential data.
 *
 */

#ifndef SEQINPUTLAYER_H_
#define SEQINPUTLAYER_H_

#include "../NetLib.h"
#include "cublas_v2.h"
#include "InputLayer.h"

namespace netlib {

//class InputLayer;
class Nonlinearity;

/**
 * @brief An input layer that handles sequential data.
 *
 */
class SeqInputLayer: public InputLayer {
public:
	SeqInputLayer(string name, cublasHandle_t& handle, int batchSize, int size, int seqLength, Nonlinearity* nonlinearity, float dropout,
			bool dirFwd);
	virtual ~SeqInputLayer();
	void forward(int batchNum, dtype2* params, bool deriv, bool stochasticDropout);
	void Rforward(dtype2* v, int batchNum);
	int getNParams();

private:
	bool dirFwd;
};

} /* namespace netlib */

#endif /* SEQINPUTLAYER_H_ */
