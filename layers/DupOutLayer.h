/**
 * @file
 * @brief Declares DupOutLayer class, which converts singular data to sequential data.
 *
 * Converts singular data to sequential data by repeating its input data. The way it works is that the following layer will
 * always take the same data, regards of position in sequence.
 */
#ifndef DUPOUTLAYER_H_
#define DUPOUTLAYER_H_

#include "Layer.h"

namespace netlib {

/**
 * @brief Declares DupOutLayer class, which converts singular data to sequential data.
 *
 * Converts singular data to sequential data by repeating its input data. The way it works is that the following layer will
 * always take the same data, regards of position in sequence.
 */
class DupOutLayer: public Layer {
public:
	DupOutLayer(string name, cublasHandle_t& handle, int batchSize, int size, Nonlinearity* nonlinearity, float dropout);
	virtual ~DupOutLayer();
	void forward(int batchNum, dtype2* params, bool deriv, bool stochasticDropout);
	void calcGrad(dtype2* grad, int batchNum);
	void addGradL2(dtype2* grad, dtypeh l2);
	void Rforward(dtype2* v, int batchNum);
	void Rback(dtype2* Gv, int batchNum);
	int getNParams();
	void setParamOffset(int offset);
};

} /* namespace netlib */

#endif /* DUPOUTLAYER_H_ */
