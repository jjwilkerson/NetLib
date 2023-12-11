/**
 * @file
 * @brief Declares classes that implement activation functions.
 *
 */

#ifndef NONLINEARITY_H_
#define NONLINEARITY_H_

#include "../NetLib.h"
#include "cublas_v2.h"

namespace netlib {

/**
 * @brief The base class for activation functions.
 *
 */
class Nonlinearity {
public:
	virtual ~Nonlinearity();
	virtual void activation(int size, dtype2* preAct, dtype2* act) = 0;
	virtual void d_activation(int size, dtype2* preAct, dtype2* act, dtype2* d_act) = 0;
	virtual bool isLinear() = 0;
protected:
	Nonlinearity(int batchSize);
	int batchSize;
};

/**
 * @brief Implements the linear (identity) activation function.
 *
 */
class Linear : public Nonlinearity {
public:
	Linear(int batchSize);
	virtual ~Linear();
	void activation(int size, dtype2* preAct, dtype2* act);
	void d_activation(int size, dtype2* preAct, dtype2* act, dtype2* d_act);
	bool isLinear();
};

/**
 * @brief Implements the tanh activation function.
 *
 */
class Tanh : public Nonlinearity {
public:
	Tanh(int batchSize);
	virtual ~Tanh();
	void activation(int size, dtype2* preAct, dtype2* act);
	void d_activation(int size, dtype2* preAct, dtype2* act, dtype2* d_act);
	bool isLinear();
};

/**
 * @brief Implements the leaky ReLU activation function.
 *
 */
class LeakyReLU : public Nonlinearity {
public:
	LeakyReLU(int batchSize);
	virtual ~LeakyReLU();
	void activation(int size, dtype2* preAct, dtype2* act);
	void d_activation(int size, dtype2* preAct, dtype2* act, dtype2* d_act);
	bool isLinear();
};

/**
 * @brief Implements the softmax activation function.
 *
 */

class Softmax : public Nonlinearity {
public:
	Softmax(cublasHandle_t& handle, int batchSize);
	virtual ~Softmax();
	void activation(int size, dtype2* preAct, dtype2* act);
	void d_activation(int size, dtype2* preAct, dtype2* act, dtype2* d_act);
	bool isLinear();
private:
	cublasHandle_t& handle;
	dtype2* tempVec = NULL;
};

/**
 * @brief Implements the bi-ReLU activation function.
 *
 */
class BiReLU : public Nonlinearity {
public:
	BiReLU(int batchSize, dtype2 x1, dtype2 x2);
	virtual ~BiReLU();
	void activation(int size, dtype2* preAct, dtype2* act);
	void d_activation(int size, dtype2* preAct, dtype2* act, dtype2* d_act);
	bool isLinear();
private:
	dtype2 x1;
	dtype2 x2;
};

/**
 * @brief Implements the leaky bi-ReLU activation function.
 *
 */
class LeakyBiReLU : public Nonlinearity {
public:
	LeakyBiReLU(int batchSize, dtype2 x1, dtype2 x2, dtype2 a);
	virtual ~LeakyBiReLU();
	void activation(int size, dtype2* preAct, dtype2* act);
	void d_activation(int size, dtype2* preAct, dtype2* act, dtype2* d_act);
	bool isLinear();
private:
	dtype2 x1;
	dtype2 x2;
	dtype2 a;
};

} /* namespace netlib */

#endif /* NONLINEARITY_H_ */
