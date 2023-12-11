/**
 * @file
 * @brief Declares classes that initializes weights (model parameters).
 *
 */

#ifndef WEIGHTINIT_H_
#define WEIGHTINIT_H_

#include "../NetLib.h"
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/variate_generator.hpp>

typedef boost::mt19937 base_generator_type;

namespace netlib {

/**
 * @brief Base class for classes that initializes weights (model parameters).
 *
 */
class WeightInit {
public:
	virtual ~WeightInit();
	virtual void initialize(dtype1* w, int preSize, int postSize) = 0;
};

/**
 * @brief Initializes weights (model parameters) with the sparse initialization scheme.
 *
 * Initializes weights (model parameters) with the sparse initialization scheme, which assigns non-zero values to 15
 * of the inputs to each neuron.
 */
class SparseWeightInit : public WeightInit {
public:
	SparseWeightInit(unsigned int seed, dtypeh coeff, dtypeh bias);
	virtual ~SparseWeightInit();
	void initialize(dtype1* w, int preSize, int postSize);
private:
	unsigned int seed;
	dtypeh coeff;
	dtypeh bias;
	base_generator_type* generator;
	boost::normal_distribution<>* normal_dist;
	boost::variate_generator<base_generator_type&, boost::normal_distribution<> >* normalRnd;
};

/**
 * @brief Initializes weights (model parameters) with the Gaussian initialization scheme.
 *
 * Initializes weights (model parameters) with the Gaussian initialization scheme, which assigns weight values from a
 * Gaussian distribution.
 */
class GaussianWeightInit : public WeightInit {
public:
	GaussianWeightInit(unsigned int seed, bool fillBias, dtypeh stddev, dtypeh mean=0.0f,
			dtypeh coeff=1.0f, dtypeh bias=0.0f);
	virtual ~GaussianWeightInit();
	void initialize(dtype1* w, int preSize, int postSize);
private:
	unsigned int seed;
	bool fillBias;
	dtypeh coeff;
	dtypeh bias;
	dtypeh mean;
	dtypeh stddev;
};

//class XavierWeightInit : public WeightInit {
//public:
//	XavierWeightInit(unsigned int seed);
//	virtual ~XavierWeightInit();
//	void initialize(dtype2* w, int preSize, int postSize);
//private:
//	unsigned int seed;
//};

} /* namespace netlib */

#endif /* WEIGHTINIT_H_ */
