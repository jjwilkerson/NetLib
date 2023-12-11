/**
 * @file
 * @brief Defines classes that initializes weights (model parameters).
 *
 */

#include "WeightInit.h"

#include "../gpu/CudaFunc.h"
#include <set>
#include <cuda_runtime.h>
#include <helper_cuda.h>

using namespace std;

namespace netlib {

WeightInit::~WeightInit() {
}

SparseWeightInit::SparseWeightInit(unsigned int seed, dtypeh coeff, dtypeh bias)
		: seed(seed), coeff(coeff), bias(bias) {
	generator = new base_generator_type(seed);
	normal_dist = new boost::normal_distribution<>();
	normalRnd = new boost::variate_generator<base_generator_type&, boost::normal_distribution<>>(*generator, *normal_dist);
}

SparseWeightInit::~SparseWeightInit() {
	delete generator;
	delete normal_dist;
	delete normalRnd;
}

void SparseWeightInit::initialize(dtype1* w, int preSize, int postSize) {
	boost::uniform_int<> uni_dist(0, preSize-1);
	boost::variate_generator<base_generator_type&, boost::uniform_int<>> uniRnd(*generator, uni_dist);

	int numRows = preSize;
	int arraySize = numRows * sizeof(dtype1);

	dtype1* hostColumn;
    hostColumn = (dtype1*) malloc(arraySize);

	static const int maxNumConn = 15;
	int numConn = min(preSize, maxNumConn);

	set<int> picked;

	for (int j = 0; j < postSize; j++) {
		memset((void *)hostColumn, 0, arraySize);
		picked.clear();
		for (int p = 0; p < numConn; p++) {
			int preIx;
			do {
				preIx = (int) uniRnd();
			} while (picked.find(preIx) != picked.end());
			picked.insert(preIx);

			dtype1 val = h21((*normalRnd)() * coeff);
			hostColumn[preIx] = val;
		}
	    checkCudaErrors(cudaMemcpy(w+IDX2(0,j,numRows), hostColumn, arraySize, cudaMemcpyHostToDevice));
	}

    free(hostColumn);
}

GaussianWeightInit::GaussianWeightInit(unsigned int seed, bool fillBias, dtypeh stddev, dtypeh mean,
		dtypeh coeff, dtypeh bias)
	: seed(seed), fillBias(fillBias), coeff(coeff), bias(bias), mean(mean), stddev(stddev) {
}

GaussianWeightInit::~GaussianWeightInit() {
}

void GaussianWeightInit::initialize(dtype1* w, int preSize, int postSize) {
	static base_generator_type generator(seed);
	static boost::normal_distribution<> normal_dist(mean, stddev);
	static boost::variate_generator<base_generator_type&, boost::normal_distribution<> > normalRnd(generator, normal_dist);

	int numWeights = preSize * postSize;
	int arraySize = numWeights * sizeof(dtype1);

	dtype1* hostArray;
    hostArray = (dtype1*) malloc(arraySize);

    for (int i = 0; i < numWeights; i++) {
    	hostArray[i] = float21(normalRnd() * coeff);
    }
    checkCudaErrors(cudaMemcpy(w, hostArray, arraySize, cudaMemcpyHostToDevice));
    free(hostArray);

    if (fillBias) {
    	dtype1* b = w + numWeights;
    	CudaFunc::fill1(b, postSize, bias);
    }
}

//XavierWeightInit::XavierWeightInit(unsigned int seed)
//	: seed(seed) {
//}
//
//XavierWeightInit::~XavierWeightInit() {
//}
//
//void XavierWeightInit::initialize(dtype1* w, int preSize, int postSize) {
//	static base_generator_type generator(seed);
//	static boost::normal_distribution<> normal_dist(mean, stddev);
//	static boost::variate_generator<base_generator_type&, boost::normal_distribution<> > normalRnd(generator, normal_dist);
//
//	int numWeights = preSize * postSize;
//	int arraySize = numWeights * sizeof(dtype2);
//
//	dtype2 c = sqrt(1.0f/preSize);
//
//	dtype2* hostArray;
//    hostArray = (dtype2*) malloc(arraySize);
//
//    for (int i = 0; i < numWeights; i++) {
//    	hostArray[i] = (dtype2) normalRnd() * c;
//    }
//    checkCudaErrors(cudaMemcpy(w, hostArray, arraySize, cudaMemcpyHostToDevice));
//    free(hostArray);
//
//    dtype2* b = w + numWeights;
//    CudaFunc::fill1(b, postSize, bias);
//}

} /* namespace netlib */
