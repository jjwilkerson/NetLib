/**
 * @file
 * @brief Defines BaseLayer class, a base layer.
 *
 * A base network layer that is meant to be used with a subclass of Core. Implements functionality common to all layers, while
 * the Core subclass implements the specific operations of a layer.
 *
 */

#include "BaseLayer.h"
#include "cores/Core.h"
#include "../Network.h"
#include "../gpu/CudaFunc.h"
#include "../gpu/CublasFunc.h"
#include <iostream>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "../config/WeightInit.h"
#include "../loss/LossFunction.h"
#include "../nonlinearity/Nonlinearity.h"

using namespace std;

namespace netlib {

BaseLayer::BaseLayer(Core* core, string name, cublasHandle_t& handle, int batchSize,
		int size, int seqLength, Nonlinearity* nonlinearity, float dropout)
	: core(core), Layer(name, handle, batchSize, size, seqLength, nonlinearity, dropout) {
}

BaseLayer::~BaseLayer() {
}

int BaseLayer::getNParams() {
	if (nParams == 0) {
		nParams = core->getNParams();
	}
	return nParams;
}

void BaseLayer::setParamOffset(int offset) {
	paramOffset = offset;
	core->setParamOffset(offset);
}

void BaseLayer::initMem(bool training, bool optHF) {
	Layer::initMem(training, optHF);
	core->initMem(training, optHF);
}

void BaseLayer::freeMem(bool training) {
	Layer::freeMem(training);
	core->freeMem(training);
}

void BaseLayer::initWeights(WeightInit& ffWeightInit, WeightInit& recWeightInit) {
	dtype1* params = net->getMasterParams();
	Layer::initWeights(ffWeightInit, recWeightInit);
	core->initWeights(params, ffWeightInit, recWeightInit);
}

void BaseLayer::forward(int batchNum, dtype2* params, bool deriv,
		bool stochasticDropout) {
	if (params == NULL) {
		params = W;
	}

	int preSize = prev->size;

	int arraySize = batchSize * sizeof(unsigned int);
	unsigned int* d_inputLengths = net->getDInputLengths()[batchNum];
	unsigned int* h_inputLengths = (unsigned int*) malloc(arraySize);
	checkCudaErrors(cudaMemcpy(h_inputLengths, d_inputLengths, arraySize, cudaMemcpyDeviceToHost));

	for (int s = 0; s < seqLength; s++) {
		dtype2* act = activation[s];

		int seqIx;
		if (prev->seqLength == 1) {
			seqIx = 0;
		} else {
			seqIx = s;
		}

		dtype2* doact;
		if (prev->dropout == 0 || !stochasticDropout) {
			doact = prev->activation[seqIx];
		} else {
			doact = prev->doActivation;
			int arraySize = batchSize * preSize * sizeof(dtype2);
			checkCudaErrors(cudaMemcpy(doact,  prev->activation[seqIx], arraySize,
					cudaMemcpyDeviceToDevice));

			CudaFunc::multiply(doact, prev->dropoutMask, doact, batchSize * preSize);
		}

		dtype2* priorAct = (s > 0)?activation[s-1]:NULL;
		core->forward(doact, ffInput, params, h_inputLengths, d_inputLengths, deriv,
				priorAct, s);

		nonlinearity->activation(size, ffInput, act);

		if (seqLength > 1) {
			applyMask(act, size, s, batchNum);
		}

		if (deriv && (!isOutputLayer() || !net->lossFunction.derivCombo())) {
			dtype2* d_act = d_activation[s];
			nonlinearity->d_activation(size, ffInput, act, d_act);
			if (seqLength > 1) {
				applyMask(d_act, size, s, batchNum);
			}
		}
	}

	free(h_inputLengths);

	if (next != NULL) {
		next->forward(batchNum, params, deriv, stochasticDropout);
	}
}

void BaseLayer::calcGrad(dtype2* grad, int batchNum) {
	dtype2* delta = singleShared1;
	dtype2** prevError = prev->error;

	int seqEnd = seqLength - 1;

	int arraySize = batchSize * sizeof(unsigned int);
	unsigned int* d_inputLengths = net->getDInputLengths()[batchNum];
	unsigned int* h_inputLengths = (unsigned int*) malloc(arraySize);
	checkCudaErrors(cudaMemcpy(h_inputLengths, d_inputLengths, arraySize, cudaMemcpyDeviceToHost));

	for (int s = seqEnd; s >= 0; s--) {
		if (isOutputLayer()) {
			net->lossFunction.d_loss(net->getTargets(), this, s, error[s], true, net->matchMasks(batchNum),
					net->getDInputLengths()[batchNum]);
		}

//		applyMask(currError, size, s, batchNum); //should apply mask to delta instead?

		// compute deltas
		if (isOutputLayer() && net->lossFunction.derivCombo()) {
//			delta = error[s];
			int arraySize = batchSize * size * sizeof(dtype2);
			checkCudaErrors(cudaMemcpy(delta,  error[s], arraySize, cudaMemcpyDeviceToDevice));
		} else {
			CudaFunc::multiply(d_activation[s], error[s], delta, batchSize * size);
			applyMask(delta, size, s, batchNum); //?
		}

		// feedforward connections
		int seqIx;
		if (prev->seqLength == 1) {
			seqIx = 0;
		} else {
			seqIx = s;
		}

		int preSize = prev->size;

		dtype2* doact;
		if (prev->dropout == 0) {
			doact = prev->activation[seqIx];
		} else {
			doact = prev->doActivation;
			int arraySize = batchSize * preSize * sizeof(dtype2);
			checkCudaErrors(cudaMemcpy(doact,  prev->activation[seqIx], arraySize,
					cudaMemcpyDeviceToDevice));

			CudaFunc::multiply(doact, prev->dropoutMask, doact, batchSize * preSize);
		}

		dtype2* act = activation[s];
		dtype2* priorAct = (s > 0)?activation[s-1]:NULL;
		dtype2* priorError = (s > 0)?error[s-1]:NULL;
		if (prev->dropout == 0) {
			core->calcGrad(doact, act, prevError[seqIx], delta, grad, W, h_inputLengths, d_inputLengths,
					priorAct, priorError, s);
		} else {
			dtype2* doerror = prev->doActivation;
			int arraySize = batchSize * preSize * sizeof(dtype2);
			checkCudaErrors(cudaMemset(doerror, 0, arraySize));

			core->calcGrad(doact, act, doerror, delta, grad, W, h_inputLengths, d_inputLengths,
					priorAct, priorError, s);

			CudaFunc::multiply(doerror, prev->dropoutMask, doerror, batchSize * preSize);
			CublasFunc::axpy(handle, batchSize * preSize, &one, doerror, 1, prevError[seqIx], 1);
		}
	}

	free(h_inputLengths);

	prev->calcGrad(grad, batchNum);
}

void BaseLayer::addGradL2(dtype2* grad, dtypeh l2) {
	core->addGradL2(W, grad, l2);
}

void BaseLayer::Rforward(dtype2* v, int batchNum) {
	cerr << "BaseLayer::Rforward not implemented!" << endl;
	exit(1);
}

void BaseLayer::Rback(dtype2* Gv, int batchNum) {
	cerr << "BaseLayer::Rback not implemented!" << endl;
	exit(1);
}

void BaseLayer::setPrev(Layer* p) {
	Layer::setPrev(p);
	core->setPreSize(p->size);
}

void BaseLayer::setBnAfter(bool b) {
	Layer::setBnAfter(b);
	core->setBnAfter(b);
}

} /* namespace netlib */

