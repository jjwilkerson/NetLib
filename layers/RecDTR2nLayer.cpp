/**
 * @file
 * @brief Defines RecDTR2nLayer class, a residual-recurrent layer.
 *
 * A residual-recurrent layer. Has 1 or more residual modules.
 */

#include "RecDTR2nLayer.h"

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

const bool resBias = false;

namespace netlib {

RecDTR2nLayer::RecDTR2nLayer(string name, cublasHandle_t& handle, int batchSize, int size, int seqLength, float dropout,
		Nonlinearity* recNonlinearity, Nonlinearity* transNonlinearity,
		float transA1Dropout, float transB11Dropout, float transB1Dropout, int numMod)
			: Layer(name, handle, batchSize, size, seqLength, NULL, dropout) {
	this->recNonlinearity = recNonlinearity;
	this->transNonlinearity = transNonlinearity;
	this->transA1Dropout = transA1Dropout;
	this->transB11Dropout = transB11Dropout;
	this->transB1Dropout = transB1Dropout;

	if (numMod < 1) {
		cerr << "numMod must be >= 1" << endl;
		exit(1);
	}
	this->numAdditional = numMod - 1;

	transB1ParamOffset = new unsigned long[numAdditional];
	transB10ParamOffset = new unsigned long[numAdditional];
	transB1BiasOffset = new unsigned long[numAdditional];
	transB2ParamOffset = new unsigned long[numAdditional];
	transB2BiasOffset = new unsigned long[numAdditional];
	transB11Activation = new dtype2**[numAdditional];
	transB1Activation = new dtype2**[numAdditional];
	transBActivation = new dtype2**[numAdditional];
	d_transB11Activation = new dtype2**[numAdditional];
	d_transB1Activation = new dtype2**[numAdditional];


	if (transB11Dropout != 0.0) {
		transB11DropoutMask = new dtype2*[numAdditional];
	}
	if (transB1Dropout != 0.0) {
		transB1DropoutMask = new dtype2*[numAdditional];
	}
}

RecDTR2nLayer::~RecDTR2nLayer() {
	delete [] transB1ParamOffset;
	delete [] transB10ParamOffset;
	delete [] transB1BiasOffset;
	delete [] transB2ParamOffset;
	delete [] transB2BiasOffset;
	delete [] transB11Activation;
	delete [] transB1Activation;
	delete [] transBActivation;
	delete [] d_transB11Activation;
	delete [] d_transB1Activation;

	if (transB11DropoutMask != NULL) {
		delete [] transB11DropoutMask;
	}
	if (transB1DropoutMask != NULL) {
		delete [] transB1DropoutMask;
	}
}

int RecDTR2nLayer::getNParams() {
	if (nParams == 0) {
		int recParams = (size + 1) * size;
		int transA1Params = (prev->size + 1) * size;
		int transA2Params = (size + 1) * size;
		int nParamsA = recParams + transA1Params + transA2Params;

		int transB1Params = (size + 1) * size;
		int transB10Params = (prev->size) * size;
		int transB2Params = (size + 1) * size;
		int nParamsB = numAdditional * (transB1Params + transB10Params + transB2Params);

		nParams = nParamsA + nParamsB;

		if (bnAfter) {
			nParams -= size;
		}
	}
	return nParams;
}

void RecDTR2nLayer::setParamOffset(int offset) {
	int prevSelf = prev->size * size;
	int selfSelf = size * size;

	recParamOffset = offset;
	recBiasOffset = recParamOffset + selfSelf;
	transA1ParamOffset = recBiasOffset + size;
	transA1BiasOffset = transA1ParamOffset + prevSelf;
	transA2ParamOffset = transA1BiasOffset + size;
	transA2BiasOffset = transA2ParamOffset + selfSelf;

	int curr = transA2BiasOffset + size;
	for (int i = 0; i < numAdditional; i++) {
		transB1ParamOffset[i] = curr;
		transB10ParamOffset[i] = transB1ParamOffset[i] + selfSelf;
		transB1BiasOffset[i] = transB10ParamOffset[i] + prevSelf;
		transB2ParamOffset[i] = transB1BiasOffset[i] + size;
		transB2BiasOffset[i] = transB2ParamOffset[i] + selfSelf;
		curr = transB2BiasOffset[i] + size;
	}

	if (bnAfter) {
		curr -= size;
	}

	cout << name << endl;
	cout << "recParamOffset: " << recParamOffset << endl;
	cout << "recBiasOffset: " << recBiasOffset << endl << endl;
	cout << "transA1ParamOffset: " << transA1ParamOffset << endl;
	cout << "transA1BiasOffset: " << transA1BiasOffset << endl << endl;
	cout << "transA2ParamOffset: " << transA2ParamOffset << endl;
	cout << "transA2BiasOffset: " << transA2BiasOffset << endl << endl;


	for (int i = 0; i < numAdditional; i++) {
		cout << "transB1ParamOffset[" << i << "]: " << transB1ParamOffset[i] << endl;
		cout << "transB10ParamOffset[" << i << "]: " << transB10ParamOffset[i] << endl;
		cout << "transB1BiasOffset[" << i << "]: " << transB1BiasOffset[i] << endl << endl;
		cout << "transB2ParamOffset[" << i << "]: " << transB2ParamOffset[i] << endl;
		cout << "transB2BiasOffset[" << i << "]: " << transB2BiasOffset[i] << endl << endl;
	}
}

void RecDTR2nLayer::initMem(bool training, bool optHF) {
	Layer::initMem(training);

	recActivation = new dtype2*[seqLength];
	transA1Activation = new dtype2*[seqLength];

	if (numAdditional == 0) {
		transAActivation = activation;
	} else {
		transAActivation = new dtype2*[seqLength];
	}

	for (int i = 0; i < numAdditional; i++) {
		transB11Activation[i] = new dtype2*[seqLength];
		transB1Activation[i] = new dtype2*[seqLength];

		if (i == numAdditional-1) {
			transBActivation[i] = activation;
		} else {
			transBActivation[i] = new dtype2*[seqLength];
		}
	}

	if (training) {
		d_recActivation = d_activation;

		d_transA1Activation = new dtype2*[seqLength];

		for (int i = 0; i < numAdditional; i++) {
			d_transB11Activation[i] = new dtype2*[seqLength];
			d_transB1Activation[i] = new dtype2*[seqLength];
		}
	}

	int arraySize = batchSize * size * sizeof(dtype2);
	for (int s = 0; s < seqLength; s++) {
        checkCudaErrors(cudaMalloc((void **)&recActivation[s], arraySize));
        checkCudaErrors(cudaMalloc((void **)&transA1Activation[s], arraySize));
        checkCudaErrors(cudaMalloc((void **)&transAActivation[s], arraySize));

		for (int i = 0; i < numAdditional; i++) {
			checkCudaErrors(cudaMalloc((void **)&transB11Activation[i][s], arraySize));
			checkCudaErrors(cudaMalloc((void **)&transB1Activation[i][s], arraySize));
			checkCudaErrors(cudaMalloc((void **)&transBActivation[i][s], arraySize));

	    	if (training) {
	            checkCudaErrors(cudaMalloc((void **)&d_transB11Activation[i][s], arraySize));
	            checkCudaErrors(cudaMalloc((void **)&d_transB1Activation[i][s], arraySize));
	    	}
		}

    	if (training) {
            checkCudaErrors(cudaMalloc((void **)&d_transA1Activation[s], arraySize));
    	}
	}

	bool haveTransDopout = false;

	if (transA1Dropout != 0.0) {
		haveTransDopout = true;
		checkCudaErrors(cudaMalloc((void **)&transA1DropoutMask, arraySize));
	}

	if (transB11Dropout != 0.0) {
		haveTransDopout = true;
		for (int i = 0; i < numAdditional; i++) checkCudaErrors(cudaMalloc((void **)&transB11DropoutMask[i], arraySize));
	}

	if (transB1Dropout != 0.0) {
		haveTransDopout = true;
		for (int i = 0; i < numAdditional; i++) checkCudaErrors(cudaMalloc((void **)&transB1DropoutMask[i], arraySize));
	}

	if (haveTransDopout) {
		checkCudaErrors(cudaMalloc((void **)&transDoActivation, arraySize));
	}

	checkCudaErrors(cudaMalloc((void **)&singleShared3, arraySize));
}

void RecDTR2nLayer::freeMem(bool training) {
	Layer::freeMem(training);

	for (int s = 0; s < seqLength; s++) {
        checkCudaErrors(cudaFree(recActivation[s]));
        checkCudaErrors(cudaFree(transA1Activation[s]));
        if (transAActivation != activation) checkCudaErrors(cudaFree(transAActivation[s]));

        for (int i = 0; i < numAdditional; i++) {
			checkCudaErrors(cudaFree(transB11Activation[i][s]));
			checkCudaErrors(cudaFree(transB1Activation[i][s]));
			if (transBActivation[i] != activation) checkCudaErrors(cudaFree(transBActivation[i][s]));

	    	if (training) {
				checkCudaErrors(cudaFree(d_transB11Activation[i][s]));
				checkCudaErrors(cudaFree(d_transB1Activation[i][s]));
	    	}
        }

    	if (training) {
			checkCudaErrors(cudaFree(d_transA1Activation[s]));
    	}
    }

	delete [] recActivation;
	delete [] transA1Activation;
	if (transAActivation != activation) delete [] transAActivation;

    for (int i = 0; i < numAdditional; i++) {
		delete [] transB11Activation[i];
		delete [] transB1Activation[i];
		if (transBActivation[i] != activation) delete [] transBActivation[i];

		if (training) {
			delete [] d_transB11Activation[i];
			delete [] d_transB1Activation[i];
		}
    }

	if (training) {
		delete [] d_transA1Activation;
	}


	if (transA1DropoutMask != NULL) {
		checkCudaErrors(cudaFree(transA1DropoutMask));
	}
	if (transB11DropoutMask != NULL) {
		for (int i = 0; i < numAdditional; i++) checkCudaErrors(cudaFree(transB11DropoutMask[i]));
	}
	if (transB1DropoutMask != NULL) {
		for (int i = 0; i < numAdditional; i++) checkCudaErrors(cudaFree(transB1DropoutMask[i]));
	}

	if (transDoActivation != NULL) {
		checkCudaErrors(cudaFree(transDoActivation));
	}

	checkCudaErrors(cudaFree(singleShared3));
}

void RecDTR2nLayer::initWeights(WeightInit& ffWeightInit, WeightInit& recWeightInit) {
	dtype1* params = net->getMasterParams();
	recWeightInit.initialize(params+recParamOffset, size, size);
	checkCudaErrors(cudaMemset(params+recBiasOffset, 0, size * sizeof(dtype1)));

	ffWeightInit.initialize(params+transA1ParamOffset, prev->size, size);
	checkCudaErrors(cudaMemset(params+transA1BiasOffset, 0, size * sizeof(dtype1)));
	ffWeightInit.initialize(params+transA2ParamOffset, size, size);
	if (numAdditional > 0 || !bnAfter) {
		checkCudaErrors(cudaMemset(params+transA2BiasOffset, 0, size * sizeof(dtype1)));
	}

    for (int i = 0; i < numAdditional; i++) {
		ffWeightInit.initialize(params+transB1ParamOffset[i], size, size);
		ffWeightInit.initialize(params+transB10ParamOffset[i], prev->size, size);
		checkCudaErrors(cudaMemset(params+transB1BiasOffset[i], 0, size * sizeof(dtype1)));
		ffWeightInit.initialize(params+transB2ParamOffset[i], size, size);

		if (!bnAfter || i < numAdditional-1) {
			checkCudaErrors(cudaMemset(params+transB2BiasOffset[i], 0, size * sizeof(dtype1)));
		}
    }
}

void RecDTR2nLayer::iterInit() {
	Layer::iterInit();
	if (transA1Dropout != 0) {
		CudaFunc::fillDropoutMask(transA1DropoutMask, batchSize * size, transA1Dropout);
	}
	if (transB11Dropout != 0) {
		for (int i = 0; i < numAdditional; i++) CudaFunc::fillDropoutMask(transB11DropoutMask[i], batchSize * size, transB11Dropout);
	}
	if (transB1Dropout != 0) {
		for (int i = 0; i < numAdditional; i++) CudaFunc::fillDropoutMask(transB1DropoutMask[i], batchSize * size, transB1Dropout);
	}
}

void RecDTR2nLayer::forward(int batchNum, dtype2* params, bool deriv, bool stochasticDropout) {
//	cout << "RecDTR2nLayer::forward" << endl;
	if (params == NULL) {
		params = W;
	}

	dtype2* wTransA1 = params + transA1ParamOffset;
	dtype2* bTransA1 = params + transA1BiasOffset;
	dtype2* wTransA2 = params + transA2ParamOffset;
	dtype2* bTransA2 = params + transA2BiasOffset;

	int preSize = prev->size;
	int arraySize = batchSize * size * sizeof(dtype2);

	for (int s = 0; s < seqLength; s++) {
		dtype2* act = activation[s];
		dtype2* transA1Act = transA1Activation[s];
		dtype2* transAAct = transAActivation[s];

		int seqIx;
		if (prev->seqLength == 1) {
			seqIx = 0;
		} else {
			seqIx = s;
		}

		dtype2* prevDoact;
		if (prev->dropout == 0) {
			prevDoact = prev->activation[seqIx];
		} else {
			prevDoact = prev->doActivation;
			int arraySize = batchSize * preSize * sizeof(dtype2);
			checkCudaErrors(cudaMemcpy(prevDoact,  prev->activation[seqIx], arraySize,
					cudaMemcpyDeviceToDevice));

			if (stochasticDropout) {
				CudaFunc::multiply(prevDoact, prev->dropoutMask, prevDoact, batchSize * preSize);
			}
		}

		//compute transition 11 activation
		CublasFunc::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, batchSize, size, preSize, &one,
				prevDoact, batchSize, wTransA1, preSize, &zero, ffInput, batchSize);
		CudaFunc::addRowVecMat(ffInput, bTransA1, batchSize, size, batchSize);

		//compute recurrent input
		if (s > 0) {
			dtype2* recAct = recActivation[s-1];
			recNonlinearity->activation(size, activation[s-1], recAct);
			applyMask(recAct, size, s, batchNum);
			if (deriv) {
				dtype2* d_act = d_recActivation[s-1];
				recNonlinearity->d_activation(size, activation[s-1], recAct, d_act);
				applyMask(d_act, size, s, batchNum);
			}

			dtype2* w = params + recParamOffset;
			CublasFunc::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, batchSize, size, size, &one,
					recAct, batchSize, w, size, &one, ffInput, batchSize);
		} else {
			dtype2* b = params + recBiasOffset;
			CudaFunc::addRowVecMat(ffInput, b, batchSize, size, batchSize);
		}

		transNonlinearity->activation(size, ffInput, transA1Act);
		applyMask(transA1Act, size, s, batchNum);

		if (deriv) {
			dtype2* d_transAct = d_transA1Activation[s];
			transNonlinearity->d_activation(size, ffInput, transA1Act, d_transAct);
			applyMask(d_transAct, size, s, batchNum);
		}

		dtype2* doact;
		if (transA1Dropout == 0) {
			doact = transA1Act;
		} else {
			cerr << "trans dropout not supported" << endl;
			exit(1);
//			doact = transDoActivation;
//			checkCudaErrors(cudaMemcpy(doact,  transAct, arraySize,
//					cudaMemcpyDeviceToDevice));
//
//			if (stochasticDropout) {
//				CudaFunc::multiply(doact, transDropoutMask, doact, batchSize * size);
//			}
		}

		//compute transition 1 activation
		CublasFunc::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, batchSize, size, size, &one,
				doact, batchSize, wTransA2, size, &zero, transAAct, batchSize);

		if (!bnAfter || numAdditional > 0) {
			CudaFunc::addRowVecMat(transAAct, bTransA2, batchSize, size, batchSize);
		}

		applyMask(transAAct, size, s, batchNum);

		//residual connection 1
		if (s > 0) {
			CublasFunc::axpy(handle, batchSize * size, &one, activation[s-1], 1, transAAct, 1);
		} else if (resBias) {
			dtype2* b = params + recBiasOffset;
			CudaFunc::addRowVecMat(transAAct, b, batchSize, size, batchSize);
		}

		dtype2 *prevAct = transAAct;
	    for (int i = 0; i < numAdditional; i++) {
	    	dtype2* wTransB1 = params + transB1ParamOffset[i];
	    	dtype2* bTransB1 = params + transB1BiasOffset[i];
	    	dtype2* wTransB10 = params + transB10ParamOffset[i];
	    	dtype2* wTransB2 = params + transB2ParamOffset[i];
	    	dtype2* bTransB2 = params + transB2BiasOffset[i];

	    	dtype2* transB11Act = transB11Activation[i][s];
			dtype2* transB1Act = transB1Activation[i][s];
			dtype2* transBAct = transBActivation[i][s];

			//compute transition 211 activation
			transNonlinearity->activation(size, prevAct, transB11Act);
			applyMask(transB11Act, size, s, batchNum);

			if (deriv) {
				dtype2* d_transAct = d_transB11Activation[i][s];
				transNonlinearity->d_activation(size, prevAct, transB11Act, d_transAct);
				applyMask(d_transAct, size, s, batchNum);
			}

			//compute transition 21 activation
			if (transB11Dropout == 0) {
				doact = transB11Act;
			} else {
				cerr << "trans dropout not supported" << endl;
				exit(1);
			}

			CublasFunc::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, batchSize, size, size, &one,
					doact, batchSize, wTransB1, size, &zero, ffInput, batchSize);
			CublasFunc::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, batchSize, size, preSize, &one,
					prevDoact, batchSize, wTransB10, preSize, &one, ffInput, batchSize);
			CudaFunc::addRowVecMat(ffInput, bTransB1, batchSize, size, batchSize);


			transNonlinearity->activation(size, ffInput, transB1Act);
			applyMask(transB1Act, size, s, batchNum);

			if (deriv) {
				dtype2* d_transAct = d_transB1Activation[i][s];
				transNonlinearity->d_activation(size, ffInput, transB1Act, d_transAct);
				applyMask(d_transAct, size, s, batchNum);
			}

			//compute activation
			if (transB1Dropout == 0) {
				doact = transB1Act;
			} else {
				cerr << "trans dropout not supported" << endl;
				exit(1);
			}

			CublasFunc::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, batchSize, size, size, &one,
					doact, batchSize, wTransB2, size, &zero, transBAct, batchSize);

			if (!bnAfter || i < numAdditional-1) {
				CudaFunc::addRowVecMat(transBAct, bTransB2, batchSize, size, batchSize);
			}

			//residual connection 2
			CublasFunc::axpy(handle, batchSize * size, &one, prevAct, 1, transBAct, 1);
			applyMask(transBAct, size, s, batchNum);

			prevAct = transBAct;
	    }
	}

	if (next != NULL) {
		next->forward(batchNum, params, deriv, stochasticDropout);
	}
}

void RecDTR2nLayer::calcGrad(dtype2* grad, int batchNum) {
	dtype2* wTransA1 = W + transA1ParamOffset;
	dtype2* bTransA1 = W + transA1BiasOffset;
	dtype2* wTransA2 = W + transA2ParamOffset;
	dtype2* bTransA2 = W + transA2BiasOffset;

	dtype2* wGradA1 = grad + transA1ParamOffset;
	dtype2* bGradA1 = grad + transA1BiasOffset;
	dtype2* wGradA2 = grad + transA2ParamOffset;
	dtype2* bGradA2 = grad + transA2BiasOffset;
	dtype2* delta;
	dtype2** prevError = prev->error;
	dtype2* doact;
	dtype2* prevDoact;

	int seqEnd = seqLength - 1;
	int arraySize = batchSize * size * sizeof(dtype2);
	int preSize = prev->size;

	for (int s = seqEnd; s >= 0; s--) {
		dtype2* transA1Act = transA1Activation[s];
		dtype2* transAAct = transAActivation[s];
		prevDoact = NULL;

		int seqIx;
		if (prev->seqLength == 1) {
			seqIx = 0;
		} else {
			seqIx = s;
		}

		if (isOutputLayer()) {
			net->lossFunction.d_loss(net->getTargets(), this, s, error[s], true, net->matchMasks(batchNum));
		}

		dtype2* errorAfter = error[s];

	    for (int i = numAdditional-1; i >= 0; i--) {
	    	dtype2* wTransB1 = W + transB1ParamOffset[i];
	    	dtype2* bTransB1 = W + transB1BiasOffset[i];
	    	dtype2* wTransB10 = W + transB10ParamOffset[i];
	    	dtype2* wTransB2 = W + transB2ParamOffset[i];
	    	dtype2* bTransB2 = W + transB2BiasOffset[i];

	    	dtype2* wGradB1 = grad + transB1ParamOffset[i];
	    	dtype2* bGradB1 = grad + transB1BiasOffset[i];
	    	dtype2* wGradB10 = grad + transB10ParamOffset[i];
	    	dtype2* wGradB2 = grad + transB2ParamOffset[i];
	    	dtype2* bGradB2 = grad + transB2BiasOffset[i];

			dtype2* transB11Act = transB11Activation[i][s];
			dtype2* transB1Act = transB1Activation[i][s];

			delta = errorAfter;

			// trans 22 grad
			if (transB1Dropout == 0) {
				doact = transB1Act;
			} else {
				cerr << "trans dropout not supported" << endl;
				exit(1);

	//			doact = transDoActivation;
	//			checkCudaErrors(cudaMemcpy(doact,  transAct, arraySize, cudaMemcpyDeviceToDevice));
	//
	//			CudaFunc::multiply(doact, transDropoutMask, doact, batchSize * size);
			}

			CublasFunc::gemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, size, size, batchSize, &one,
					doact, batchSize, delta, batchSize, &one, wGradB2, size);

			if (!bnAfter || i < numAdditional-1) {
				CudaFunc::sum_cols_reduce4(delta, bGradB2, batchSize, size);
			}

			//trans 22 bp
			if (transB1Dropout == 0) {
				CublasFunc::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, batchSize, size, size, &one,
						delta, batchSize, wTransB2, size, &zero, singleShared2, batchSize);
			} else {
				cerr << "trans dropout not supported" << endl;
				exit(1);

	//			CublasFunc::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, batchSize, size, size, &one,
	//					delta, batchSize, w, size, &zero, error[s], batchSize);
	//			CudaFunc::multiply(error[s], transDropoutMask, error[s], batchSize * size);
			}

			delta = singleShared1;
			CudaFunc::multiply(d_transB1Activation[i][s], singleShared2, delta, batchSize * size);
			applyMask(delta, size, s, batchNum); //needed?

			// trans 21 grad
			if (transB11Dropout == 0) {
				doact = transB11Act;
			} else {
				cerr << "trans dropout not supported" << endl;
				exit(1);
			}
			CublasFunc::gemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, size, size, batchSize, &one,
					doact, batchSize, delta, batchSize, &one, wGradB1, size);
			CudaFunc::sum_cols_reduce4(delta, bGradB1, batchSize, size);

			//skip connection
			if (prevDoact == NULL) {
				if (prev->dropout == 0) {
					prevDoact = prev->activation[seqIx];
				} else {
					prevDoact = prev->doActivation;
					int arraySize = batchSize * preSize * sizeof(dtype2);
					checkCudaErrors(cudaMemcpy(prevDoact,  prev->activation[seqIx], arraySize,
							cudaMemcpyDeviceToDevice));

					CudaFunc::multiply(prevDoact, prev->dropoutMask, prevDoact, batchSize * preSize);
				}
			}

			CublasFunc::gemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, preSize, size, batchSize, &one,
					prevDoact, batchSize, delta, batchSize, &one, wGradB10, preSize);

			//skip connection bp
			if (prev->dropout == 0) {
				CublasFunc::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, batchSize, preSize, size, &one,
						delta, batchSize, wTransB10, preSize, &one, prevError[seqIx], batchSize);
			} else {
				dtype2* doerror = prev->singleShared1;
				int arraySize = batchSize * preSize * sizeof(dtype2);
				CublasFunc::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, batchSize, preSize, size, &one,
						delta, batchSize, wTransB10, preSize, &zero, doerror, batchSize);
				CudaFunc::multiply(doerror, prev->dropoutMask, doerror, batchSize * preSize);
				CublasFunc::axpy(handle, batchSize * preSize, &one, doerror, 1, prevError[seqIx], 1);
			}

			//trans 21 bp
			if (transB11Dropout == 0) {
				CublasFunc::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, batchSize, size, size, &one,
						delta, batchSize, wTransB1, size, &zero, singleShared2, batchSize);
			} else {
				cerr << "trans dropout not supported" << endl;
				exit(1);
			}

			CudaFunc::multiply(d_transB11Activation[i][s], singleShared2, delta, batchSize * size);
			applyMask(delta, size, s, batchNum); //needed?

			//residual connection 2
			CublasFunc::axpy(handle, batchSize * size, &one, errorAfter, 1, delta, 1);

			errorAfter = singleShared3;
			checkCudaErrors(cudaMemcpy(errorAfter, delta, arraySize, cudaMemcpyDeviceToDevice));
	    }

	    delta = errorAfter;

		//residual connection 1 (recurrent)
		if (s > 0) {
			CublasFunc::axpy(handle, batchSize * size, &one, delta, 1, error[s-1], 1);
		} else if (resBias) {
			dtype2* bGrad = grad + recBiasOffset;
		    CudaFunc::sum_cols_reduce4(delta, bGrad, batchSize, size);
		}

		//trans 12 grad
		if (transA1Dropout == 0) {
			doact = transA1Act;
		} else {
			cerr << "trans dropout not supported" << endl;
			exit(1);

//			doact = transDoActivation;
//			checkCudaErrors(cudaMemcpy(doact,  transAct, arraySize, cudaMemcpyDeviceToDevice));
//
//			CudaFunc::multiply(doact, transDropoutMask, doact, batchSize * size);
		}

		CublasFunc::gemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, size, size, batchSize, &one,
				doact, batchSize, delta, batchSize, &one, wGradA2, size);

		if (!bnAfter || numAdditional > 0) {
			CudaFunc::sum_cols_reduce4(delta, bGradA2, batchSize, size);
		}

		//trans 12 bp
		if (transA1Dropout == 0) {
			CublasFunc::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, batchSize, size, size, &one,
					delta, batchSize, wTransA2, size, &zero, singleShared2, batchSize);
		} else {
			cerr << "trans dropout not supported" << endl;
			exit(1);
		}

		CudaFunc::multiply(d_transA1Activation[s], singleShared2, delta, batchSize * size);
		applyMask(delta, size, s, batchNum); //needed?

		// gradient for recurrent weights
		if (s > 0) {
			dtype2* wGrad = grad + recParamOffset;
			CublasFunc::gemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, size, size, batchSize, &one,
					recActivation[s-1], batchSize, delta, batchSize, &one, wGrad, size);
		} else {
			//put remaining gradient into initial bias
			dtype2* bGrad = grad + recBiasOffset;
		    CudaFunc::sum_cols_reduce4(delta, bGrad, batchSize, size);
		}

		// feedforward connections
		if (prevDoact == NULL) {
			if (prev->dropout == 0) {
				prevDoact = prev->activation[seqIx];
			} else {
				prevDoact = prev->doActivation;
				int arraySize = batchSize * preSize * sizeof(dtype2);
				checkCudaErrors(cudaMemcpy(prevDoact,  prev->activation[seqIx], arraySize,
						cudaMemcpyDeviceToDevice));

				CudaFunc::multiply(prevDoact, prev->dropoutMask, prevDoact, batchSize * preSize);
			}
		}
		CublasFunc::gemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, preSize, size, batchSize, &one,
				prevDoact, batchSize, delta, batchSize, &one, wGradA1, preSize);
		CudaFunc::sum_cols_reduce4(delta, bGradA1, batchSize, size);

		// backpropagate error
		if (prev->dropout == 0) {
			CublasFunc::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, batchSize, preSize, size, &one,
					delta, batchSize, wTransA1, preSize, &one, prevError[seqIx], batchSize);
		} else {
			dtype2* doerror = prev->singleShared1;
			int arraySize = batchSize * preSize * sizeof(dtype2);
			CublasFunc::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, batchSize, preSize, size, &one,
					delta, batchSize, wTransA1, preSize, &zero, doerror, batchSize);
			CudaFunc::multiply(doerror, prev->dropoutMask, doerror, batchSize * preSize);
			CublasFunc::axpy(handle, batchSize * preSize, &one, doerror, 1, prevError[seqIx], 1);
		}

		// add recurrent error for step before
		if (s > 0) {
			dtype2* w = W + recParamOffset;
			CublasFunc::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, batchSize, size, size, &one,
					delta, batchSize, w, size, &zero, singleShared2, batchSize);
			CudaFunc::multiply(d_recActivation[s-1], singleShared2, singleShared2, batchSize * size);
			CublasFunc::axpy(handle, batchSize * size, &one, singleShared2, 1, error[s-1], 1);
		}
	}

	prev->calcGrad(grad, batchNum);
}

void RecDTR2nLayer::addGradL2(dtype2* grad, dtypeh l2) {
	int numWeightsPrevSelf = prev->size * size;
	int numWeightsSelfSelf = size * size;

	CublasFunc::axpy(handle, numWeightsSelfSelf, &l2, W+recParamOffset, 1, grad+recParamOffset, 1);

	CublasFunc::axpy(handle, numWeightsPrevSelf, &l2, W+transA1ParamOffset, 1, grad+transA1ParamOffset, 1);
	CublasFunc::axpy(handle, numWeightsSelfSelf, &l2, W+transA2ParamOffset, 1, grad+transA2ParamOffset, 1);

    for (int i = 0; i < numAdditional; i++) {
		CublasFunc::axpy(handle, numWeightsSelfSelf, &l2, W+transB1ParamOffset[i], 1, grad+transB1ParamOffset[i], 1);
		CublasFunc::axpy(handle, numWeightsPrevSelf, &l2, W+transB10ParamOffset[i], 1, grad+transB10ParamOffset[i], 1);
		CublasFunc::axpy(handle, numWeightsSelfSelf, &l2, W+transB2ParamOffset[i], 1, grad+transB2ParamOffset[i], 1);
    }
}

void RecDTR2nLayer::Rforward(dtype2* v, int batchNum) {
	cerr << "RecDTR2nLayer::Rforward not implemented!" << endl;
	exit(1);
}

void RecDTR2nLayer::Rback(dtype2* Gv, int batchNum) {
	cerr << "RecDTR2nLayer::Rback not implemented!" << endl;
	exit(1);
}

void RecDTR2nLayer::clearRact() {
	cerr << "RecDTR2nLayer::clearRact not implemented!" << endl;
	exit(1);
}

bool RecDTR2nLayer::hasParam(int i) {
	unsigned long endPos;
	if (numAdditional == 0) {
		endPos = transA2BiasOffset;
		if (!bnAfter) {
			endPos += size;
		}
	} else {
		endPos = transB2BiasOffset[numAdditional-1];
		if (!bnAfter) {
			endPos += size;
		}
	}
	return (i >= recParamOffset) && (i < endPos);
}

bool RecDTR2nLayer::isWeight(int i) {
	if (((i >= recParamOffset) && (i < recBiasOffset))
			|| ((i >= transA1ParamOffset) && (i < transA1BiasOffset))
			|| ((i >= transA2ParamOffset) && (i < transA2BiasOffset))) {
		return true;
	}

    for (int m = 0; m < numAdditional; m++) {
    	if (((i >= transB1ParamOffset[m]) && (i < transB1BiasOffset[m]))
    			|| ((i >= transB2ParamOffset[m]) && (i < transB2BiasOffset[m]))) {
    		return true;
    	}
    }

    return false;
}

} /* namespace netlib */

