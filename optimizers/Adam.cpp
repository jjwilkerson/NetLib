/**
 * @file
 * @brief Defines Adam class, an ADAM optimizer.
 *
 * An optimizer that uses the ADAM algorithm.
 */

#include "Adam.h"

#include "../Network.h"
#include "../gpu/CudaFunc.h"
#include "../gpu/CublasFunc.h"
#include "../state/IterInfo.h"
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <helper_cuda.h>
#include <cmath>
#include "../config/Config.h"
#include "../loss/LossFunction.h"
#include "../util/FileUtil.h"

#ifdef HOST_DOUBLE
#define dhpow(b,e) pow(b,e)
#else
#define dhpow(b,e) powf(b,e)
#endif

namespace netlib {

dtypeh delta = 1e-4;
Adam::Adam(cublasHandle_t& handle, Network& net, Config config, int startIter, bool debug, string initDeltaFilename)
	: handle(handle), Optimizer(net), debug(debug) {

	unsigned int paramArraySize = net.nParams * sizeof(dtypea);
	checkCudaErrors(cudaMalloc((void **)&m1, paramArraySize));
	checkCudaErrors(cudaMalloc((void **)&m2, paramArraySize));

	if (initDeltaFilename == "") {
		checkCudaErrors(cudaMemset((void *)m1, 0, paramArraySize));
		checkCudaErrors(cudaMemset((void *)m2, 0, paramArraySize));
		t = startIter;
	} else {
		loadInitDelta(initDeltaFilename);
	}

	lr = config.learningRate;
	neg_lr = -lr;
	lrDecay = config.lrDecay;
	gradClipThresh = config.gradClipThresh;
	gradClipMax = config.gradClipMax;

	p1 = config.adamP1;
	p2 = config.adamP2;
	p1_inv = 1.0 - p1;
	p2_inv = 1.0 - p2;

	cout << endl << "lr: " << lr << endl;
	cout << endl << "lrDecay: " << lrDecay << endl;
	cout << endl << "gradClipThresh: " << gradClipThresh << endl;
	cout << endl << "gradClipMax: " << gradClipMax << endl;
	cout << endl << "p1: " << p1 << endl;
	cout << endl << "p2: " << p2 << endl;
}

Adam::~Adam() {
	if (m1 != NULL) {
	    checkCudaErrors(cudaFree(m1));
	}
	if (m2 != NULL) {
	    checkCudaErrors(cudaFree(m2));
	}
}

void Adam::computeUpdate(IterInfo& iterInfo, bool printing) {
	static unsigned long paramArraySize = net.nParams * sizeof(dtypea);

	net.iterInit();

	dtype2 err = net.error(NULL, true, false);

	if (printing) {
		cout << "initial err: " << err << endl;
	}

	iterInfo.initialErr = err;
	iterInfo.lRate = -1 * neg_lr;

	dtype2 *grad = net.calcGrad();

	if (net.isGradOverflow()) {
		checkCudaErrors(cudaFree(grad));
		return;
	}

	dtype2 gradNorm_d;
	CublasFunc::nrm2(handle, net.nParams, grad, 1, &gradNorm_d);
	dtypeh gradNorm = d2h(gradNorm_d);

	if (printing) {
		cout << "grad norm: " << gradNorm << endl;
		int numZero;
		CudaFunc::count(grad, net.nParams, 0.0, &numZero);
		cout << "grad num zero: " << numZero << endl;
	}

	if (gradClipThresh > 0.0f) {
		if (gradNorm > gradClipThresh) {
			if (gradClipMax == 0) {
				if (printing) {
					cout << "clipping gradient norm to " << gradClipThresh << endl;
				}
				dtypeh a = gradClipThresh / gradNorm;
				CublasFunc::scal(handle, net.nParams, &a, grad, 1);
			} else {
				if (printing) {
					cout << "clipping gradient elements to " << gradClipMax << endl;
				}
				CudaFunc::iclip(grad, net.nParams, gradClipMax);
			}

			if (printing) {
				CublasFunc::nrm2(handle, net.nParams, grad, 1, &gradNorm_d);
				cout << "grad norm after clip: " << d2h(gradNorm_d) << endl;
			}
		}
	} else if (gradClipMax > 0.0f) {
		CudaFunc::iclip(grad, net.nParams, gradClipMax);

		if (printing) {
			CublasFunc::nrm2(handle, net.nParams, grad, 1, &gradNorm_d);
			cout << "grad norm after clip: " << d2h(gradNorm_d) << endl;
		}
	}


	iterInfo.gradNorm = gradNorm;

	t++;

	//update biased first moment estimate
	CublasFunc::scalA(handle, net.nParams, &p1, m1, 1);
	CudaFunc::scaleAdd2A(m1, grad, p1_inv, net.nParams);

	//update biased second moment estimate
	CublasFunc::scalA(handle, net.nParams, &p2, m2, 1);
	CudaFunc::isquare(grad, net.nParams);
	CudaFunc::scaleAdd2A(m2, grad, p2_inv, net.nParams);

	//compute update
	dtypea* update;
#ifdef SINGLE_ASINGLE_HALF
	checkCudaErrors(cudaFree(grad));
	checkCudaErrors(cudaMalloc((void **)&update, paramArraySize));
#else
	update = grad;
#endif
	//correct bias in second moment
	dtypeh f2 = 1.0f / (1.0f - dhpow(p2, t));

	checkCudaErrors(cudaMemcpy(update, m2, paramArraySize, cudaMemcpyDeviceToDevice));
	CublasFunc::scalA(handle, net.nParams, &f2, update, 1);
	CudaFunc::isqrtA(update, net.nParams);
	CudaFunc::iaddA(update, net.nParams, delta);

	CudaFunc::divideA(m1, update, update, net.nParams);

	//correct bias in first moment
	dtypeh f1 = 1.0f / (1.0f - dhpow(p1, t));

	CublasFunc::scalA(handle, net.nParams, &f1, update, 1);

#ifdef PARAM_SINGLE_HALF
	dtype1* params = net.W1;
#else
	dtype1* params = net.W;
#endif

	CudaFunc::scaleAddA1(params, update, neg_lr, net.nParams);

	//decay learning rate
	lr *= lrDecay;
	neg_lr = -lr;

	checkCudaErrors(cudaFree(update));

	if (printing) {
		net.copyParams();
		dtype2 err = net.error(NULL, true, false);
		cout << "final err: " << err << endl;
		iterInfo.lRateErr = err;
		iterInfo.improvement = err - iterInfo.initialErr;
		cout << "improvement: " << iterInfo.improvement << endl;
	}
}

dtypeh Adam::getDamping() {
	return 0.0f;
}

dtypeh Adam::getDeltaDecay() {
	return 0.0f;
}

int Adam::getMaxIterCG() {
	return 0;
}

float Adam::getLearningRate() {
	return lr;
}

void Adam::saveInitDelta(const char* filename) {
	ofstream file(filename, ios::binary | ios::trunc);

	FileUtil::writeInt(t, file);

#ifdef SINGLE_ASINGLE_HALF
	dtypea* hostArray = net.h_W1;
#else
	dtypea* hostArray = net.h_W;
#endif

	unsigned arraySize = net.nParams * sizeof(dtypea);
	checkCudaErrors(cudaMemcpy(hostArray, m1, arraySize, cudaMemcpyDeviceToHost));
	for (int i = 0; i < net.nParams; i++) {
		FileUtil::writeFloat(da2float(hostArray[i]), file);
	}

	checkCudaErrors(cudaMemcpy(hostArray, m2, arraySize, cudaMemcpyDeviceToHost));
	for (int i = 0; i < net.nParams; i++) {
		FileUtil::writeFloat(da2float(hostArray[i]), file);
	}

	file.close();
	if (!file) {
		cerr << "write error" << endl;
		exit(EXIT_FAILURE);
	}
}

void Adam::loadInitDelta(string filename) {
	cout << endl << "loadInitDelta" << endl;

	ifstream file(filename.c_str(), ios::binary);

	t = FileUtil::readInt(file);

#ifdef SINGLE_ASINGLE_HALF
	dtypea* hostArray = net.h_W1;
#else
	dtypea* hostArray = net.h_W;
#endif

	unsigned arraySize = net.nParams * sizeof(dtypea);
	for (int i = 0; i < net.nParams; i++) {
		float p = FileUtil::readFloat(file);
		hostArray[i] = float2da(p);
	}
	checkCudaErrors(cudaMemcpy(m1, hostArray, arraySize, cudaMemcpyHostToDevice));

	for (int i = 0; i < net.nParams; i++) {
		float p = FileUtil::readFloat(file);
		hostArray[i] = float2da(p);
	}
	checkCudaErrors(cudaMemcpy(m2, hostArray, arraySize, cudaMemcpyHostToDevice));

	file.close();
}

} /* namespace netlib */

