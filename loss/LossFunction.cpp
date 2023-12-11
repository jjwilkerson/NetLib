/**
 * @file
 * @brief Defines classes that implement loss functions.
 *
 */

#include "LossFunction.h"

#include "../gpu/CudaFunc.h"
#include "../gpu/CublasFunc.h"
#include "../Network.h"
#include "../layers/Layer.h"
#include <iostream>
#include <cassert>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "cublas_v2.h"

#define EPS 1e-6

using namespace std;

namespace netlib {

LossFunction::LossFunction(int batchSize, int maxSeqLength) : batchSize(batchSize), maxSeqLength(maxSeqLength) {
}

LossFunction::~LossFunction() {
}

void LossFunction::setNetwork(Network* n) {
	net = n;
}

dtypeh LossFunction::batchLoss(dtype2** outputs, dtype2** targets, dtype2** losses_d, bool average, dtype2** masks,
		unsigned int* d_inputLengths) {
	dtypeh totalLoss = 0;
	for (int s = 0; s < maxSeqLength; s++) {
		totalLoss += loss(outputs, targets, s, masks, d_inputLengths);
	}

	return totalLoss;
}

bool LossFunction::derivCombo() {
	return false;
}

LossSet::LossSet(int batchSize, int maxSeqLength, int numLoss, LossFunction** lossFunctions)
		: LossFunction(batchSize, maxSeqLength), numLoss(numLoss), lossFunctions(lossFunctions) {
}

LossSet::~LossSet() {
}

void LossSet::setNetwork(Network* n) {
	LossFunction::setNetwork(n);
	for (int f = 0; f < numLoss; f++) {
		lossFunctions[f]->setNetwork(n);
	}
}

dtypeh LossSet::loss(dtype2** outputs, dtype2** targets, int s, dtype2** masks,
		unsigned int* d_inputLengths) {
	dtypeh sum = 0;
	for (int f = 0; f < numLoss; f++) {
		sum += lossFunctions[f]->loss(outputs, targets, s, masks, d_inputLengths);
	}
	return sum;
}

void LossSet::d_loss(dtype2** targets, Layer *layer, int s,
		dtype2* out, bool increment, dtype2** masks,
		unsigned int* d_inputLengths) {
	if (!increment) {
		checkCudaErrors(cudaMemset(out, 0, batchSize * layer->size * sizeof(dtype2)));
	}
	for (int f = 0; f < numLoss; f++) {
		lossFunctions[f]->d_loss(targets, layer, s, out, true, masks);
	}
}

void LossSet::d2_loss(dtype2** targets, Layer *layer, int s,
		dtype2* out, bool increment, dtype2** masks) {
	if (!increment) {
		checkCudaErrors(cudaMemset(out, 0, batchSize * layer->size * sizeof(dtype2)));
	}
	for (int f = 0; f < numLoss; f++) {
		lossFunctions[f]->d2_loss(targets, layer, s, out, true, masks);
	}
}

SquaredError::SquaredError(int batchSize, int maxSeqLength, cublasHandle_t& handle)
		: LossFunction(batchSize, maxSeqLength), handle(handle) {
}

SquaredError::~SquaredError() {
    checkCudaErrors(cudaFree(intermed));
    checkCudaErrors(cudaFree(d_sum));
}

void SquaredError::setNetwork(Network* n) {
	LossFunction::setNetwork(n);

	outputLength = net->getOutputLayer()->size;
    checkCudaErrors(cudaMalloc((void **)&intermed, batchSize * outputLength * sizeof(dtype2)));
    checkCudaErrors(cudaMalloc((void **)&d_sum, sizeof(dtype2)));
}

dtypeh SquaredError::loss(dtype2** outputs, dtype2** targets, int s, dtype2** masks,
		unsigned int* d_inputLengths) {
	int print_s = -1;
	int print_b = 3;

	int length = batchSize * outputLength;
	int arraySize = length * sizeof(dtype2);

	checkCudaErrors(cudaMemcpy(intermed, outputs[s], arraySize, cudaMemcpyDeviceToDevice));
	CublasFunc::axpy(handle, length, &minus_one, targets[s], 1, intermed, 1);

//	CudaFunc::squaredDiff(outputs[s], targets[s], intermed, length);
	checkCudaErrors(cudaDeviceSynchronize());

//	CudaFunc::maskByLength(intermed, d_inputLengths, s, outputLength, batchSize);
	if (masks != NULL) {
		dtype2* mask = masks[s];
		CudaFunc::maskColVecMat(intermed, mask, batchSize, outputLength, batchSize);
	}

	dtypeh sum;
	CublasFunc::dot(handle, length, intermed, 1, intermed, 1, &sum);

//	CublasFunc::asum(handle, length, intermed, 1, &sum);

	return sum / 2;
}

void SquaredError::d_loss(dtype2** targets, Layer *layer, int s,
		dtype2* out, bool increment, dtype2** masks,
		unsigned int* d_inputLengths) {
	int size = batchSize * layer->size;

	if (layer->isOutputLayer()) {
		if (!increment) {
			checkCudaErrors(cudaMemset(out, 0, size * sizeof(dtype2)));
		}

		const dtypeh plus = net->getLossScaleFac();
		const dtypeh minus = -plus;

		CublasFunc::axpy(handle, size, &plus, layer->activation[s], 1, out, 1);
		CublasFunc::axpy(handle, size, &minus, targets[s], 1, out, 1);

//		CudaFunc::maskByLength(out, d_inputLengths, s, outputLength, batchSize);
		if (masks != NULL) {
			dtype2* mask = masks[s];
			CudaFunc::maskColVecMat(out, mask, batchSize, layer->size, batchSize);
		}
	} else {
		if (!increment) {
			checkCudaErrors(cudaMemset(out, 0, size * sizeof(dtype2)));
		}
	}
}

void SquaredError::d2_loss(dtype2** targets, Layer *layer, int s,
		dtype2* out, bool increment, dtype2** masks) {
	int size = batchSize * layer->size;
	if (layer == net->getOutputLayer()) {
		if (increment) {
			CudaFunc::iadd(out, size, 1.0);
		} else {
			CudaFunc::fill(out, size, 1.0);
		}

		if (masks != NULL) {
			dtype2* mask = masks[s];
			CudaFunc::maskColVecMat(out, mask, batchSize, layer->size, batchSize);
		}
	} else {
		if (!increment) {
			checkCudaErrors(cudaMemset(out, 0, size * sizeof(dtype2)));
		}
	}
}

CrossEntropy::CrossEntropy(int batchSize, int maxSeqLength, cublasHandle_t& handle)
		: LossFunction(batchSize, maxSeqLength), handle(handle) {
}

CrossEntropy::~CrossEntropy() {
    checkCudaErrors(cudaFree(intermedVec1));
    checkCudaErrors(cudaFree(intermedMat1));
    checkCudaErrors(cudaFree(d_sum));
}

void CrossEntropy::setNetwork(Network* n) {
	LossFunction::setNetwork(n);

	outputLength = net->getOutputLayer()->size;
	int arraySize = batchSize * sizeof(dtype2);
	checkCudaErrors(cudaMalloc((void **)&intermedVec1, arraySize));

	arraySize = batchSize * outputLength * sizeof(dtype2);
	checkCudaErrors(cudaMalloc((void **)&intermedMat1, arraySize));

	checkCudaErrors(cudaMalloc((void **)&d_sum, sizeof(dtype2)));
}

dtypeh CrossEntropy::loss(dtype2** outputs, dtype2** targets, int s, dtype2** masks,
		unsigned int* d_inputLengths) {
	CudaFunc::selectIxByRow(outputs[s], targets[s], intermedVec1, d_inputLengths, s);
	CudaFunc::inegLog(intermedVec1, batchSize);
	CudaFunc::maskByLength(intermedVec1, d_inputLengths, s, 1, batchSize);

	if (masks != NULL) {
		dtype2* mask = masks[s];
		CudaFunc::maskColVecMat(intermedVec1, mask, batchSize, 1, batchSize);
	}

	dtype2 sum;
	checkCudaErrors(cudaMemset((void *)d_sum, 0, sizeof(dtype2)));
	CudaFunc::sum_cols_reduce4(intermedVec1, d_sum, batchSize, 1);
	checkCudaErrors(cudaMemcpy(&sum, d_sum, sizeof(dtype2), cudaMemcpyDeviceToHost));

	return sum;
}

void CrossEntropy::d_loss(dtype2** targets, Layer *layer, int s,
		dtype2* out, bool increment, dtype2** masks,
		unsigned int* d_inputLengths) {
	int size = batchSize * layer->size;

	if (layer->isOutputLayer()) {
		if (!increment) {
			checkCudaErrors(cudaMemset(out, 0, size * sizeof(dtype2)));
		}

		CublasFunc::axpy(handle, size, &one, layer->activation[s], 1, out, 1);
		CudaFunc::addToIxByRow(out, targets[s], minus_one);
		CudaFunc::maskByLength(out, d_inputLengths, s, outputLength, batchSize);

		if (masks != NULL) {
			dtype2* mask = masks[s];
			CudaFunc::maskColVecMat(out, mask, batchSize, layer->size, batchSize);
		}
	} else {
		if (!increment) {
			checkCudaErrors(cudaMemset(out, 0, size * sizeof(dtype2)));
		}
	}
}

void CrossEntropy::d2_loss(dtype2** targets, Layer *layer, int s,
		dtype2* out, bool increment, dtype2** masks) {
	assert(false);
}

bool CrossEntropy::derivCombo() {
	return true;
}

CosineSim::CosineSim(int batchSize, int maxSeqLength, cublasHandle_t& handle)
		: LossFunction(batchSize, maxSeqLength), handle(handle) {
}

CosineSim::~CosineSim() {
    checkCudaErrors(cudaFree(intermedMat1));
    checkCudaErrors(cudaFree(intermedMat2));
    checkCudaErrors(cudaFree(intermedVec1));
    checkCudaErrors(cudaFree(intermedVec2));
    checkCudaErrors(cudaFree(intermedVec3));
}

void CosineSim::setNetwork(Network* n) {
	LossFunction::setNetwork(n);

	outputLength = net->getOutputLayer()->size;
    checkCudaErrors(cudaMalloc((void **)&intermedMat1, batchSize * outputLength * sizeof(dtype2)));
    checkCudaErrors(cudaMalloc((void **)&intermedMat2, batchSize * outputLength * sizeof(dtype2)));
    checkCudaErrors(cudaMalloc((void **)&intermedVec1, batchSize * sizeof(dtype2)));
    checkCudaErrors(cudaMalloc((void **)&intermedVec2, batchSize * sizeof(dtype2)));
    checkCudaErrors(cudaMalloc((void **)&intermedVec3, batchSize * sizeof(dtype2)));
}

dtypeh CosineSim::loss(dtype2** outputs, dtype2** targets, int s, dtype2** masks,
		unsigned int* d_inputLengths) {
	int matSize = batchSize * outputLength;
	int matArraySize = batchSize * outputLength * sizeof(dtype2);
	int vecArraySize = batchSize * sizeof(dtype2);

	unsigned int* h_inputLengths = net->getHInputLengths();

	dtype2* output = outputs[s];
	dtype2* target = targets[s];

	calcBatchErrors(output, target);

	CudaFunc::maskByLength(intermedVec1, d_inputLengths, s, 1, batchSize);
	if (masks != NULL) {
		dtype2* mask = masks[s];
		CudaFunc::maskColVecMat(intermedVec1, mask, batchSize, 1, 1);
	}

	checkCudaErrors(cudaMemset((void *)intermedVec2, 0,  sizeof(dtype2)));
	CudaFunc::sum_cols_reduce4(intermedVec1, intermedVec2, batchSize, 1);
	dtype2 sum;
	checkCudaErrors(cudaMemcpy(&sum, intermedVec2, sizeof(dtype2), cudaMemcpyDeviceToHost));

	return -sum;
}

void CosineSim::calcBatchErrors(dtype2* output, dtype2* target) {
	int matSize = batchSize * outputLength;
	int matArraySize = batchSize * outputLength * sizeof(dtype2);
	int vecArraySize = batchSize * sizeof(dtype2);

	checkCudaErrors(cudaMemset((void *)intermedVec1, 0,  vecArraySize));
	checkCudaErrors(cudaMemset((void *)intermedVec2, 0,  vecArraySize));
	checkCudaErrors(cudaMemset((void *)intermedVec3, 0,  vecArraySize));

	//output . target
	CudaFunc::multiply(output, target, intermedMat1, matSize);
	CudaFunc::sum_rows_reduce4(intermedMat1, intermedVec1, batchSize, outputLength);

	//norm output
	checkCudaErrors(cudaMemcpy(intermedMat1, output, matArraySize, cudaMemcpyDeviceToDevice));
	CudaFunc::isquare(intermedMat1, matSize);
	CudaFunc::sum_rows_reduce4(intermedMat1, intermedVec2, batchSize, outputLength);
	CudaFunc::isqrt(intermedVec2, batchSize);

	//norm target
	checkCudaErrors(cudaMemcpy(intermedMat1, target, matArraySize, cudaMemcpyDeviceToDevice));
	CudaFunc::isquare(intermedMat1, matSize);
	CudaFunc::sum_rows_reduce4(intermedMat1, intermedVec3, batchSize, outputLength);
	CudaFunc::isqrt(intermedVec3, batchSize);

	//(a . b) / (norm a * norm b)
	CudaFunc::multiply(intermedVec2, intermedVec3, intermedVec2, batchSize);
	CudaFunc::imax(intermedVec2, batchSize, EPS);
	CudaFunc::divide(intermedVec1, intermedVec2, intermedVec1, batchSize);
}

void CosineSim::d_loss(dtype2** targets, Layer *layer, int s,
		dtype2* out, bool increment, dtype2** masks,
		unsigned int* d_inputLengths) {
	int matSize = batchSize * outputLength;
	int matArraySize = batchSize * outputLength * sizeof(dtype2);
	int vecArraySize = batchSize * sizeof(dtype2);

	if (layer->isOutputLayer()) {
		if (!increment) {
			checkCudaErrors(cudaMemset(out, 0, matArraySize));
		}

		dtype2* output = layer->activation[s];
		dtype2* target = targets[s];

		calcBatchErrors(output, target);

		checkCudaErrors(cudaMemset((void *)intermedVec2, 0,  vecArraySize));

		//norm(a)^2 (output)
		checkCudaErrors(cudaMemcpy(intermedMat1, output, matArraySize, cudaMemcpyDeviceToDevice));
		CudaFunc::isquare(intermedMat1, matSize);
		CudaFunc::sum_rows_reduce4(intermedMat1, intermedVec2, batchSize, outputLength);

		//errs / norm(a)^2
		checkCudaErrors(cudaMemcpy(intermedVec3, intermedVec2, vecArraySize, cudaMemcpyDeviceToDevice));
		CudaFunc::imax(intermedVec3, batchSize, (EPS)*(EPS));
		CudaFunc::divide(intermedVec1, intermedVec3, intermedVec1, batchSize);

		//norm(a)
		CudaFunc::isqrt(intermedVec2, batchSize);

		//norm(b) target
		checkCudaErrors(cudaMemset((void *)intermedVec3, 0,  vecArraySize));
		checkCudaErrors(cudaMemcpy(intermedMat1, target, matArraySize, cudaMemcpyDeviceToDevice));
		CudaFunc::isquare(intermedMat1, matSize);
		CudaFunc::sum_rows_reduce4(intermedMat1, intermedVec3, batchSize, outputLength);
		CudaFunc::isqrt(intermedVec3, batchSize);

		//b / (norm a * norm b)
		CudaFunc::multiply(intermedVec2, intermedVec3, intermedVec3, batchSize);
		CudaFunc::imax(intermedVec3, batchSize, EPS);
		CudaFunc::iinvert(intermedVec3, batchSize);

		CudaFunc::maskByLength(intermedVec3, d_inputLengths, s, 1, batchSize);
		if (masks != NULL) {
			dtype2* mask = masks[s];
			CudaFunc::maskColVecMat(intermedVec3, mask, batchSize, 1, 1);
		}

		checkCudaErrors(cudaMemcpy(intermedMat1, target, matArraySize, cudaMemcpyDeviceToDevice));
		CudaFunc::multColVecMat(intermedMat1, intermedVec3, batchSize, outputLength, batchSize);

		//a * (errs / sqr(norm a))

		CudaFunc::maskByLength(intermedVec1, d_inputLengths, s, 1, batchSize);
		if (masks != NULL) {
			dtype2* mask = masks[s];
			CudaFunc::maskColVecMat(intermedVec1, mask, batchSize, 1, 1);
		}

		checkCudaErrors(cudaMemcpy(intermedMat2, output, matArraySize, cudaMemcpyDeviceToDevice));
		CudaFunc::multColVecMat(intermedMat2, intermedVec1, batchSize, outputLength, batchSize);

		CublasFunc::axpy(handle, matSize, &minus_one, intermedMat1, 1, intermedMat2, 1);
		CublasFunc::axpy(handle, matSize, &one, intermedMat2, 1, out, 1);
	} else {
		if (!increment) {
			checkCudaErrors(cudaMemset(out, 0, matArraySize));
		}
	}
}

void CosineSim::d2_loss(dtype2** targets, Layer *layer, int s,
		dtype2* out, bool increment, dtype2** masks) {
	assert(false);
}

StructuralDamping::StructuralDamping(int batchSize, int maxSeqLength, dtype2 weight)
		: LossFunction(batchSize, maxSeqLength), weight(weight) {
	damping = 1.0;
}

StructuralDamping::~StructuralDamping() {
}

dtypeh StructuralDamping::loss(dtype2** outputs, dtype2** targets, int s, dtype2** masks,
		unsigned int* d_inputLengths) {
	return 0.0;
}

void StructuralDamping::d_loss(dtype2** targets, Layer *layer,
		int s, dtype2* out, bool increment, dtype2** masks,
		unsigned int* d_inputLengths) {
	if (!increment) {
		checkCudaErrors(cudaMemset(out, 0, batchSize * layer->size * sizeof(dtype2)));
	}
}

void StructuralDamping::d2_loss(dtype2** targets, Layer *layer,
		int s, dtype2* out, bool increment, dtype2** masks) {

	if (layer->structDamp) {
		int size = batchSize * layer->size;
		dtype2 val = weight * damping;
		if (increment) {
			CudaFunc::iadd(out, size, val);
		} else {
			CudaFunc::fill(out, size, val);
		}

		if (masks != NULL) {
			dtype2* mask = masks[s];
			CudaFunc::maskColVecMat(out, mask, batchSize, layer->size, batchSize);
		}
	} else {
		if (!increment) {
			checkCudaErrors(cudaMemset(out, 0, batchSize * layer->size * sizeof(dtype2)));
		}
	}
}

} /* namespace netlib */
