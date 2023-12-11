/**
 * @file
 * @brief Defines CublasFunc class, which has methods that wrap cuBLAS methods.
 *
 * Class CublasFunc has methods that wrap cuBLAS methods, to perform linear algebra operations on GPU. Provides a single
 * method that can be used regardless of currently selected data type.
 *
 */

#include "CublasFunc.h"
#include <iostream>
#include <cassert>
#include <cuda_runtime.h>
#include <helper_cuda.h>

using namespace std;

namespace netlib {

CublasFunc::CublasFunc() {
}

CublasFunc::~CublasFunc() {
}

void CublasFunc::nrm2(cublasHandle_t handle, int n,
		const dtype2* x, int incx, dtype2* result) {
#ifdef DEVICE_DOUBLE
	checkCudaErrors(cublasDnrm2(handle, n, x, incx, result));
#elif defined(DEVICE_SINGLE)
	checkCudaErrors(cublasSnrm2(handle, n, x, incx, result));
#else
	checkCudaErrors(cublasNrm2Ex(handle, n, x, CUDA_R_16F, incx, result, CUDA_R_16F, CUDA_R_32F));
#endif
}

void CublasFunc::dot(cublasHandle_t handle, int n, const dtype2* x, int incx,
		const dtype2* y, int incy, dtypeh* result) {
#ifdef DEVICE_DOUBLE
	checkCudaErrors(cublasDdot(handle, n, x, incx, y, incy, result));
#elif defined(DEVICE_SINGLE)
	checkCudaErrors(cublasSdot(handle, n, x, incx, y, incy, result));
#else
	half result_half;
	checkCudaErrors(cublasDotEx(handle, n, x, CUDA_R_16F, incx, y, CUDA_R_16F, incy, &result_half,
			CUDA_R_16F, CUDA_R_32F));
	*result = __half2float(result_half);
#endif
}

void CublasFunc::gemm(cublasHandle_t handle, cublasOperation_t transa,
		cublasOperation_t transb, int m, int n, int k, const dtypeh* alpha,
		const dtype2* A, int lda, const dtype2* B, int ldb, const dtypeh* beta,
		dtype2* C, int ldc) {
#ifdef DEVICE_DOUBLE
	checkCudaErrors(cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda,
			B, ldb, beta, C, ldc));
#elif defined(DEVICE_SINGLE)
	checkCudaErrors(cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda,
			B, ldb, beta, C, ldc));
#else
	const half alpha_half = __float2half(*alpha);
	const half beta_half = __float2half(*beta);
	checkCudaErrors(cublasGemmEx(handle, transa, transb, m, n, k, &alpha_half, A, CUDA_R_16F, lda,
			B, CUDA_R_16F, ldb, &beta_half, C, CUDA_R_16F, ldc, CUDA_R_16F,
			CUBLAS_GEMM_DEFAULT_TENSOR_OP));
#endif
}

void CublasFunc::gemmUp(cublasHandle_t handle, cublasOperation_t transa,
		cublasOperation_t transb, int m, int n, int k, const dtypeh* alpha,
		const dtype2* A, int lda, const dtype2* B, int ldb, const dtypeh* beta,
		dtypeup* C, int ldc) {
#ifdef DEVICE_DOUBLE
	checkCudaErrors(cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda,
			B, ldb, beta, C, ldc));
#elif defined(DEVICE_SINGLE)
	checkCudaErrors(cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda,
			B, ldb, beta, C, ldc));
#else
	const half alpha_half = __float2half(*alpha);
	const half beta_half = __float2half(*beta);
	checkCudaErrors(cublasGemmEx(handle, transa, transb, m, n, k, &alpha_half, A, CUDA_R_16F, lda,
			B, CUDA_R_16F, ldb, &beta_half, C, CUDA_R_32F, ldc, CUDA_R_32F,
			CUBLAS_GEMM_DEFAULT_TENSOR_OP));
#endif
}

void CublasFunc::scal(cublasHandle_t handle, int n, const dtypeh* alpha,
		dtype2* x, int incx) {
#ifdef DEVICE_DOUBLE
	checkCudaErrors(cublasDscal(handle, n, alpha, x, incx));
#elif defined(DEVICE_SINGLE)
	checkCudaErrors(cublasSscal(handle, n, alpha, x, incx));
#else
	checkCudaErrors(cublasScalEx(handle, n, alpha, CUDA_R_32F, x, CUDA_R_16F, incx, CUDA_R_32F));
#endif
}

void CublasFunc::scalA(cublasHandle_t handle, int n, const dtypeh* alpha,
		dtypea* x, int incx) {
#ifdef DEVICE_DOUBLE
	checkCudaErrors(cublasDscal(handle, n, alpha, x, incx));
#elif defined(DEVICEA_SINGLE)
	checkCudaErrors(cublasSscal(handle, n, alpha, x, incx));
#else
	checkCudaErrors(cublasScalEx(handle, n, alpha, CUDA_R_32F, x, CUDA_R_16F, incx, CUDA_R_32F));
#endif
}

//void CublasFunc::asum(cublasHandle_t handle, int n, const dtype2* x, int incx,
//		dtype2* result) {
//#ifdef DEBUG
//	checkCudaErrors(cublasDasum(handle, n, x, incx, result));
//#else
//	checkCudaErrors(cublasSasum(handle, n, x, incx, result));
//#endif
//}

void CublasFunc::axpy(cublasHandle_t handle, int n, const dtypeh* alpha,
		const dtype2* x, int incx, dtype2* y, int incy) {
#ifdef DEVICE_DOUBLE
	checkCudaErrors(cublasDaxpy(handle, n, alpha, x, incx, y, incy));
#elif defined(DEVICE_SINGLE)
	checkCudaErrors(cublasSaxpy(handle, n, alpha, x, incx, y, incy));
#else
	checkCudaErrors(cublasAxpyEx(handle, n, alpha, CUDA_R_32F, x, CUDA_R_16F,
			incx, y, CUDA_R_16F, incy, CUDA_R_32F));
#endif
}

void CublasFunc::copy(cublasHandle_t handle, int n, const dtype2* x, int incx,
		dtype2* y, int incy) {
#ifdef DEVICE_DOUBLE
	checkCudaErrors(cublasDcopy(handle, n, x, incx, y, incy));
#elif defined(DEVICE_SINGLE)
	checkCudaErrors(cublasScopy(handle, n, x, incx, y, incy));
#else
	assert(incy == 1);
	unsigned int arraySize = n * sizeof(dtype2);
	checkCudaErrors(cudaMemset((void *)y, 0, arraySize));
	float alpha = 1.0f;
	checkCudaErrors(cublasAxpyEx(handle, n, &alpha, CUDA_R_32F, x, CUDA_R_16F,
			incx, y, CUDA_R_16F, incy, CUDA_R_32F));
#endif
}

void CublasFunc::dgmm(cublasHandle_t handle, cublasSideMode_t mode, int m,
		int n, const dtype2* A, int lda, const dtype2* x, int incx, dtype2* C,
		int ldc) {
#ifdef DEVICE_DOUBLE
	checkCudaErrors(cublasDdgmm(handle, mode, m, n, A, lda, x, incx, C, ldc));
#elif defined(DEVICE_SINGLE)
	checkCudaErrors(cublasSdgmm(handle, mode, m, n, A, lda, x, incx, C, ldc));
#else
	cerr << "dgmm not implemented for half precision" << endl;
	exit(-1);
#endif
}

void CublasFunc::gemv(cublasHandle_t handle, cublasOperation_t transa, int m,
		int n, const dtypeh* alpha, const dtype2* A, int lda, const dtype2* x,
		int incx, const dtypeh* beta, dtype2* y, int incy) {
#ifdef DEVICE_DOUBLE
	checkCudaErrors(cublasDgemv(handle, transa, m, n, alpha, A, lda, x, incx,
			beta, y, incy));
#elif defined(DEVICE_SINGLE)
	checkCudaErrors(cublasSgemv(handle, transa, m, n, alpha, A, lda, x, incx,
			beta, y, incy));
#else
	const half alpha_half = __float2half(*alpha);
	const half beta_half = __float2half(*beta);
	checkCudaErrors(cublasGemmEx(handle, transa, CUBLAS_OP_N, m, 1, n, &alpha_half, A, CUDA_R_16F, lda,
			x, CUDA_R_16F, n, &beta_half, y, CUDA_R_16F, m, CUDA_R_16F,
			CUBLAS_GEMM_DEFAULT_TENSOR_OP));
#endif
}

void CublasFunc::gemvUp(cublasHandle_t handle, cublasOperation_t transa, int m,
		int n, const dtypeh* alpha, const dtype2* A, int lda, const dtype2* x,
		int incx, const dtypeh* beta, dtypeup* y, int incy) {
#ifdef DEVICE_DOUBLE
	checkCudaErrors(cublasDgemv(handle, transa, m, n, alpha, A, lda, x, incx,
			beta, y, incy));
#elif defined(DEVICE_SINGLE)
	checkCudaErrors(cublasSgemv(handle, transa, m, n, alpha, A, lda, x, incx,
			beta, y, incy));
#else
	const half alpha_half = __float2half(*alpha);
	const half beta_half = __float2half(*beta);
	checkCudaErrors(cublasGemmEx(handle, transa, CUBLAS_OP_N, m, 1, n, &alpha_half, A, CUDA_R_16F, lda,
			x, CUDA_R_16F, n, &beta_half, y, CUDA_R_32F, m, CUDA_R_32F,
			CUBLAS_GEMM_DEFAULT_TENSOR_OP));
#endif
}

//void CublasFunc::geam(cublasHandle_t handle,
//		    cublasOperation_t transa, cublasOperation_t transb,
//		    int m, int n,
//		    const dtype2 *alpha,
//		    const dtype2 *A, int lda,
//		    const dtype2 *beta, const dtype2 *B, int ldb,
//		    dtype2 *C, int ldc) {
//#ifdef DEBUG
//	checkCudaErrors(cublasDgeam(handle, transa, transb, m, n, alpha, A, lda,
//			beta, B, ldb, C, ldc));
//#else
//	checkCudaErrors(cublasSgeam(handle, transa, transb, m, n, alpha, A, lda,
//			beta, B, ldb, C, ldc));
//#endif
//}

void CublasFunc::iamax(cublasHandle_t handle, int n, const dtype2 *x,
		int incx, int *result) {
#ifdef DEVICE_DOUBLE
	checkCudaErrors(cublasIdamax(handle, n, x, incx, result));
#elif defined(DEVICE_SINGLE)
	checkCudaErrors(cublasIsamax(handle, n, x, incx, result));
#else
	cerr << "iamax not implemented for half precision" << endl;
	exit(-1);
#endif
}

void CublasFunc::iamaxUp(cublasHandle_t handle, int n, const dtypeup *x,
		int incx, int *result) {
#ifdef DEVICE_DOUBLE
	checkCudaErrors(cublasIdamax(handle, n, x, incx, result));
#elif defined(DEVICE_SINGLE)
	checkCudaErrors(cublasIsamax(handle, n, x, incx, result));
#else
	checkCudaErrors(cublasIsamax(handle, n, x, incx, result));
#endif
}

void CublasFunc::iamin(cublasHandle_t handle, int n, const dtype2 *x,
		int incx, int *result) {
#ifdef DEVICE_DOUBLE
	checkCudaErrors(cublasIdamin(handle, n, x, incx, result));
#elif defined(DEVICE_SINGLE)
	checkCudaErrors(cublasIsamin(handle, n, x, incx, result));
#else
	cerr << "iamin not implemented for half precision" << endl;
	exit(-1);
#endif
}

void CublasFunc::getPointerMode(cublasHandle_t handle, cublasPointerMode_t *mode) {
	checkCudaErrors(cublasGetPointerMode_v2(handle, mode));
}

void CublasFunc::setPointerMode(cublasHandle_t handle, cublasPointerMode_t mode) {
	checkCudaErrors(cublasSetPointerMode_v2(handle, mode));
}

//void CublasFunc::setStream(cublasHandle_t handle, cudaStream_t streamId) {
//	checkCudaErrors(cublasSetStream(handle, streamId));
//}

} /* namespace netlib */
