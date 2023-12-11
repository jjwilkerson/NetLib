/**
 * @file
 * @brief Declares CublasFunc class, which has methods that wrap cuBLAS methods.
 *
 * Class CublasFunc has methods that wrap cuBLAS methods, to perform linear algebra operations on GPU. Provides a single
 * method that can be used regardless of currently selected data type.
 *
 */

#ifndef CUBLASFUNC_H_
#define CUBLASFUNC_H_

#include "../NetLib.h"
//#include <cuda_runtime.h>
#include "cublas_v2.h"

namespace netlib {

/**
 * @brief Has methods that wrap cuBLAS methods.
 *
 * Has methods that wrap cuBLAS methods, to perform linear algebra operations on GPU. Provides a single
 * method that can be used regardless of currently selected data type.
 *
 */
class CublasFunc {
public:
	CublasFunc();
	virtual ~CublasFunc();
	static void nrm2(cublasHandle_t handle, int n, const dtype2 *x, int incx, dtype2 *result);
	static void dot(cublasHandle_t handle, int n, const dtype2 *x, int incx, const dtype2 *y,
			int incy, dtypeh *result);
	static void gemm(cublasHandle_t handle,
		    cublasOperation_t transa, cublasOperation_t transb,
		    int m, int n, int k,
		    const dtypeh *alpha,
		    const dtype2 *A, int lda,
		    const dtype2 *B, int ldb,
		    const dtypeh *beta, dtype2 *C, int ldc);
	static void gemmUp(cublasHandle_t handle,
		    cublasOperation_t transa, cublasOperation_t transb,
		    int m, int n, int k,
		    const dtypeh *alpha,
		    const dtype2 *A, int lda,
		    const dtype2 *B, int ldb,
		    const dtypeh *beta, dtypeup *C, int ldc);
	static void scal(cublasHandle_t handle, int n, const dtypeh *alpha, dtype2 *x, int incx);
	static void scalA(cublasHandle_t handle, int n, const dtypeh *alpha, dtypea *x, int incx);
//	static void asum(cublasHandle_t handle, int n, const dtype2 *x, int incx, dtype2 *result);
	static void axpy(cublasHandle_t handle, int n, const dtypeh *alpha, const dtype2 *x, int incx,
			dtype2 *y, int incy);
	static void copy(cublasHandle_t handle, int n, const dtype2 *x, int incx, dtype2 *y, int incy);
	static void dgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n, const dtype2 *A,
			int lda, const dtype2 *x, int incx, dtype2 *C, int ldc);
	static void gemv(cublasHandle_t handle,
		    cublasOperation_t transa,
		    int m, int n,
		    const dtypeh *alpha,
		    const dtype2 *A, int lda,
		    const dtype2 *x, int incx,
		    const dtypeh *beta, dtype2 *y, int incy);
	static void gemvUp(cublasHandle_t handle,
		    cublasOperation_t transa,
		    int m, int n,
		    const dtypeh *alpha,
		    const dtype2 *A, int lda,
		    const dtype2 *x, int incx,
		    const dtypeh *beta, dtypeup *y, int incy);
//	static void geam(cublasHandle_t handle,
//		    cublasOperation_t transa, cublasOperation_t transb,
//		    int m, int n,
//		    const dtype2 *alpha,
//		    const dtype2 *A, int lda,
//		    const dtype2 *beta, const dtype2 *B, int ldb,
//		    dtype2 *C, int ldc);
	static void iamax(cublasHandle_t handle, int n,
			const dtype2 *x, int incx, int *result);
	static void iamaxUp(cublasHandle_t handle, int n,
			const dtypeup *x, int incx, int *result);
	static void iamin(cublasHandle_t handle, int n,
			const dtype2 *x, int incx, int *result);
	static void getPointerMode(cublasHandle_t handle, cublasPointerMode_t *mode);
	static void setPointerMode(cublasHandle_t handle, cublasPointerMode_t mode);
	static void setStream(cublasHandle_t handle, cudaStream_t streamId);
};

} /* namespace netlib */

#endif /* CUBLASFUNC_H_ */
