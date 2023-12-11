/**
 * @file
 * @brief Defines CudaFunc class, which contains CUDA functions for execution on GPU.
 *
 */

#include "CudaFunc.h"
//#include "kernels.h"
#include <helper_cuda.h>
#include <curand_kernel.h>

#define IDX2(i,j,ld) (((j)*(ld))+(i))
#define WV_LENGTH 300

#ifdef DEBUG
#define BLOCK_HEIGHT 8
#define CURAND_GRID_DIM 8
#else
#define BLOCK_HEIGHT 32
#define CURAND_GRID_DIM 32
#endif

#ifdef DEVICE_DOUBLE
#define dpow(b,e) pow(b,e)
#define dtanh(a) tanh(a)
#define dsqrt(a) sqrt(a)
#define dexp(a) exp(a)
#define dlog(a) log(a)
#define d2int_rn(a) __double2int_rn(a)
#define d2int_rd(a) __double2int_rd(a)
#define d2int_ru(a) __double2int_ru(a)
#define dadd(a,b) a+b
#define dsub(a,b) a-b
#define dmul(a,b) a*b
#define ddiv(a,b) (a/b)
#define dneg(a) (-(a))
#define deq(a,b) (a == b)
#define dge(a,b) (a >= b)
#define dle(a,b) (a <= b)
#define dgt(a,b) (a > b)
#define dlt(a,b) (a < b)
#define dupmul(a,b) (a*b)
#define dupadd(a,b) (a+b)
#define dupdiv(a,b) (a/b)
#define disnan(a) isnan(a)
#define disinf(a) isinf(a)
#define dmax(a,b) max(a,b)
#define d2float(a) (a)
#define float2d(a) (a)
#define d1add(a,b) (a+b)
#define d1mul(a,b) (a*b)
#elif defined(DEVICE_SINGLE)
#define dpow(b,e) powf(b,e)
#define dtanh(a) tanhf(a)
#define dsqrt(a) sqrtf(a)
#define dexp(a) expf(a)
#define dlog(a) logf(a)
#define d2int_rn(a) __float2int_rn(a)
#define d2int_rd(a) __float2int_rd(a)
#define d2int_ru(a) __float2int_ru(a)
#define dadd(a,b) (a+b)
#define dsub(a,b) (a-b)
#define dmul(a,b) (a*b)
#define ddiv(a,b) (a/b)
#define dneg(a) (-(a))
#define deq(a,b) (a == b)
#define dge(a,b) (a >= b)
#define dle(a,b) (a <= b)
#define dgt(a,b) (a > b)
#define dlt(a,b) (a < b)
#define dupmul(a,b) (a*b)
#define dupadd(a,b) (a+b)
#define dupdiv(a,b) (a/b)
#define disnan(a) isnan(a)
#define disinf(a) isinf(a)
#define dmax(a,b) max(a,b)
#define d2float(a) (a)
#define float2d(a) (a)
#define d1add(a,b) (a+b)
#define d1mul(a,b) (a*b)
#else //DEVICE_HALF
#define dpow(b,e) __float2half(powf(b,e))
#define dtanh(a) __float2half(tanhf(a))
#define dsqrt(a) hsqrt(a)
#define dexp(a) hexp(a)
#define dlog(a) hlog(a)
#define d2int_rn(a) __half2int_rn(a)
#define d2int_rd(a) __half2int_rd(a)
#define d2int_ru(a) __half2int_ru(a)
#define dadd(a,b) __hadd(a,b)
#define dsub(a,b) __hsub(a,b)
#define dmul(a,b) __hmul(a,b)
#define ddiv(a,b) __hdiv(a,b)
#define dneg(a) __hneg(a)
#define deq(a,b) __heq(a,b)
#define dge(a,b) __hge(a,b)
#define dle(a,b) __hle(a,b)
#define dgt(a,b) __hgt(a,b)
#define dlt(a,b) __hlt(a,b)
#define dupmul(a,b) (a*b)
#define dupadd(a,b) (a+b)
#define dupdiv(a,b) (a/b)
#define disnan(a) __hisnan(a)
#define disinf(a) __hisinf(a)
#define dmax(a,b) (__hge(a,b)?(a):(b))
#define d2float(a) __half2float(a)
#define float2d(a) __float2half(a)
#ifdef DEVICE1_SINGLE
#define d1add(a,b) (a+b)
#define d1mul(a,b) (a*b)
#else
#define d1add(a,b) __hadd(a,b)
#define d1mul(a,b) __hmul(a,b)
#endif
#endif
#ifdef DEVICEA_SINGLE
#define dasqrt(a) sqrtf(a)
#define daadd(a,b) (a+b)
#define damul(a,b) (a*b)
#define dadiv(a,b) (a/b)
#else
#define dasqrt(a) hsqrt(a)
#define daadd(a,b) __hadd(a,b)
#define damul(a,b) __hmul(a,b)
#define dadiv(a,b) __hdiv(a,b)
#endif

//typedef dtype2 (*op_func) (dtype2, dtype2);

namespace netlib {

float CudaFunc::pDropMatch = 1.0f;
int CudaFunc::batchSize = 128;
int CudaFunc::maxSentenceLength = 40;

size_t curandStatesArraySize;
__device__ __managed__ curandState *curandStates = NULL;


CudaFunc::CudaFunc() {
}

CudaFunc::~CudaFunc() {
}

template <typename T>
__global__ void mask_length(T* mat, unsigned int* lengths, const int s, const int width,
		const int height) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= height) {
		return;
	}
	if (s >= lengths[i]) {
		mat[IDX2(i,blockIdx.y,height)] = 0;
	}
}

void CudaFunc::maskByLength(dtype2* mat, unsigned int* lengths, int s, int layerSize,
		int batchSize) {
	dim3 threadsPerBlock(BLOCK_HEIGHT, 1);
	dim3 numBlocks(ceil((float) batchSize / threadsPerBlock.x), layerSize);
	mask_length<dtype2><<<numBlocks, threadsPerBlock>>>(mat, lengths, s,
			layerSize, batchSize);
	CudaFunc::checkErrors();
}

template <typename T>
__global__ void mult_col_vec_mat(T* mat, T* vec, const int rows, const int cols, int ldm) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= rows) {
		return;
	}
	int j = blockIdx.y;
	mat[IDX2(i,j,ldm)] = dmul(mat[IDX2(i,j,ldm)], vec[i]);
}

void CudaFunc::multColVecMat(dtype2* mat, dtype2* vec, int rows, int cols, int ldm) {
	dim3 threadsPerBlock(BLOCK_HEIGHT, 1);
	dim3 numBlocks(ceil((float) rows / threadsPerBlock.x), cols);
	mult_col_vec_mat<dtype2><<<numBlocks, threadsPerBlock>>>(mat, vec, rows, cols, ldm);
	CudaFunc::checkErrors();
}

template <typename T, typename U>
__global__ void mult_col_vec_mat_up(U* mat, T* vec, const int rows, const int cols, int ldm) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= rows) {
		return;
	}
	int j = blockIdx.y;
	mat[IDX2(i,j,ldm)] = mat[IDX2(i,j,ldm)] * d2up(vec[i]);
}

void CudaFunc::multColVecMatUp(dtypeup* mat, dtype2* vec, int rows, int cols, int ldm) {
	dim3 threadsPerBlock(BLOCK_HEIGHT, 1);
	dim3 numBlocks(ceil((float) rows / threadsPerBlock.x), cols);
	mult_col_vec_mat_up<dtype2, dtypeup><<<numBlocks, threadsPerBlock>>>(mat, vec, rows, cols, ldm);
	CudaFunc::checkErrors();
}

template <typename T>
__global__ void mult_row_vec_mat(T* mat, T* vec, const int rows, const int cols, int ldm) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= rows) {
		return;
	}
	int j = blockIdx.y;
	mat[IDX2(i,j,ldm)] = dmul(mat[IDX2(i,j,ldm)], vec[j]);
}

void CudaFunc::multRowVecMat(dtype2* mat, dtype2* vec, int rows, int cols, int ldm) {
	dim3 threadsPerBlock(BLOCK_HEIGHT, 1);
	dim3 numBlocks(ceil((float) rows / threadsPerBlock.x), cols);
	mult_row_vec_mat<dtype2><<<numBlocks, threadsPerBlock>>>(mat, vec, rows, cols, ldm);
	CudaFunc::checkErrors();
}

template <typename T>
__global__ void div_col_vec_mat(T* mat, T* vec, const int rows, const int cols, int ldm) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= rows) {
		return;
	}
	int j = blockIdx.y;
	mat[IDX2(i,j,ldm)] = ddiv(mat[IDX2(i,j,ldm)], vec[i]);
}

void CudaFunc::divColVecMat(dtype2* mat, dtype2* vec, int rows, int cols, int ldm) {
	dim3 threadsPerBlock(BLOCK_HEIGHT, 1);
	dim3 numBlocks(ceil((float) rows / threadsPerBlock.x), cols);
	div_col_vec_mat<dtype2><<<numBlocks, threadsPerBlock>>>(mat, vec, rows, cols, ldm);
	CudaFunc::checkErrors();
}

template <typename T>
__global__ void div_row_vec_mat(T* mat, T* vec, const int rows, const int cols, int ldm) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= rows) {
		return;
	}
	int j = blockIdx.y;
	mat[IDX2(i,j,ldm)] /= vec[j];
}

void CudaFunc::divRowVecMat(dtype2* mat, dtype2* vec, int rows, int cols, int ldm) {
	dim3 threadsPerBlock(BLOCK_HEIGHT, 1);
	dim3 numBlocks(ceil((float) rows / threadsPerBlock.x), cols);
	mult_row_vec_mat<dtype2><<<numBlocks, threadsPerBlock>>>(mat, vec, rows, cols, ldm);
	CudaFunc::checkErrors();
}

template <typename T>
__global__ void fill_(T* arr, const int size, const T val) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		arr[i] = val;
	}
}

void CudaFunc::fill(dtype2* arr, const int size, const dtypeh val) {
	int numBlocks = ceil((float) size / BLOCK_HEIGHT);
	fill_<dtype2><<<numBlocks, BLOCK_HEIGHT>>>(arr, size, float2d(val));
	CudaFunc::checkErrors();
}

template <typename T>
__global__ void fill_1(T* arr, const int size, const T val) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		arr[i] = val;
	}
}

void CudaFunc::fill1(dtype1* arr, const int size, const dtypeh val) {
	int numBlocks = ceil((float) size / BLOCK_HEIGHT);
	fill_1<dtype1><<<numBlocks, BLOCK_HEIGHT>>>(arr, size, float21(val));
	CudaFunc::checkErrors();
}

template <typename T>
__global__ void iclip_(T* arr, const int size, const T maxVal) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		if (dgt(arr[i], maxVal)) {
			arr[i] = maxVal;
		} else if (dlt(arr[i], dneg(maxVal))) {
			arr[i] = dneg(maxVal);
		}
	}
}

void CudaFunc::iclip(dtype2* arr, const int size, const dtypeh maxVal) {
	int numBlocks = ceil((float) size / BLOCK_HEIGHT);
	iclip_<dtype2><<<numBlocks, BLOCK_HEIGHT>>>(arr, size, float2d(maxVal));
	CudaFunc::checkErrors();
}

template <typename T>
__global__ void iadd_(T* arr, const int size, const T val) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		arr[i] = dadd(arr[i], val);
	}
}

template <typename T>
__global__ void iadd_a(T* arr, const int size, const T val) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		arr[i] = daadd(arr[i], val);
	}
}

void CudaFunc::iadd(dtype2* arr, const int size, const dtypeh val) {
	int numBlocks = ceil((float) size / BLOCK_HEIGHT);
	iadd_<dtype2><<<numBlocks, BLOCK_HEIGHT>>>(arr, size, float2d(val));
	CudaFunc::checkErrors();
}

void CudaFunc::iaddA(dtypea* arr, const int size, const dtypeh val) {
	int numBlocks = ceil((float) size / BLOCK_HEIGHT);
	iadd_a<dtypea><<<numBlocks, BLOCK_HEIGHT>>>(arr, size, float2d(val));
	CudaFunc::checkErrors();
}

template <typename T>
__global__ void isquare_(T* arr, const int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	const T val = arr[i];
	if (i < size) {
		dtypeup valup = d2up(val);
		arr[i] = dup22(dupmul(valup, valup));
	}
}
void CudaFunc::isquare(dtype2* arr, const int size) {
	int numBlocks = ceil((float) size / BLOCK_HEIGHT);
	isquare_<dtype2><<<numBlocks, BLOCK_HEIGHT>>>(arr, size);
	CudaFunc::checkErrors();
}

template <typename T>
__global__ void isqrt_(T* arr, const int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		arr[i] = dsqrt(arr[i]);
	}
}

template <typename T>
__global__ void isqrt_a(T* arr, const int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		arr[i] = dasqrt(arr[i]);
	}
}

void CudaFunc::isqrt(dtype2* arr, const int size) {
	int numBlocks = ceil((float) size / BLOCK_HEIGHT);
	isqrt_<dtype2><<<numBlocks, BLOCK_HEIGHT>>>(arr, size);
	CudaFunc::checkErrors();
}

void CudaFunc::isqrtA(dtypea* arr, const int size) {
	int numBlocks = ceil((float) size / BLOCK_HEIGHT);
	isqrt_a<dtypea><<<numBlocks, BLOCK_HEIGHT>>>(arr, size);
	CudaFunc::checkErrors();
}

template <typename T>
__global__ void iinvert_(T* arr, const int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		arr[i] = dup22(dupdiv(d2up(1.0), d2up(arr[i])));
	}
}

void CudaFunc::iinvert(dtype2* arr, const int size) {
	int numBlocks = ceil((float) size / BLOCK_HEIGHT);
	iinvert_<dtype2><<<numBlocks, BLOCK_HEIGHT>>>(arr, size);
	CudaFunc::checkErrors();
}

template <typename T>
__global__ void iexp_(T* arr, const int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		arr[i] = dexp(arr[i]);
	}
}

void CudaFunc::iexp(dtype2* arr, const int size) {
	int numBlocks = ceil((float) size / BLOCK_HEIGHT);
	iexp_<dtype2><<<numBlocks, BLOCK_HEIGHT>>>(arr, size);
	CudaFunc::checkErrors();
}

template <typename T>
__global__ void ineg_log(T* arr, const int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		arr[i] = -dlog(arr[i]);
	}
}

void CudaFunc::inegLog(dtype2* arr, const int size) {
	int numBlocks = ceil((float) size / BLOCK_HEIGHT);
	ineg_log<dtype2><<<numBlocks, BLOCK_HEIGHT>>>(arr, size);
	CudaFunc::checkErrors();
}

template <typename T>
__global__ void ineg_log_1minus(T* arr, const int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		arr[i] = -dlog(dsub((T) 1.0f, arr[i]));
	}
}

void CudaFunc::inegLog1minus(dtype2* arr, const int size) {
	int numBlocks = ceil((float) size / BLOCK_HEIGHT);
	ineg_log_1minus<dtype2><<<numBlocks, BLOCK_HEIGHT>>>(arr, size);
	CudaFunc::checkErrors();
}

template <typename T>
__global__ void count_(T* arr, const int size, const T val, int* result) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		if(deq(arr[i], val)) {
			atomicAdd(result, 1);
		}
	}
}

void CudaFunc::count(dtype2* arr, const int size, const dtypeh val, int* result) {
	int* d_result;
	checkCudaErrors(cudaMalloc((void **)&d_result, sizeof(int)));
	checkCudaErrors(cudaMemset((void *)d_result, 0, sizeof(int)));

	int numBlocks = ceil((float) size / BLOCK_HEIGHT);
	count_<dtype2><<<numBlocks, BLOCK_HEIGHT>>>(arr, size, float2d(val), d_result);

	checkCudaErrors(cudaMemcpy(result, d_result, sizeof(int), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaFree(d_result));
	CudaFunc::checkErrors();
}

template <typename T>
__global__ void imax_(T* arr, const int size, const T val) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		arr[i] = dmax(arr[i], val);
	}
}

void CudaFunc::imax(dtype2* arr, const int size, const dtypeh val) {
	int numBlocks = ceil((float) size / BLOCK_HEIGHT);
	imax_<dtype2><<<numBlocks, BLOCK_HEIGHT>>>(arr, size, float2d(val));
	CudaFunc::checkErrors();
}

//template <typename T>
//__global__ void scale_and_check(T* arr, const int size, const T val, bool* overflow) {
//	int i = blockIdx.x * blockDim.x + threadIdx.x;
//	if (i < size) {
//		T a = arr[i];
//		if (disnan(a) || disinf(a)) {
//			*overflow = true;
//		}
//		arr[i] = dmul(a, val);
//		arr[i] = __float2half(__half2float(arr[i]) * __half2float(val));
//	}
//}
//
//void CudaFunc::scaleAndCheck(dtype2* arr, const int size, const dtypeh val, bool* overflow) {
//	bool* d_overflow = NULL;
//	checkCudaErrors(cudaMalloc((void **)&d_overflow, sizeof(bool)));
//	checkCudaErrors(cudaMemset((void *)d_overflow, 0, sizeof(bool)));
//
//	int numBlocks = ceil((float) size / BLOCK_HEIGHT);
//	scale_and_check<dtype2><<<numBlocks, BLOCK_HEIGHT>>>(arr, size, float2d(val), d_overflow);
//
//	checkCudaErrors(cudaMemcpy(overflow, d_overflow, sizeof(bool), cudaMemcpyDeviceToHost));
//	checkCudaErrors(cudaFree(d_overflow));
//	*overflow = false;
//}

template <typename T>
__global__ void check_for_overflow(T* arr, const int size, bool* overflow) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		T a = arr[i];
		if (disnan(a) || disinf(a)) {
			*overflow = true;
		}
	}
}

void CudaFunc::checkForOverflow(dtype2* arr, const int size, bool* overflow) {
	bool* d_overflow = NULL;
	checkCudaErrors(cudaMalloc((void **)&d_overflow, sizeof(bool)));
	checkCudaErrors(cudaMemset((void *)d_overflow, 0, sizeof(bool)));

	int numBlocks = ceil((float) size / BLOCK_HEIGHT);
	check_for_overflow<dtype2><<<numBlocks, BLOCK_HEIGHT>>>(arr, size, d_overflow);

	checkCudaErrors(cudaMemcpy(overflow, d_overflow, sizeof(bool), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaFree(d_overflow));
}

template <typename T>
__global__ void clear_ix_by_row(T* activation, T* ix, int batchSize,
		unsigned int* lengths, const int s) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int length = lengths[i];

	if (s >= length) {
		return;
	} else if (i < batchSize) {
		int curr = d2int_rd(dadd(ix[i], (T) 0.5));
		activation[IDX2(i,curr,batchSize)] = 0.0f;
	}
}

void CudaFunc::clearIxByRow(dtype2* activation, dtype2* ix,
		unsigned int* inputLengths, int s) {
	int numBlocks = ceil((float) batchSize / BLOCK_HEIGHT);
	clear_ix_by_row<dtype2><<<numBlocks, BLOCK_HEIGHT>>>(activation, ix, batchSize,
			inputLengths, s);
	CudaFunc::checkErrors();
}

template <typename T>
__global__ void select_ix_by_row(T* activation, T* ix, T* out, int batchSize,
		unsigned int* lengths, const int s) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int length = lengths[i];

	if (s >= length) {
		return;
	} else if (i < batchSize) {
		int curr = d2int_rd(dadd(ix[i], (T) 0.5));
		out[i] = activation[IDX2(i,curr,batchSize)];
	}
}

void CudaFunc::selectIxByRow(dtype2* activation, dtype2* ix, dtype2* out,
		unsigned int* inputLengths, int s) {
	int numBlocks = ceil((float) batchSize / BLOCK_HEIGHT);
	select_ix_by_row<dtype2><<<numBlocks, BLOCK_HEIGHT>>>(activation, ix, out,
			batchSize, inputLengths, s);
	CudaFunc::checkErrors();
}

template <typename T>
__global__ void add_to_ix_by_row(T* activation, T* ix, const T val, int batchSize) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < batchSize) {
		int currCol = d2int_rd(dadd(ix[i], (T) 0.5));
		int curr = IDX2(i,currCol,batchSize);
		activation[curr] = dadd(activation[curr], val);
	}
}

void CudaFunc::addToIxByRow(dtype2* activation, dtype2* ix, const dtypeh val) {
	int numBlocks = ceil((float) batchSize / BLOCK_HEIGHT);
	add_to_ix_by_row<dtype2><<<numBlocks, BLOCK_HEIGHT>>>(activation, ix, float2d(val), batchSize);
	CudaFunc::checkErrors();
}

template <typename T>
__global__ void dup_in(T* dest, T* src, unsigned int* lengths, const int s, const int width,
		const int height) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= height) {
		return;
	}
	if (s == lengths[i]-1) {
		unsigned idx = IDX2(i,blockIdx.y,height);
		dest[idx] = src[idx];
	}
}

void CudaFunc::dupIn(dtype2* dest, dtype2* src, unsigned int* lengths, int s,
		int layerSize, int batchSize) {
	dim3 threadsPerBlock(BLOCK_HEIGHT, 1);
	dim3 numBlocks(ceil((float) batchSize / threadsPerBlock.x), layerSize);
	dup_in<dtype2><<<numBlocks, threadsPerBlock>>>(dest, src, lengths, s,
			layerSize, batchSize);
	CudaFunc::checkErrors();
}

template <typename T>
__global__ void add_column_vec_mat(T* mat, T* vec, const int rows, const int cols, int ldm) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= rows) {
		return;
	}
	int j = blockIdx.y;
	mat[IDX2(i,j,ldm)] = dadd(mat[IDX2(i,j,ldm)], vec[i]);
}

void CudaFunc::addColumnVecMat(dtype2* mat, dtype2* vec, int rows, int cols, int ldm) {
	dim3 threadsPerBlock(BLOCK_HEIGHT, 1);
	dim3 numBlocks(ceil((float) rows / threadsPerBlock.x), cols);
	add_column_vec_mat<dtype2><<<numBlocks, threadsPerBlock>>>(mat, vec, rows, cols, ldm);
	CudaFunc::checkErrors();
}

template <typename T>
__global__ void subtract_column_vec_mat(T* mat, T* vec, const int rows, const int cols, int ldm) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= rows) {
		return;
	}
	int j = blockIdx.y;
	mat[IDX2(i,j,ldm)] = dsub(mat[IDX2(i,j,ldm)], vec[i]);
}

void CudaFunc::subtractColumnVecMat(dtype2* mat, dtype2* vec, int rows, int cols, int ldm) {
	dim3 threadsPerBlock(BLOCK_HEIGHT, 1);
	dim3 numBlocks(ceil((float) rows / threadsPerBlock.x), cols);
	subtract_column_vec_mat<dtype2><<<numBlocks, threadsPerBlock>>>(mat, vec, rows, cols, ldm);
	CudaFunc::checkErrors();
}

template <typename T>
__global__ void add_row_vec_mat(T* mat, T* vec, const int rows, const int cols, int ldm) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= rows) {
		return;
	}
	int j = blockIdx.y;
	mat[IDX2(i,j,ldm)] = dadd(mat[IDX2(i,j,ldm)], vec[j]);
}

void CudaFunc::addRowVecMat(dtype2* mat, dtype2* vec, int rows, int cols, int ldm) {
	dim3 threadsPerBlock(BLOCK_HEIGHT, 1);
	dim3 numBlocks(ceil((float) rows / threadsPerBlock.x), cols);
	add_row_vec_mat<dtype2><<<numBlocks, threadsPerBlock>>>(mat, vec, rows, cols, ldm);
	CudaFunc::checkErrors();
}

template <typename T>
__global__ void subtract_row_vec_mat(T* mat, T* vec, const int rows, const int cols, int ldm) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= rows) {
		return;
	}
	int j = blockIdx.y;
	mat[IDX2(i,j,ldm)] = dsub(mat[IDX2(i,j,ldm)], vec[j]);
}

void CudaFunc::subtractRowVecMat(dtype2* mat, dtype2* vec, int rows, int cols, int ldm) {
	dim3 threadsPerBlock(BLOCK_HEIGHT, 1);
	dim3 numBlocks(ceil((float) rows / threadsPerBlock.x), cols);
	subtract_row_vec_mat<dtype2><<<numBlocks, threadsPerBlock>>>(mat, vec, rows, cols, ldm);
	CudaFunc::checkErrors();
}

template <typename T>
__global__ void squared_diff(T* arr1, T* arr2, T* result, const int length) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < length) {
		result[i] = dpow(dsub(arr1[i], arr2[i]), (T) 2);
	}
}

void CudaFunc::squaredDiff(dtype2* arr1, dtype2* arr2, dtype2* result, int length) {
	int numBlocks = ceil((float) length / BLOCK_HEIGHT);
	squared_diff<dtype2><<<numBlocks, BLOCK_HEIGHT>>>(arr1, arr2, result, length);
	CudaFunc::checkErrors();
}

template <typename S, typename T, typename U>
__global__ void scale_add_2_1(S* arr1, T* arr2, U val, int length) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < length) {
		arr1[i] = dup21(dupadd(d1up(arr1[i]), dupmul(d2up(arr2[i]), val)));
	}
}

template <typename S, typename T, typename U>
__global__ void scale_add_a_1(S* arr1, T* arr2, U val, int length) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < length) {
		arr1[i] = dup21(dupadd(d1up(arr1[i]), dupmul(daup(arr2[i]), val)));
	}
}

template <typename S, typename T, typename U>
__global__ void scale_add_2_a(S* arr1, T* arr2, U val, int length) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < length) {
		arr1[i] = dup2a(dupadd(daup(arr1[i]), dupmul(d2up(arr2[i]), val)));
	}
}

void CudaFunc::scaleAdd21(dtype1* arr1, dtype2* arr2, dtypeh val, int length) {
	int numBlocks = ceil((float) length / BLOCK_HEIGHT);
	scale_add_2_1<dtype1, dtype2, dtypeup><<<numBlocks, BLOCK_HEIGHT>>>(arr1, arr2, h2up(val), length);
	CudaFunc::checkErrors();
}

void CudaFunc::scaleAddA1(dtype1* arr1, dtypea* arr2, dtypeh val, int length) {
	int numBlocks = ceil((float) length / BLOCK_HEIGHT);
	scale_add_a_1<dtype1, dtypea, dtypeup><<<numBlocks, BLOCK_HEIGHT>>>(arr1, arr2, h2up(val), length);
	CudaFunc::checkErrors();
}

void CudaFunc::scaleAdd2A(dtypea* arr1, dtype2* arr2, dtypeh val, int length) {
	int numBlocks = ceil((float) length / BLOCK_HEIGHT);
	scale_add_2_a<dtypea, dtype2, dtypeup><<<numBlocks, BLOCK_HEIGHT>>>(arr1, arr2, h2up(val), length);
	CudaFunc::checkErrors();
}

template <typename S, typename T>
__global__ void copy_1_2(S* arr1, T* arr2, int length) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < length) {
		arr2[i] = d12(arr1[i]);
	}
}

void CudaFunc::copy12(dtype1* arr1, dtype2* arr2, int length) {
	int numBlocks = ceil((float) length / BLOCK_HEIGHT);
	copy_1_2<dtype1, dtype2><<<numBlocks, BLOCK_HEIGHT>>>(arr1, arr2, length);
	CudaFunc::checkErrors();
}

template <typename T>
__global__ void tanh_(T* arr, T* result, const int length) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < length) {
		result[i] = dtanh(arr[i]);
	}
}

void CudaFunc::tanh(dtype2* arr, dtype2* result, int length) {
	int numBlocks = ceil((float) length / BLOCK_HEIGHT);
	tanh_<dtype2><<<numBlocks, BLOCK_HEIGHT>>>(arr, result, length);
	CudaFunc::checkErrors();
}

template <typename T>
__global__ void d_tanh(T* arr, T* result, const int length) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < length) {
		result[i] = dsub((T) 1, dpow(arr[i], (T) 2));
	}
}

void CudaFunc::dTanh(dtype2* arr, dtype2* result, int length) {
	int numBlocks = ceil((float) length / BLOCK_HEIGHT);
	d_tanh<dtype2><<<numBlocks, BLOCK_HEIGHT>>>(arr, result, length);
	CudaFunc::checkErrors();
}

template <typename T>
__global__ void leaky_relu(T* arr, T* result, const int length) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < length) {
		T val = arr[i];
		if (dge(val, (T) 0)) {
			result[i] = val;
		} else {
			result[i] = dmul((T) 0.01, val);
		}
	}
}

void CudaFunc::leakyReLU(dtype2* arr, dtype2* result, int length) {
	int numBlocks = ceil((float) length / BLOCK_HEIGHT);
	leaky_relu<dtype2><<<numBlocks, BLOCK_HEIGHT>>>(arr, result, length);
	CudaFunc::checkErrors();
}

template <typename T>
__global__ void d_leaky_relu(T* arr, T* result, const int length) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < length) {
		T val = arr[i];
		if (dge(val, (T) 0)) {
			result[i] = 1.0;
		} else {
			result[i] = 0.01;
		}
	}
}

void CudaFunc::dLeakyReLU(dtype2* arr, dtype2* result, int length) {
	int numBlocks = ceil((float) length / BLOCK_HEIGHT);
	d_leaky_relu<dtype2><<<numBlocks, BLOCK_HEIGHT>>>(arr, result, length);
	CudaFunc::checkErrors();
}

template <typename T>
__global__ void bi_relu(T* arr, T* result, const int length, T x1, T x2) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < length) {
		T val = arr[i];
		if (dge(val, x2)) {
			result[i] = dsub(val, x2);
		} else if (dle(val, x1)) {
			result[i] = dsub(val, x1);
		} else {
			result[i] = 0.0;
		}
	}
}

void CudaFunc::biReLU(dtype2* arr, dtype2* result, int length, dtype2 x1, dtype2 x2) {
	int numBlocks = ceil((float) length / BLOCK_HEIGHT);
	bi_relu<dtype2><<<numBlocks, BLOCK_HEIGHT>>>(arr, result, length, x1, x2);
	CudaFunc::checkErrors();
}

template <typename T>
__global__ void d_bi_relu(T* arr, T* result, const int length, T x1, T x2) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < length) {
		T val = arr[i];
		if (dge(val, x2)) {
			result[i] = 1.0;
		} else if (dle(val, x1)) {
			result[i] = -1.0;
		} else {
			result[i] = 0.0;
		}
	}
}

void CudaFunc::dBiReLU(dtype2* arr, dtype2* result, int length, dtype2 x1, dtype2 x2) {
	int numBlocks = ceil((float) length / BLOCK_HEIGHT);
	d_bi_relu<dtype2><<<numBlocks, BLOCK_HEIGHT>>>(arr, result, length, x1, x2);
	CudaFunc::checkErrors();
}

template <typename T>
__global__ void leaky_bi_relu(T* arr, T* result, const int length, T x1, T x2, T a) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < length) {
		T val = arr[i];
		if (dge(val, x2)) {
			result[i] = dadd(dsub(val, x2), dmul(a, x2));
		} else if (dle(val, x1)) {
			result[i] = dadd(dsub(val, x1), dmul(a, x1));
		} else {
			result[i] = dmul(a, val);
		}
	}
}

void CudaFunc::leakyBiReLU(dtype2* arr, dtype2* result, int length, dtype2 x1, dtype2 x2, dtype2 a) {
	int numBlocks = ceil((float) length / BLOCK_HEIGHT);
	leaky_bi_relu<dtype2><<<numBlocks, BLOCK_HEIGHT>>>(arr, result, length, x1, x2, a);
	CudaFunc::checkErrors();
}

template <typename T>
__global__ void d_leaky_bi_relu(T* arr, T* result, const int length, T x1, T x2, T a) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < length) {
		T val = arr[i];
		if (dge(val, x2)) {
			result[i] = 1.0;
		} else if (dle(val, x1)) {
			result[i] = -1.0;
		} else {
			result[i] = a;
		}
	}
}

void CudaFunc::dLeakyBiReLU(dtype2* arr, dtype2* result, int length, dtype2 x1, dtype2 x2, dtype2 a) {
	int numBlocks = ceil((float) length / BLOCK_HEIGHT);
	d_leaky_bi_relu<dtype2><<<numBlocks, BLOCK_HEIGHT>>>(arr, result, length, x1, x2, a);
	CudaFunc::checkErrors();
}

template <typename T>
__global__ void sigmoid_(T* arr, T* result, const int length) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < length) {
		result[i] = ddiv(float2d(1.0), dadd(float2d(1.0), dexp(dneg(arr[i]))));
	}
}

void CudaFunc::sigmoid(dtype2* arr, dtype2* result, int length) {
	int numBlocks = ceil((float) length / BLOCK_HEIGHT);
	sigmoid_<dtype2><<<numBlocks, BLOCK_HEIGHT>>>(arr, result, length);
	CudaFunc::checkErrors();
}

template <typename T>
__global__ void dSigmoid_(T* arr, T* result, const int length) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < length) {
		T sigmoid = ddiv(float2d(1.0), dadd(float2d(1.0), dexp(dneg(arr[i]))));
		result[i] = dmul(sigmoid, dsub(float2d(1.0), sigmoid));
	}
}

void CudaFunc::dSigmoid(dtype2* arr, dtype2* result, int length) {
	int numBlocks = ceil((float) length / BLOCK_HEIGHT);
	dSigmoid_<dtype2><<<numBlocks, BLOCK_HEIGHT>>>(arr, result, length);
	CudaFunc::checkErrors();
}

template <typename T>
__global__ void multiply_(T* arr1, T* arr2, T* result, const int length) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < length) {
		result[i] = dup22(dupmul(d2up(arr1[i]), d2up(arr2[i])));
	}
}

void CudaFunc::multiply(dtype2* arr1, dtype2* arr2, dtype2* result, int length) {
	int numBlocks = ceil((float) length / BLOCK_HEIGHT);
	multiply_<dtype2><<<numBlocks, BLOCK_HEIGHT>>>(arr1, arr2, result, length);
	CudaFunc::checkErrors();
}

template <typename T, typename U>
__global__ void multiply_up2(U* arr1, T* arr2, U* result, const int length) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < length) {
		result[i] = arr1[i] * d2up(arr2[i]);
	}
}

void CudaFunc::multiplyUp2(dtypeup* arr1, dtype2* arr2, dtypeup* result, int length) {
	int numBlocks = ceil((float) length / BLOCK_HEIGHT);
	multiply_up2<dtype2, dtypeup><<<numBlocks, BLOCK_HEIGHT>>>(arr1, arr2, result, length);
	CudaFunc::checkErrors();
}

template <typename T>
__global__ void divide_(T* arr1, T* arr2, T* result, const int length) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < length) {
		result[i] = ddiv(arr1[i], arr2[i]);
	}
}

template <typename T>
__global__ void divide_a(T* arr1, T* arr2, T* result, const int length) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < length) {
		result[i] = dup2a(dupdiv(daup(arr1[i]), daup(arr2[i])));
	}
}
//		result[i] = dup22(dupmul(d2up(arr1[i]), d2up(arr2[i])));

void CudaFunc::divide(dtype2* arr1, dtype2* arr2, dtype2* result, int length) {
	int numBlocks = ceil((float) length / BLOCK_HEIGHT);
	divide_<dtype2><<<numBlocks, BLOCK_HEIGHT>>>(arr1, arr2, result, length);
	CudaFunc::checkErrors();
}

void CudaFunc::divideA(dtypea* arr1, dtypea* arr2, dtypea* result, int length) {
	int numBlocks = ceil((float) length / BLOCK_HEIGHT);
	divide_a<dtypea><<<numBlocks, BLOCK_HEIGHT>>>(arr1, arr2, result, length);
	CudaFunc::checkErrors();
}

template <typename T>
__global__ void add_(T* arr1, T* arr2, T* result, const int length) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < length) {
		result[i] = dadd(arr1[i], arr2[i]);
	}
}

void CudaFunc::add(dtype2* arr1, dtype2* arr2, dtype2* result, int length) {
	int numBlocks = ceil((float) length / BLOCK_HEIGHT);
	add_<dtype2><<<numBlocks, BLOCK_HEIGHT>>>(arr1, arr2, result, length);
	CudaFunc::checkErrors();
}

template <typename T>
__global__ void subtract_(T* arr1, T* arr2, T* result, const int length) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < length) {
		result[i] = dsub(arr1[i], arr2[i]);
	}
}

void CudaFunc::subtract(dtype2* arr1, dtype2* arr2, dtype2* result, int length) {
	int numBlocks = ceil((float) length / BLOCK_HEIGHT);
	subtract_<dtype2><<<numBlocks, BLOCK_HEIGHT>>>(arr1, arr2, result, length);
	CudaFunc::checkErrors();
}

template <typename T>
__global__ void subtract_from_one(T* arr, T* result, const int length) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < length) {
		result[i] = dsub(float2d(1.0), arr[i]);
	}
}

void CudaFunc::subtractFromOne(dtype2* arr, dtype2* result, int length) {
	int numBlocks = ceil((float) length / BLOCK_HEIGHT);
	subtract_from_one<dtype2><<<numBlocks, BLOCK_HEIGHT>>>(arr, result, length);
	CudaFunc::checkErrors();
}

template <typename T>
__global__ void match_mask(int* ixMatch, int* ixTarget, T* mask, int length, unsigned int *inputLengths,
		int s, float pDropMatch) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= length) {
		return;
	}

	int sentLength = inputLengths[i];
	if (s >= sentLength) {
		mask[i] = float2d(0.0);
		return;
	}

	int match = ixMatch[i] - 1;
	if (match == ixTarget[i]) {
		if (pDropMatch == 1.0f) {
			mask[i] = float2d(0.0);
		} else {
			curandState localState = curandStates[i];
			float r = curand_uniform(&localState);

			if (r < pDropMatch) {
				mask[i] = float2d(0.0);
			} else {
				mask[i] = float2d(1.0);
			}
			curandStates[i] = localState;
		}
	} else {
		mask[i] = float2d(1.0);
	}
}

void CudaFunc::matchMask(int* ixMatch, int* ixTarget, dtype2* mask, int length, unsigned int *inputLengths,
		int s) {
	int numBlocks = ceil((float) length / CURAND_GRID_DIM);
	match_mask<dtype2><<<numBlocks, CURAND_GRID_DIM>>>(ixMatch, ixTarget, mask, length, inputLengths,
			s, pDropMatch);
	CudaFunc::checkErrors();
}

template <typename T>
__global__ void match_mask_d(int* ixMatch, T* ixTarget, T* mask, int length, unsigned int *inputLengths,
		int s, float pDropMatch) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= length) {
		return;
	}

	int sentLength = inputLengths[i];
	if (s >= sentLength) {
		mask[i] = float2d(0.0);
		return;
	}

	int match = ixMatch[i] - 1;
	int target = d2int_rd(dadd(ixTarget[i], (T) 0.5));
	if (match == target) {
		if (pDropMatch == 1.0f) {
			mask[i] = float2d(0.0);
		} else {
			curandState localState = curandStates[i];
			float r = curand_uniform(&localState);

			if (r < pDropMatch) {
				mask[i] = float2d(0.0);
			} else {
				mask[i] = float2d(1.0);
			}
			curandStates[i] = localState;
		}
	} else {
		mask[i] = float2d(1.0);
	}
}

void CudaFunc::matchMaskd(int* ixMatch, dtype2* ixTarget, dtype2* mask, int length, unsigned int *inputLengths,
		int s) {
	int numBlocks = ceil((float) length / CURAND_GRID_DIM);
	match_mask_d<dtype2><<<numBlocks, CURAND_GRID_DIM>>>(ixMatch, ixTarget, mask, length, inputLengths,
			s, pDropMatch);
	CudaFunc::checkErrors();
}

template <typename T>
__global__ void fill_input(T **inputs_fwd, T **inputs_rev, T *targets, int *batchIx, unsigned int *inputLengths,
		int s, dtype2* dict, int numWords, int batchSize) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= batchSize) {
		return;
	}
	int j = blockIdx.y;

	int length = inputLengths[i];
	int valIx = IDX2(i,j,batchSize);

	if (s >= length) {
		if (inputs_rev != NULL) {
			inputs_rev[s][valIx] = 0.0;
		}
		if (inputs_fwd != NULL) {
			inputs_fwd[s][valIx] = 0.0;
		}
		targets[valIx] = 0.0;
	} else {
		int ix = batchIx[i];
		T val = dict[IDX2(ix,j,numWords)];

		if (inputs_rev != NULL) {
			int input_rev_ix = length - 1 - s;
			inputs_rev[input_rev_ix][valIx] = val;
		}

		if (inputs_fwd != NULL) {
			inputs_fwd[s][valIx] = val;
		}

		targets[valIx] = val;
	}
}

void CudaFunc::fillInput(dtype2 **inputs_fwd, dtype2 **inputs_rev, dtype2 *targets, int *batchIx, unsigned int *inputLengths,
		int s, dtype2* dict, int numWords) {
	dim3 threadsPerBlock(BLOCK_HEIGHT, 1);
	dim3 numBlocks(ceil((float) batchSize / threadsPerBlock.x), WV_LENGTH);
	fill_input<dtype2><<<numBlocks, threadsPerBlock>>>(inputs_fwd, inputs_rev, targets, batchIx, inputLengths,
			s, dict, numWords, batchSize);
	CudaFunc::checkErrors();
}

template <typename T>
__global__ void copy_targets_to_inputs(T **inputs_fwd, T **inputs_rev, T *targets, unsigned int *inputLengths,
		int s, int batchSize) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= batchSize) {
		return;
	}
	int j = blockIdx.y;

	int length = inputLengths[i];
	int valIx = IDX2(i,j,batchSize);

	if (s >= length) {
		if (inputs_rev != NULL) {
			inputs_rev[s][valIx] = 0.0;
		}
		if (inputs_fwd != NULL) {
			inputs_fwd[s][valIx] = 0.0;
		}
	} else {
		T val = targets[valIx];

		if (inputs_rev != NULL) {
			int input_rev_ix = length - 1 - s;
			inputs_rev[input_rev_ix][valIx] = val;
		}

		if (inputs_fwd != NULL) {
			inputs_fwd[s][valIx] = val;
		}
	}
}

void CudaFunc::copyTargetsToInputs(dtype2 **inputs_fwd, dtype2 **inputs_rev, dtype2 *targets,
		unsigned int *inputLengths, int s) {
	dim3 threadsPerBlock(BLOCK_HEIGHT, 1);
	dim3 numBlocks(ceil((float) batchSize / threadsPerBlock.x), WV_LENGTH);
	copy_targets_to_inputs<dtype2><<<numBlocks, threadsPerBlock>>>(inputs_fwd, inputs_rev, targets, inputLengths,
			s, batchSize);
	CudaFunc::checkErrors();
}

__global__ void init_curand_state(unsigned long long seed) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    /* Each thread gets same seed, a different sequence
       number, no offset */
    curand_init(seed, id, 0, &curandStates[id]);
}

void CudaFunc::initCurandStates(unsigned long long seed, int maxLength) {
	int curandNumBlocks = ceil((float) maxLength / (float) CURAND_GRID_DIM);

	if (curandStates == NULL) {
		curandStatesArraySize = curandNumBlocks * CURAND_GRID_DIM * sizeof(curandState);
		checkCudaErrors(cudaMalloc((void **)&curandStates, curandStatesArraySize));
	}

    init_curand_state<<<curandNumBlocks, CURAND_GRID_DIM>>>(seed);
	CudaFunc::checkErrors();
}

void CudaFunc::getCurandStates(char** a_h, size_t *arraySize) {
    *a_h = (char*) malloc(curandStatesArraySize);
	checkCudaErrors(cudaMemcpy(*a_h, curandStates, curandStatesArraySize, cudaMemcpyDeviceToHost));
	*arraySize = curandStatesArraySize;
}

void CudaFunc::setCurandStates(char* a_h, size_t arraySize) {
	curandStatesArraySize = arraySize;
	if (curandStates == NULL) {
		checkCudaErrors(cudaMalloc((void **)&curandStates, curandStatesArraySize));
	}
	checkCudaErrors(cudaMemcpy(curandStates, a_h, curandStatesArraySize, cudaMemcpyHostToDevice));
}

void CudaFunc::freeCurandStates() {
	checkCudaErrors(cudaFree(curandStates));
	curandStates = NULL;
}

template <typename T>
__global__ void fill_dropout_mask(T* mask, const int length, const float p) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= length) {
		return;
	}
	curandState localState = curandStates[i];
	float r = curand_uniform(&localState);
	float pr = 1.0/p; //p is retention probability
	if (r > p) {
		mask[i] = 0;
	} else {
		mask[i] = float2d(pr);
	}
	curandStates[i] = localState;
}

void CudaFunc::fillDropoutMask(dtype2* mask, const int length, const float p) {
	int numBlocks = ceil((float) length / CURAND_GRID_DIM);
	fill_dropout_mask<dtype2><<<numBlocks, CURAND_GRID_DIM>>>(mask, length, p);
	CudaFunc::checkErrors();
}

template <typename T>
__global__ void mask_row_vec_mat(T* mat, T* mask, const int rows, const int cols, int ldm) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= rows) {
		return;
	}
	int j = blockIdx.y;

	if (deq(mask[j], 0)) {
		mat[IDX2(i,j,ldm)] = 0;
	}
}

void CudaFunc::maskRowVecMat(dtype2* mat, dtype2* mask, int rows, int cols, int ldm) {
	dim3 threadsPerBlock(BLOCK_HEIGHT, 1);
	dim3 numBlocks(ceil((float) rows / threadsPerBlock.x), cols);
	mask_row_vec_mat<dtype2><<<numBlocks, threadsPerBlock>>>(mat, mask, rows, cols, ldm);
	CudaFunc::checkErrors();
}

template <typename T>
__global__ void mask_col_vec_mat(T* mat, T* mask, const int rows, const int cols, int ldm) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= rows) {
		return;
	}
	int j = blockIdx.y;

	if (deq(mask[i], 0)) {
		mat[IDX2(i,j,ldm)] = 0;
	}
}

void CudaFunc::maskColVecMat(dtype2* mat, dtype2* mask, int rows, int cols, int ldm) {
	dim3 threadsPerBlock(BLOCK_HEIGHT, 1);
	dim3 numBlocks(ceil((float) rows / threadsPerBlock.x), cols);
	mask_col_vec_mat<dtype2><<<numBlocks, threadsPerBlock>>>(mat, mask, rows, cols, ldm);
	CudaFunc::checkErrors();
}

template <typename T>
__global__ void mask_mat(T* a, T* mask, const int rows, const int cols, int ldm) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= rows) {
		return;
	}
	int j = blockIdx.y;

	int ix = IDX2(i,j,ldm);
	if (deq(mask[ix], 0)) {
		a[ix] = 0;
	}
}

void CudaFunc::maskMat(dtype2* a, dtype2* mask, int rows, int cols, int ldm) {
	dim3 threadsPerBlock(BLOCK_HEIGHT, 1);
	dim3 numBlocks(ceil((float) rows / threadsPerBlock.x), cols);
	mask_mat<dtype2><<<numBlocks, threadsPerBlock>>>(a, mask, rows, cols, ldm);
	CudaFunc::checkErrors();
}

template <typename T>
__global__ void fill_seq_masks(T* masks, unsigned int* lengths, const int rows, const int cols) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= rows) {
		return;
	}
	int j = blockIdx.y;

	int ix = IDX2(i,j,rows);
	if (j < lengths[i]) {
		masks[ix] = 1.0;
	} else {
		masks[ix] = 0.0;
	}
}

void CudaFunc::fillSeqMasks(dtype2* masks, unsigned int* lengths) {
	int rows = batchSize;
	int cols = maxSentenceLength;
	dim3 threadsPerBlock(BLOCK_HEIGHT, 1);
	dim3 numBlocks(ceil((float) rows / threadsPerBlock.x), cols);
	fill_seq_masks<dtype2><<<numBlocks, threadsPerBlock>>>(masks, lengths, rows, cols);
	CudaFunc::checkErrors();
}

template <typename T>
__global__ void sum_cols1_(T *A, T *out, const int increment,
		const int a0, const int a1)
{
    const int t_i = threadIdx.y;
    const int t_j = threadIdx.x;
    const int dim_i = blockDim.y;
    const int dim_j = blockDim.x;
    const int col = dim_j*blockIdx.x + t_j;
    const int A_offset = t_i*a1 + col;
    const int data_offset = t_i*dim_j + t_j;

    extern __shared__ dtype2 shared_data[];
    T* data = (T*)shared_data;

    // stage 1: loop threads across A to reduce to shared memory block
    const int step = dim_i*a1;
    const int limit = a0*a1;
    T sum = 0;
    int index = A_offset;
    for (int i=0; i < limit; i += step)
    {
        if (index < limit)
            sum += A[index];
        index += step;
    }
    data[data_offset] = sum;

    // stage 2: reduction within block
    // note: assumes that dim_i is divisible by 2
    for (int s=dim_i/2; s > 0; s>>=1)
    {
        __syncthreads();

        if (t_i < s)
            data[data_offset] += data[data_offset + s*dim_j];
    }

    if (t_i == 0)
    {
        if (increment)
            out[col] += data[t_j];
        else
            out[col] = data[t_j];
    }
}

//void sum_cols1() {
//	int block_x = min()
//
////    block_x = min(a._block[0] // 32, a.shape[1])
////    block_y = 32
////    grid = (a.shape[1] // block_x + (a.shape[1] % block_x != 0), 1)
//
//}

//template <unsigned int blockSize>
//__global__ void sum_cols2_(dtype2 *g_idata, dtype2 *g_odata, unsigned int n)
//{
//	extern __shared__ dtype2 sdata[];
//
//	unsigned int tid = threadIdx.x;
//	unsigned int i = blockIdx.x*n + tid;
//	unsigned int gridSize = blockSize*2*gridDim.x;
//	sdata[tid] = 0;
//
//	while (i < n) {
//		sdata[tid] += g_idata[i] + g_idata[i+blockSize]; i += gridSize;
//	}
//	__syncthreads();
//
//	if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
//	if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
//	if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
//
//
//	if (tid < 32) {
//		if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
//		if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
//		if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
//		if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
//		if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
//		if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
//	}
//
//	if (tid == 0) g_odata[blockIdx.x] += sdata[0];
//}
//
//void CudaFunc::sum_cols2(dtype2* in, dtype2* out, int numRows, int numCols) {
//
//	int blockSize = min(32, numRows/2);
//	int numBlocks = numCols;
////	int smemSize = blockSize;
//
//	sum_cols2_<32><<< numBlocks, blockSize >>>(in, out, numRows);
//
////	switch (blockSize)
////	{
////		case 512:
////		reduce5<512><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata); break;
////		case 256:
////		reduce5<256><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata); break;
////		case 128:
////		reduce5<128><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata); break;
////		case 64:
////		reduce5< 64><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata); break;
////		case 32:
////		reduce5< 32><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata); break;
////		case 16:
////		reduce5< 16><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata); break;
////		case 8:
////		reduce5< 8><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata); break;
////		case 4:
////		reduce5< 4><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata); break;
////		case 2:
////		reduce5< 2><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata); break;
////		case 1:
////		reduce5< 1><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata); break;
////	}
//}
//
//__global__ void reduce3(dtype2 *g_idata, dtype2 *g_odata) {
//    extern __shared__ dtype2 sdata[];
//
//    // each thread loads one element from global to shared mem
//    unsigned int tid = threadIdx.x;
//    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
//    sdata[tid] = g_idata[i];
//    __syncthreads();
//
//    // do reduction in shared mem
//    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
//        if (tid < s) {
//            sdata[tid] += sdata[tid + s];
//        }
//        __syncthreads();
//    }
//
//    // write result for this block to global mem
//    if (tid == 0) g_odata[blockIdx.x] += sdata[0];
//}
//
//void CudaFunc::sum_cols_reduce3(dtype2* in, dtype2* out, int numRows, int numCols) {
//	int blockSize = numRows;
//	int numBlocks = numCols;
//	int smemSize = blockSize * sizeof(dtype2);
//	reduce3<<< numBlocks, blockSize, smemSize >>>(in, out);
//}

__global__ void reduce4(dtype2 *g_idata, dtype2 *g_odata) {
    extern __shared__ dtype2 sdata[];

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
    sdata[tid] = dadd(g_idata[i], g_idata[i+blockDim.x]);
    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if (tid < s) {
            sdata[tid] = dadd(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = dadd(g_odata[blockIdx.x], sdata[0]);
}

void CudaFunc::sum_cols_reduce4(dtype2* in, dtype2* out, int numRows, int numCols) {
	int blockSize = numRows / 2;
	int numBlocks = numCols;
	int smemSize = blockSize * sizeof(dtype2);
	reduce4<<< numBlocks, blockSize, smemSize >>>(in, out);
	CudaFunc::checkErrors();
}

//__global__ void reduce5(dtype2 *g_idata, dtype2 *g_odata) {
//    extern __shared__ dtype2 sdata[];
//
//    // perform first level of reduction,
//    // reading from global memory, writing to shared memory
//    unsigned int tid = threadIdx.x;
//    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
//    sdata[tid] = g_idata[i] + g_idata[i+blockDim.x];
//    __syncthreads();
//
//    // do reduction in shared mem
//    for (unsigned int s=blockDim.x/2; s>32; s>>=1) {
//        if (tid < s) {
//            sdata[tid] += sdata[tid + s];
//        }
//        __syncthreads();
//    }
//
//    if (tid < 32)
//    {
//        sdata[tid] += sdata[tid + 32];
//        sdata[tid] += sdata[tid + 16];
//        sdata[tid] += sdata[tid + 8];
//        sdata[tid] += sdata[tid + 4];
//        sdata[tid] += sdata[tid + 2];
//        sdata[tid] += sdata[tid + 1];
//    }
//
//    // write result for this block to global mem
//    if (tid == 0) g_odata[blockIdx.x] += sdata[0];
//}
//
//void CudaFunc::sum_cols_reduce5(dtype2* in, dtype2* out, int numRows, int numCols) {
//	int blockSize = numRows / 2;
//	int numBlocks = numCols;
//	int smemSize = blockSize * sizeof(dtype2);
//	reduce5<<< numBlocks, blockSize, smemSize >>>(in, out);
//}
//
//template <unsigned int blockSize>
//__global__ void reduce6(dtype2 *g_idata, dtype2 *g_odata) {
//    extern __shared__ dtype2 sdata[];
//
//    // perform first level of reduction,
//    // reading from global memory, writing to shared memory
//    unsigned int tid = threadIdx.x;
//    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
//    sdata[tid] = g_idata[i] + g_idata[i+blockDim.x];
//    __syncthreads();
//
//    // do reduction in shared mem
//    if (blockSize >= 512) {
//        if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads();
//    }
//    if (blockSize >= 256) {
//        if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads();
//    }
//    if (blockSize >= 128) {
//        if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads();
//    }
//
//    if (tid < 32) {
//		if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
//		if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
//		if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
//		if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
//		if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
//		if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
//    }
//
//    // write result for this block to global mem
//    if (tid == 0) g_odata[blockIdx.x] += sdata[0];
//}
//
//void CudaFunc::sum_cols_reduce6(dtype2* in, dtype2* out, int numRows, int numCols) {
//	int blockSize = numRows / 2;
//	int numBlocks = numCols;
//	int smemSize = blockSize * sizeof(dtype2);
//
//	switch (blockSize)
//	{
//		case 512:
//		reduce6<512><<< numBlocks, blockSize, smemSize >>>(in, out); break;
//		case 256:
//		reduce6<256><<< numBlocks, blockSize, smemSize >>>(in, out); break;
//		case 128:
//		reduce6<128><<< numBlocks, blockSize, smemSize >>>(in, out); break;
//		case 64:
//		reduce6< 64><<< numBlocks, blockSize, smemSize >>>(in, out); break;
//		case 32:
//		reduce6< 32><<< numBlocks, blockSize, smemSize >>>(in, out); break;
//		case 16:
//		reduce6< 16><<< numBlocks, blockSize, smemSize >>>(in, out); break;
//		case 8:
//		reduce6< 8><<< numBlocks, blockSize, smemSize >>>(in, out); break;
//		case 4:
//		reduce6< 4><<< numBlocks, blockSize, smemSize >>>(in, out); break;
//		case 2:
//		reduce6< 2><<< numBlocks, blockSize, smemSize >>>(in, out); break;
//		case 1:
//		reduce6< 1><<< numBlocks, blockSize, smemSize >>>(in, out); break;
//	}
//}
//
//template <unsigned int blockSize>
//__global__ void reduce7(dtype2 *g_idata, dtype2 *g_odata, unsigned int n) {
//    extern __shared__ dtype2 sdata[];
//
//    unsigned int tid = threadIdx.x;
//    unsigned int i = blockIdx.x*(blockSize*2) + threadIdx.x;
//    unsigned int gridSize = blockSize*2*gridDim.x;
//    sdata[tid] = 0;
//
//    while (i < n) {
//        sdata[tid] += g_idata[i] + g_idata[i+blockSize];
//        i += gridSize;
//    }
//    __syncthreads();
//
//    if (blockSize >= 512) {
//        if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads();
//    }
//    if (blockSize >= 256) {
//        if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads();
//    }
//    if (blockSize >= 128) {
//        if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads();
//    }
//
//    if (tid < 32) {
//		if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
//		if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
//		if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
//		if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
//		if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
//		if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
//    }
//
//    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
//}
//
////reduce7 not working yet
//void CudaFunc::sum_cols_reduce7(dtype2* in, dtype2* out, int numRows, int numCols) {
//	int blockSize = numRows / 2;
//	int numBlocks = numCols;
//	int smemSize = blockSize * sizeof(dtype2);
//	reduce7<2><<< numBlocks, blockSize, smemSize >>>(in, out, numRows);
//}

__global__ void sum_rows_reduce4_(dtype2 *g_idata, dtype2 *g_odata, unsigned int lda, int numCols) {
    extern __shared__ dtype2 sdata[];

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
//    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
    unsigned int i = threadIdx.x*lda + blockIdx.x;
    int next = i+blockDim.x*lda;
    if (next < numCols*lda) {
        sdata[tid] = dadd(g_idata[i], g_idata[next]);
    } else {
    	sdata[tid] = g_idata[i];
    }
    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if (tid < s) {
            sdata[tid] = dadd(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = dadd(g_odata[blockIdx.x], sdata[0]);
}

int upToPowerOf2(int in) {
	float log2 = logf((float) in) / logf((float) 2);
	int pow2 = ceil(log2);
	return (int) (powf(2, pow2) + 0.5);
}

void CudaFunc::sum_rows_reduce4(dtype2* in, dtype2* out, int numRows, int numCols) {
	int half = ceil((float) numCols / 2.0f);
	int blockSize = upToPowerOf2(half);
	int numBlocks = numRows;
	int smemSize = blockSize * sizeof(dtype2);
	sum_rows_reduce4_<<< numBlocks, blockSize, smemSize >>>(in, out, numRows, numCols);
	CudaFunc::checkErrors();
}

//__global__ void amax_cols_reduce4(dtype2 *g_idata, int *g_odata, int numRows) {
//    extern __shared__ dtype2 sdata[];
//
//    // perform first level of reduction,
//    // reading from global memory, writing to shared memory
//    unsigned int tid = threadIdx.x;
//    unsigned int ix1 = blockIdx.x*(blockDim.x*2) + threadIdx.x;
//    int ix2 = ix1+blockDim.x;
//    if  (threadIdx.x+blockDim.x < numRows) {
//		dtype2 v1 = g_idata[ix1];
//		dtype2 v2 = g_idata[ix2];
//		if (dgt(v2, v1)) {
//			sdata[tid] = v2;
//			sdata[tid+blockDim.x] = ix2;
//		} else {
//			sdata[tid] = v1;
//			sdata[tid+blockDim.x] = ix1;
//		}
//    } else {
//    	sdata[tid] = g_idata[ix1];
//    }
//    __syncthreads();
//
//    // do reduction in shared mem
//    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
//        if (tid < s) {
//        	if (dgt(sdata[tid + s], sdata[tid])) {
//        		sdata[tid] = sdata[tid+s];
//        		sdata[tid+blockDim.x] = sdata[tid+s+blockDim.x];
//        	}
//        }
//        __syncthreads();
//    }
//
//    // write result for this block to global mem
//    if (tid == 0) g_odata[blockIdx.x] = d2int_rn(sdata[blockDim.x]);
//}
//
//void CudaFunc::amaxColsReduce4(dtype2* in, int* result, int numRows, int numCols) {
//	int half = ceil((float) numRows / 2.0f);
//	int blockSize = upToPowerOf2(half);
//	int numBlocks = numCols;
//	int smemSize = 2 * blockSize * sizeof(dtype2);
//	amax_cols_reduce4<<< numBlocks, blockSize, smemSize >>>(in, result, numRows);
//}

void CudaFunc::checkErrors() {
	checkCudaErrors(cudaPeekAtLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

} /* namespace netlib */
