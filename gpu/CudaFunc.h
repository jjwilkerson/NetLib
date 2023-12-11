/**
 * @file
 * @brief Declares CudaFunc class, which contains CUDA functions for execution on GPU.
 *
 */

#ifndef CUDAFUNC_H_
#define CUDAFUNC_H_

#include "../NetLib.h"

namespace netlib {

/**
 * @brief Contains CUDA functions for execution on GPU.
 *
 */
class CudaFunc {
public:
	CudaFunc();
	virtual ~CudaFunc();
	static float pDropMatch;
	static int batchSize;
	static int maxSentenceLength;
	static void maskByLength(dtype2* mat, unsigned int* lengths, int s, int layerSize,
			int batchSize);
	static void multColVecMat(dtype2* mat, dtype2* vec, int rows, int cols, int ldm);
	static void multColVecMatUp(dtypeup* mat, dtype2* vec, int rows, int cols, int ldm);
	static void multRowVecMat(dtype2* mat, dtype2* vec, int rows, int cols, int ldm);
	static void divColVecMat(dtype2* mat, dtype2* vec, int rows, int cols, int ldm);
	static void divRowVecMat(dtype2* mat, dtype2* vec, int rows, int cols, int ldm);
	static void dupIn(dtype2* dest, dtype2* src, unsigned int* lengths, int s, int layerSize,
			int batchSize);
	static void fill(dtype2* arr, const int size, const dtypeh val);
	static void fill1(dtype1* arr, const int size, const dtypeh val);
	static void iadd(dtype2* arr, const int size, const dtypeh val);
	static void iaddA(dtypea* arr, const int size, const dtypeh val);
	static void isquare(dtype2* arr, const int size);
	static void isqrt(dtype2* arr, const int size);
	static void isqrtA(dtypea* arr, const int size);
	static void iinvert(dtype2* arr, const int size);
	static void iexp(dtype2* arr, const int size);
	static void inegLog(dtype2* arr, const int size);
	static void inegLog1minus(dtype2* arr, const int size);
	static void count(dtype2* arr, const int size, const dtypeh val, int* result);
	static void iclip(dtype2* arr, const int size, const dtypeh maxVal);
	static void imax(dtype2* arr, const int size, const dtypeh val);
	static void scaleAndCheck(dtype2* arr, const int size, const dtypeh val, bool* overflow);
	static void checkForOverflow(dtype2* arr, const int size, bool* overflow);
	static void addColumnVecMat(dtype2* mat, dtype2* vec, int rows, int cols, int ldm);
	static void subtractColumnVecMat(dtype2* mat, dtype2* vec, int rows, int cols, int ldm);
	static void addRowVecMat(dtype2* mat, dtype2* vec, int rows, int cols, int ldm);
	static void subtractRowVecMat(dtype2* mat, dtype2* vec, int rows, int cols, int ldm);
	static void squaredDiff(dtype2 *arr1, dtype2* arr2, dtype2* result, int length);
	static void scaleAdd21(dtype1* arr1, dtype2* arr2, const dtypeh val, int length);
	static void scaleAddA1(dtype1* arr1, dtypea* arr2, const dtypeh val, int length);
	static void scaleAdd2A(dtypea* arr1, dtype2* arr2, const dtypeh val, int length);
	static void copy12(dtype1* arr1, dtype2* arr2, int length);
	static void tanh(dtype2* arr, dtype2* result, int length);
	static void dTanh(dtype2* arr, dtype2* result, int length);
	static void leakyReLU(dtype2* arr, dtype2* result, int length);
	static void dLeakyReLU(dtype2* arr, dtype2* result, int length);
	static void biReLU(dtype2* arr, dtype2* result, int length, dtype2 x1, dtype2 x2);
	static void dBiReLU(dtype2* arr, dtype2* result, int length, dtype2 x1, dtype2 x2);
	static void leakyBiReLU(dtype2* arr, dtype2* result, int length, dtype2 x1, dtype2 x2, dtype2 a);
	static void dLeakyBiReLU(dtype2* arr, dtype2* result, int length, dtype2 x1, dtype2 x2, dtype2 a);
	static void sigmoid(dtype2* arr, dtype2* result, int length);
	static void dSigmoid(dtype2* arr, dtype2* result, int length);

	static void multiply(dtype2 *arr1, dtype2* arr2, dtype2* result, int length);
	static void multiplyUp2(dtypeup *arr1, dtype2* arr2, dtypeup* result, int length);
	static void divide(dtype2 *arr1, dtype2* arr2, dtype2* result, int length);
	static void divideA(dtypea *arr1, dtypea* arr2, dtypea* result, int length);
	static void add(dtype2 *arr1, dtype2* arr2, dtype2* result, int length);
	static void subtract(dtype2 *arr1, dtype2* arr2, dtype2* result, int length);
	static void subtractFromOne(dtype2 *arr, dtype2* result, int length);

	static void initCurandStates(unsigned long long seed, int maxLength);
	static void getCurandStates(char** a_h, size_t *arraySize);
	static void setCurandStates(char* a_h, size_t arraySize);
	static void freeCurandStates();
	static void fillDropoutMask(dtype2* mask, const int length, const float p);
	static void fillSeqMasks(dtype2* masks, unsigned int* lengths);
	static void maskRowVecMat(dtype2* mat, dtype2* mask, int rows, int cols, int ldm);
	static void maskColVecMat(dtype2* mat, dtype2* mask, int rows, int cols, int ldm);
	static void maskMat(dtype2* a, dtype2* mask, int rows, int cols, int ldm);
	static void matchMask(int* ixMatch, int* ixTarget, dtype2* mask, int length, unsigned int *inputLengths,
			int s);
	static void matchMaskd(int* ixMatch, dtype2* ixTarget, dtype2* mask, int length, unsigned int *inputLengths,
			int s);
	static void fillInput(dtype2 **inputs_fwd, dtype2 **inputs_rev, dtype2 *targets, int *batchIx,
			unsigned int *inputLengths, int s, dtype2* dict, int numWords);
	static void copyTargetsToInputs(dtype2 **inputs_fwd, dtype2 **inputs_rev, dtype2 *targets,
			unsigned int *inputLengths, int s);
	static void selectIxByRow(dtype2* activation, dtype2* ix, dtype2* out,
			unsigned int* inputLengths, int s);
	static void clearIxByRow(dtype2* activation, dtype2* ix, unsigned int* inputLengths, int s);
	static void addToIxByRow(dtype2* activation, dtype2* ix, const dtypeh val);

	static void sum_cols2(dtype2* in, dtype2* out, int numRows, int numCols);
	static void sum_cols_reduce3(dtype2* in, dtype2* out, int numRows, int numCols);
	static void sum_cols_reduce4(dtype2* in, dtype2* out, int numRows, int numCols);
	static void sum_cols_reduce5(dtype2* in, dtype2* out, int numRows, int numCols);
	static void sum_cols_reduce6(dtype2* in, dtype2* out, int numRows, int numCols);
	static void sum_cols_reduce7(dtype2* in, dtype2* out, int numRows, int numCols);

	static void sum_rows_reduce4(dtype2* in, dtype2* out, int numRows, int numCols);

	static void amaxColsReduce4(dtype2* in, int* result, int numRows, int numCols);
	static void checkErrors();
};

} /* namespace netlib */

#endif /* CUDAFUNC_H_ */

