/**
 * @file
 * @brief Holds global function and constant definitions.
 *
 */

#ifndef NETLIB_H_
#define NETLIB_H_

#include <cstddef>
#include <string>
#include <cuda_fp16.h>

//#define DEBUG 1
#define SINGLE_HALF 1
//#define SINGLE_ASINGLE_HALF 1
//#define HALF 1

#ifdef DEBUG
#define DEVICE_DOUBLE 1
#define HOST_DOUBLE 1
#define DICT_SINGLE 1
typedef double dtype1; //param storage, device
typedef double dtype2; //other, device
typedef double dtype3; //dict, device
typedef double dtypea;
typedef double dtypeup;
typedef double dtypeh; //host
#define d2h(a) (a)
#define d1h(a) (a)
#define d2up(a) (a)
#define d1up(a) (a)
#define daup(a) (a)
#define dup21(a) (a)
#define dup2a(a) (a)
#define dup22(a) (a)
#define d21(a) (a)
#define da1(a) (a)
#define d2a(a) (a)
#define d12(a) (a)
#define h21(a) (a)
#define h2a(a) (a)
#define h2up(a) (a)
#define h23(a) (a)
#define dup2h(a) (a)
#define d2float(a) (a)
#define da2float(a) (a)
#define float2d3(a) (a)
#define float2d(a) (a)
#define float2da(a) (a)
#define float21(a) (a)
#define float2h(a) (a)
#elif defined(SINGLE_HALF)
#define DEVICE_HALF 1
#define DEVICE1_SINGLE 1
#define HOST_SINGLE 1
#define DICT_HALF 1
#define PARAM_SINGLE_HALF 1
typedef float dtype1;
typedef half dtype2;
typedef half dtype3;
typedef half dtypea;
typedef float dtypeup;
typedef float dtypeh;
#define d2h(a) __half2float(a)
#define d1h(a) (a)
#define d2up(a) __half2float(a)
#define d1up(a) (a)
#define daup(a) __half2float(a)
#define dup21(a) (a)
#define dup2a(a) __float2half(a)
#define dup22(a) __float2half(a)
#define d21(a) __half2float(a)
#define da1(a) __half2float(a)
#define d2a(a) (a)
#define d12(a) __float2half(a)
#define h21(a) (a)
#define h2a(a) __float2half(a)
#define h2up(a) (a)
#define h23(a) __float2half(a)
#define dup2h(a) (a)
#define d2float(a) __half2float(a)
#define da2float(a) __half2float(a)
#define float2d3(a) __float2half(a)
#define float2d(a) __float2half(a)
#define float2da(a) __float2half(a)
#define float21(a) (a)
#define float2h(a) (a)
#elif defined(SINGLE_ASINGLE_HALF)
#define DEVICE_HALF 1
#define DEVICE1_SINGLE 1
#define DEVICEA_SINGLE 1
#define HOST_SINGLE 1
#define DICT_HALF 1
#define PARAM_SINGLE_HALF 1
typedef float dtype1;
typedef half dtype2;
typedef half dtype3;
typedef float dtypea;
typedef float dtypeup;
typedef float dtypeh;
#define d2h(a) __half2float(a)
#define d1h(a) (a)
#define d2up(a) __half2float(a)
#define d1up(a) (a)
#define daup(a) (a)
#define dup21(a) (a)
#define dup2a(a) (a)
#define dup22(a) __float2half(a)
#define d21(a) __half2float(a)
#define da1(a) (a)
#define d2a(a) __half2float(a)
#define d12(a) __float2half(a)
#define h21(a) (a)
#define h2a(a) (a)
#define h2up(a) (a)
#define dup2h(a) (a)
#define d2float(a) __half2float(a)
#define da2float(a) (a)
#define float2d3(a) __float2half(a)
#define float2d(a) __float2half(a)
#define float2da(a) (a)
#define float21(a) (a)
#elif defined(HALF)
#define DEVICE_HALF 1
#define HOST_SINGLE 1
#define DICT_HALF 1
typedef half dtype1;
typedef half dtype2;
typedef half dtype3;
typedef half dtypea;
typedef float dtypeup;
typedef float dtypeh;
#define d2h(a) __half2float(a)
#define d1h(a) __half2float(a)
#define d2up(a) __half2float(a)
#define d1up(a) __half2float(a)
#define daup(a) __half2float(a)
#define dup21(a) __float2half(a)
#define dup2a(a) __float2half(a)
#define dup22(a) __float2half(a)
#define d21(a) (a)
#define da1(a) (a)
#define d2a(a) (a)
#define d12(a) (a)
#define h21(a) __float2half(a)
#define h2a(a) __float2half(a)
#define h2up(a) (a)
#define dup2h(a) (a)
#define d2float(a) __half2float(a)
#define da2float(a) __half2float(a)
#define float2d3(a) __float2half(a)
#define float2d(a) __float2half(a)
#define float2da(a) __float2half(a)
#define float21(a) __float2half(a)
#else
#define DEVICE_SINGLE 1
#define DEVICEA_SINGLE 1
#define HOST_SINGLE 1
#define DICT_SINGLE 1
typedef float dtype1;
typedef float dtype2;
typedef float dtype3;
typedef float dtypea;
typedef float dtypeup;
typedef float dtypeh;
#define d2h(a) (a)
#define d1h(a) (a)
#define d2up(a) (a)
#define d1up(a) (a)
#define daup(a) (a)
#define dup21(a) (a)
#define dup2a(a) (a)
#define dup22(a) (a)
#define d21(a) (a)
#define da1(a) (a)
#define d2a(a) (a)
#define d12(a) (a)
#define h21(a) (a)
#define h2a(a) (a)
#define h2up(a) (a)
#define h23(a) (a)
#define dup2h(a) (a)
#define d2float(a) (a)
#define da2float(a) (a)
#define float2d3(a) (a)
#define float2d(a) (a)
#define float2da(a) (a)
#define float21(a) (a)
#define float2h(a) (a)
#endif

typedef unsigned long long int ix_type;

//#define NUM_LAYERS 6
//#define MAX_SENTENCE_LENGTH 40
//#define BATCH_SIZE 256
//#define WV_LENGTH 300

#define IDX2(i,j,ld) (((j)*(ld))+(i))

const dtypeh zero = 0.0;
const dtypeh one = 1.0;
const dtypeh minus_one = -1.0;

extern std::string datasetsDir;

#endif /* NETLIB_H_ */
