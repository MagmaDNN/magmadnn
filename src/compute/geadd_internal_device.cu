/**
 * @file geadd_internal_device.cu
 * @author Daniel Nichols
 * @version 0.0.1
 * @date 2019-02-22
 *
 * @copyright Copyright (c) 2019
 */
#include "compute/geadd_internal.h"


namespace skepsi {
namespace internal {


template <typename T>
__global__ void kernel_geadd_full_device(unsigned int M, unsigned int N, T alpha, T *A, T beta, T *B, T *C) {

	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int stride = blockDim.x * gridDim.x;

	for (unsigned int i = idx; i < M*N; i += stride) {
		C[i] = alpha*A[i] + beta*B[i];
	}
}

template <typename T>
void geadd_full_device(unsigned int M, unsigned int N, T alpha, T *A, T beta, T *B, T *C) {
	kernel_geadd_full_device <<<M,N>>> (M, N, alpha, A, beta, B, C);
}

template void geadd_full_device(unsigned int M, unsigned int N, int alpha, int *A, int beta, int *B, int *C);
template void geadd_full_device(unsigned int M, unsigned int N, float alpha, float *A, float beta, float *B, float *C);
template void geadd_full_device(unsigned int M, unsigned int N, double alpha, double *A, double beta, double *B, double *C);


}	// namespace internal
}	// namespace skepsi
