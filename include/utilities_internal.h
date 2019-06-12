/**
 * @file utilities_internal.h
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-05-21
 * 
 * @copyright Copyright (c) 2019
 */
#pragma once

#include <cstdio>
#include <cstdarg>
#include <cstdint>
#include <vector>
#include <set>
#include <deque>

#if defined(_HAS_CUDA_)
#include "cudnn_v7.h"
#endif



#define T_IS_SCALAR(tensor_ptr) ((tensor_ptr)->get_size() == 1)
#define T_IS_VECTOR(tensor_ptr) ((tensor_ptr)->get_size() != 1 && ((tensor_ptr)->get_shape().size() == 1))
#define T_IS_MATRIX(tensor_ptr) ((tensor_ptr)->get_shape().size() == 2)
#define T_IS_N_DIMENSIONAL(tensor_ptr, N) ((tensor_ptr)->get_shape().size() == N)
#define OP_IS_SCALAR(op_ptr) ((op_ptr)->get_output_size() == 1)
#define OP_IS_VECTOR(op_ptr) (((op_ptr)->get_output_size() != 1) && ((op_ptr)->get_output_shape().size() == 1))
#define OP_IS_MATRIX(op_ptr) ((op_ptr)->get_output_shape().size() == 2)
#define OP_IS_N_DIMENSIONAL(op_ptr, N) ((op_ptr)->get_output_shape().size() == N)


namespace magmadnn {
namespace internal {

/** Only prints #if DEBUG macro is defined.
 * @param fmt 
 * @param ... 
 */
int debugf(const char *fmt, ...);


void print_vector(const std::vector<unsigned int>& vec, bool debug=true, char begin='{', char end='}', char delim=',');

/* 
template <typename T>
void print_tensor(const Tensor<T>& t, bool print_flat=false, bool debug=true, const char *begin="{", const char *end="}\n", const char *delim=", ");

template <typename T>
void print_compute_graph(op::Operation<T> *node, bool debug=true);
*/

#if defined(_HAS_CUDA_)
template <typename T>
cudnnDataType_t get_cudnn_data_type(T val);
#endif

}   // namespace internal
}   // namespace magmadnn