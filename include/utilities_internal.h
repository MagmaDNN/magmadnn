/**
 * @file utilities_internal.h
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-05-21
 *
 * @copyright Copyright (c) 2019
 */
#pragma once

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <set>
#include <vector>
#include "data_types.h"

#if defined(_HAS_CUDA_)
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include "cudnn.h"

#define cudaErrchk(ans) \
    { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#define cudnnErrchk(ans) \
    { cudnnAssert((ans), __FILE__, __LINE__); }
inline void cudnnAssert(cudnnStatus_t code, const char *file, int line, bool abort = true) {
    if (code != CUDNN_STATUS_SUCCESS) {
        fprintf(stderr, "CuDNNassert: %s %s %d\n", cudnnGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#define curandErrchk(ans) \
    { curandAssert((ans), __FILE__, __LINE__); }
inline void curandAssert(curandStatus_t code, const char *file, int line, bool abort = true) {
    if (code != CURAND_STATUS_SUCCESS) {
        fprintf(stderr, "CuRandAssert: %d %s %d\n", code, file, line);
    }
}

#endif

#define T_IS_SCALAR(tensor_ptr) ((tensor_ptr)->get_size() == 1)
#define T_IS_VECTOR(tensor_ptr) ((tensor_ptr)->get_size() != 1 && ((tensor_ptr)->get_shape().size() == 1))
#define T_IS_MATRIX(tensor_ptr) ((tensor_ptr)->get_shape().size() == 2)
#define T_IS_N_DIMENSIONAL(tensor_ptr, N) ((tensor_ptr)->get_shape().size() == N)
#define OP_IS_SCALAR(op_ptr) ((op_ptr)->get_output_size() == 1)
#define OP_IS_VECTOR(op_ptr) (((op_ptr)->get_output_size() != 1) && ((op_ptr)->get_output_shape().size() == 1))
#define OP_IS_MATRIX(op_ptr) ((op_ptr)->get_output_shape().size() == 2)
#define OP_IS_N_DIMENSIONAL(op_ptr, N) ((op_ptr)->get_output_shape().size() == N)

/* TODO -- change T_IS_SAME_MEMORY_TYPE to T_PTR_IS_SAME_MEMORY_TYPE
    not changed yet, so as not to break things
*/
#define T_IS_SAME_MEMORY_TYPE(x_ptr, y_ptr) ((x_ptr)->get_memory_type() == (y_ptr)->get_memory_type())
#define T_SAME_MEM_TYPE(x, y) ((x).get_memory_type() == (y).get_memory_type())
#define OP_IS_SAME_MEMORY_TYPE(x_ptr, y_ptr) ((x_ptr)->get_memory_type() == (y_ptr)->get_memory_type())

#define T_IS_SAME_DTYPE(x, y) ((x).dtype() == (y).dtype())
#define TYPES_MATCH(type, type_enum) (GetDataType<type>::value == type_enum)

#if defined(NDEBUG)
#define MAGMADNN_ASSERT(ans, message) ((void) 0)
#else
#define MAGMADNN_ASSERT(ans, message) \
    { magmadnn_assert_((ans), (message), __FILE__, __LINE__, __func__); }
#endif

inline void magmadnn_assert_(bool ans, const char *message, const char *file, int line, const char *func,
                             bool abort = true) {
    if (!ans) {
        std::cerr << "[ERROR](" << file << ":" << line << " in " << func << ") -- " << message << "\n";
        if (abort) std::exit(1);
    }
}

#define LOG(type) magmadnn_log_(#type, __FILE__, __LINE__)
inline std::ostream &magmadnn_log_(const std::string &type, const char *file, int line) {
    if (type == "ERROR") {
        return std::cerr << "[ERROR](" << file << ":" << line << ") -- ";
    } else if (type == "OK") {
        return std::cout << "[OK](" << file << ":" << line << ") -- ";
    } else {
        return std::cerr << "INVALID LOG TYPE ---- ";
    }
}

namespace magmadnn {
namespace internal {

/** Only prints #if DEBUG macro is defined.
 * @param fmt
 * @param ...
 */
int debugf(const char *fmt, ...);

void print_vector(const std::vector<unsigned int> &vec, bool debug = true, char begin = '{', char end = '}',
                  char delim = ',');

/*
template <typename T>
void print_tensor(const Tensor<T>& t, bool print_flat=false, bool debug=true, const char *begin="{", const char
*end="}\n", const char *delim=", ");

template <typename T>
void print_compute_graph(op::Operation<T> *node, bool debug=true);
*/

#if defined(_HAS_CUDA_)
template <typename T>
cudnnDataType_t get_cudnn_data_type(T val);
#endif

}  // namespace internal
}  // namespace magmadnn