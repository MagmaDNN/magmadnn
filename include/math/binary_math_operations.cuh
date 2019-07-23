/**
 * @file binary_math_operations.cuh
 * @author Daniel Nichols
 * @version 1.0
 * @date 2019-07-22
 *
 * @copyright Copyright (c) 2019
 */

#include "types.h"

namespace magmadnn {
namespace math {
 
#define NEW_BINARY_DEVICE_KERNEL(name, expr)                                         \
    struct name##_map_device {                                                       \
        template <typename Dtype>                                             \
        __device__ __forceinline__ static void map(index_t idx, const Dtype *a, const Dtype *b, Dtype *out) { \
            expr;                                                             \
        }                                                                     \
    }
 
NEW_BINARY_DEVICE_KERNEL(add, out[idx] = a[idx] + b[idx]);
NEW_BINARY_DEVICE_KERNEL(sub, out[idx] = a[idx] - b[idx]);
NEW_BINARY_DEVICE_KERNEL(product, out[idx] = a[idx] * b[idx]);
NEW_BINARY_DEVICE_KERNEL(div, out[idx] = a[idx] / b[idx]);
 
NEW_BINARY_DEVICE_KERNEL(dive, out[idx] = std::modf(a[idx], b[idx]));
NEW_BINARY_DEVICE_KERNEL(pow, out[idx] = std::pow(a[idx], b[idx]));
 
#undef NEW_BINARY_DEVICE_KERNEL

#define NEW_SCALAR_TENSOR_DEVICE_KERNEL(name, expr) \
    struct name##_map_device { \
        template <typename Dtype> \
        __device__ __forceinline__ static void map(index_t idx, const Dtype a, const Dtype *b, Dtype *out) { \
            expr; \
        } \
    }

NEW_SCALAR_TENSOR_DEVICE_KERNEL(scalar_tensor_product, out[idx] = a * b[idx]);
NEW_SCALAR_TENSOR_DEVICE_KERNEL(scalar_tensor_add, out[idx] = a + b[idx]);

#undef NEW_SCALAR_TENSOR_DEVICE_KERNEL
 
}  // namespace math
}  // namespace magmadnn