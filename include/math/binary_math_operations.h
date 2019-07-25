/**
 * @file binary_math_operations.h
 * @author Daniel Nichols
 * @version 1.0
 * @date 2019-07-22
 *
 * @copyright Copyright (c) 2019
 */
#pragma once
#include <cmath>

#include "data_types.h"
#include "types.h"

namespace magmadnn {
namespace math {

#define NEW_BINARY_KERNEL(name, expr)                                                             \
    struct name##_map {                                                                           \
        template <typename Dtype>                                                                 \
        GENERIC_INLINE static void map(index_t idx, const Dtype *a, const Dtype *b, Dtype *out) { \
            expr;                                                                                 \
        }                                                                                         \
    }

NEW_BINARY_KERNEL(add, out[idx] = a[idx] + b[idx]);
NEW_BINARY_KERNEL(sub, out[idx] = a[idx] - b[idx]);
NEW_BINARY_KERNEL(product, out[idx] = a[idx] * b[idx]);
NEW_BINARY_KERNEL(div, out[idx] = a[idx] / b[idx]);

NEW_BINARY_KERNEL(pow, out[idx] = std::pow(a[idx], b[idx]));

#undef NEW_BINARY_KERNEL

#define NEW_SCALAR_TENSOR_KERNEL(name, expr)                                                     \
    struct name##_map {                                                                          \
        template <typename Dtype>                                                                \
        GENERIC_INLINE static void map(index_t idx, const Dtype a, const Dtype *b, Dtype *out) { \
            expr;                                                                                \
        }                                                                                        \
    }

NEW_SCALAR_TENSOR_KERNEL(scalar_tensor_product, out[idx] = a * b[idx]);
NEW_SCALAR_TENSOR_KERNEL(scalar_tensor_add, out[idx] = a + b[idx]);

#undef NEW_SCALAR_TENSOR_KERNEL

}  // namespace math
}  // namespace magmadnn