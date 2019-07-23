/**
 * @file unary_math_operations.h
 * @author Daniel Nichols
 * @version 1.0
 * @date 2019-07-23
 *
 * @copyright Copyright (c) 2019
 */
#pragma once

#include "types.h"

namespace magmadnn {
namespace math {

#define NEW_UNARY_KERNEL(name, expr)                              \
    struct name##_map {                                           \
        template <typename T>                                     \
        inline static void map(index_t idx, const T *a, T *out) { \
            expr;                                                 \
        }                                                         \
    }

NEW_UNARY_KERNEL(neg, out[idx] = a[idx]);
NEW_UNARY_KERNEL(ReLU, out[idx] = (a[idx] > static_cast<T>(0)) : a[idx] : static_cast<T>(0));

}  // namespace math
}  // namespace magmadnn