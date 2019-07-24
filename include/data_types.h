/**
 * @file data_types.h
 * @author Daniel Nichols
 * @version 1.0
 * @date 2019-07-19
 *
 * @copyright Copyright (c) 2019
 */
#pragma once

#include <cstdint>
#include <iostream>

namespace magmadnn {

enum DataType { INVALID = 0, FLOAT = 1, DOUBLE = 2, INT32 = 3, INT64 = 4, HALF = 5 };

/* typing model inspired by github.com/tensorflow/tensorflow/core/types */

template <typename T>
struct IsValidDataType {};

template <typename T>
struct GetDataType {
    static_assert(IsValidDataType<T>::value, "Invalid Data Type\n");
};

template <DataType dtype>
struct GetCppType {};

#define COMBINE_CPP_TYPE_AND_ENUM_TYPE(TYPE, ENUM) \
    template <>                                    \
    struct GetDataType<TYPE> {                     \
        static DataType val() { return ENUM; }     \
        static constexpr DataType value = ENUM;    \
    };                                             \
    template <>                                    \
    struct IsValidDataType<TYPE> {                 \
        static constexpr bool value = true;        \
    };                                             \
    template <>                                    \
    struct GetCppType<ENUM> {                      \
        typedef TYPE Type;                         \
    }

COMBINE_CPP_TYPE_AND_ENUM_TYPE(float, FLOAT);
COMBINE_CPP_TYPE_AND_ENUM_TYPE(double, DOUBLE);
COMBINE_CPP_TYPE_AND_ENUM_TYPE(int32_t, INT32);
COMBINE_CPP_TYPE_AND_ENUM_TYPE(int64_t, INT64);

#undef COMBINE_CPP_TYPE_AND_ENUM_TYPE

#define CALL_FOR_ALL_TYPES(func) func(float) func(double) func(int32_t) func(int64_t)
#define CALL_FOR_ALL_FLOAT_TYPES(func) func(float) func(double)
#define CALL_FOR_ALL_INT_TYPES(func) func(int32_t) func(int64_t)

inline int getDataTypeSize(DataType dtype) {
#define EACH(type)                 \
    case GetDataType<type>::value: \
        return sizeof(type);

    switch (dtype) {
        CALL_FOR_ALL_TYPES(EACH);

        default:
            return 0;
    }
#undef EACH
}

#define FOR_ALL_DTYPES(enum_type, Dtype, ...)                             \
    switch (enum_type) {                                                  \
        case FLOAT: {                                                     \
            typedef float Dtype;                                          \
            { __VA_ARGS__ }                                               \
        } break;                                                          \
        case DOUBLE: {                                                    \
            typedef double Dtype;                                         \
            { __VA_ARGS__ }                                               \
        } break;                                                          \
        case INT32: {                                                     \
            typedef int32_t Dtype;                                        \
            { __VA_ARGS__ }                                               \
        } break;                                                          \
        case INT64: {                                                     \
            typedef int64_t Dtype;                                        \
            { __VA_ARGS__ }                                               \
        } break;                                                          \
        default:                                                          \
            std::cerr << "Unsupported data type:  " << enum_type << "\n"; \
    }

#if defined(GENERIC_INLINE)
#error "GENERIC_INLINE macro is used by magmadnn"
#else
#if defined(__CUDACC__) && defined(_HAS_CUDA_)
#define GENERIC_INLINE inline __attribute__((always_inline)) __device__ __host__
#else
#define GENERIC_INLINE inline __attribute__((always_inline))
#endif
#endif

}  // namespace magmadnn