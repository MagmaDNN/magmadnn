/**
 * @file types.h
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-02-07
 *
 * @copyright Copyright (c) 2019
 */
#pragma once

#include <cstdint>
#include <vector>  // for tensor_filler_t

#if defined(MAGMADNN_HAVE_CUDA)
#include "cublas_v2.h"
#include "cudnn.h"
#endif

namespace magmadnn {

typedef enum memory_t {
                       HOST,
#if defined(MAGMADNN_HAVE_CUDA)
                       DEVICE,
                       MANAGED,
                       CUDA_MANAGED
#endif
} memory_t;

typedef unsigned int device_t;
typedef unsigned int magmadnn_error_t;

struct magmadnn_settings_t {
    unsigned int n_devices;
#if defined(MAGMADNN_HAVE_CUDA)
    cudnnHandle_t cudnn_handle;
    cublasHandle_t cublas_handle;
#endif
};
namespace internal {
extern magmadnn_settings_t *MAGMADNN_SETTINGS; /* make this available to everything */
}  // namespace internal

/**	 Different ways to initialize the tensor on creation.
 */
enum tensor_fill_t { UNIFORM, GLOROT, MASK, CONSTANT, ZERO, ONE, DIAGONAL, IDENTITY, NONE };

/** Defines how to fill a tensor and with what parameters.
 * fill_type: use UNIFORM, GLOROT, MASK, CONSTANT, ZERO, ONE, or NONE @see tensor_fill_t
 * values: the parameters for the fill_type
 */
template <typename T>
struct tensor_filler_t {
    tensor_fill_t fill_type;
    std::vector<T> values;
};
   
/**
 * Integral type used for allocation quantities.
 */
using size_type = std::size_t;

/**
 * 8-bit signed integral type.
 */
using int8 = std::int8_t;

/**
 * 16-bit signed integral type.
 */
using int16 = std::int16_t;


/**
 * 32-bit signed integral type.
 */
using int32 = std::int32_t;


/**
 * 64-bit signed integral type.
 */
using int64 = std::int64_t;

/**
 * 8-bit unsigned integral type.
 */
using uint8 = std::uint8_t;

/**
 * 16-bit unsigned integral type.
 */
using uint16 = std::uint16_t;


/**
 * 32-bit unsigned integral type.
 */
using uint32 = std::uint32_t;


/**
 * 64-bit unsigned integral type.
 */
using uint64 = std::uint64_t;


/**
 *
 */
using uintptr = std::uintptr_t;

/**
 * Single precision floating point type.
 */
using float32 = float;


/**
 * Double precision floating point type.
 */
using float64 = double;

}  // namespace magmadnn
