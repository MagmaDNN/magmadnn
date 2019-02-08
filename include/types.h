/**
 * @file types.h
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-02-07
 * 
 * @copyright Copyright (c) 2019
 */
#pragma once

namespace skepsi {

typedef enum memory_t {
	HOST,
	#ifdef _HAS_CUDA_
	DEVICE,
	MANAGED,
	CUDA_MANAGED
	#endif
} memory_t;

typedef unsigned int device_t;
typedef unsigned int skepsi_error_t;

} // namespace skepsi