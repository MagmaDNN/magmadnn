/**
 * @file types.h
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-02-07
 * 
 * @copyright Copyright (c) 2019
 */
#pragma once


typedef enum memory_t {
	DEVICE,
	HOST,
	MANAGED,
	CUDA_MANAGED
} memory_t;

typedef unsigned int device_t;
typedef unsigned int skepsi_error_t;