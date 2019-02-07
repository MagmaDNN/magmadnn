#pragma once


typedef enum memory_t {
	DEVICE,
	HOST,
	MANAGED,
	CUDA_MANAGED
} memory_t;

typedef unsigned int device_t;


typedef unsigned int error_t;