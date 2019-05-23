/**
 * @file utilities.h
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-03-14
 * 
 * @copyright Copyright (c) 2019
 */

#pragma once

#include "magmadnn.h"

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"

void show_success() {
    printf(ANSI_COLOR_GREEN "Success!" ANSI_COLOR_RESET "\n");
}

bool fequal(float a, float b) {
	return (fabs(a - b) <= 1E-8);
}

void test_for_all_mem_types(void (*tester)(magmadnn::memory_t, unsigned int), unsigned int param) {
	tester(magmadnn::HOST, param);
	#if defined(_HAS_CUDA_)
	tester(magmadnn::DEVICE, param);
	tester(magmadnn::MANAGED, param);
	tester(magmadnn::CUDA_MANAGED, param);
	#endif
}

const char* get_memory_type_name(magmadnn::memory_t mem) {
	switch (mem) {
		case magmadnn::HOST: 			return "HOST";
		#if defined(_HAS_CUDA_)
		case magmadnn::DEVICE: 		return "DEVICE";
		case magmadnn::MANAGED: 		return "MANAGED";
		case magmadnn::CUDA_MANAGED: 	return "CUDA_MANAGED";
		#endif
		default: 			return "UNDEFINED_MEMORY_TYPE";
	}
}

template <typename T>
void sync(magmadnn::Tensor<T> *t) {
	#if defined(_HAS_CUDA_)
	if (t->get_memory_type() == magmadnn::DEVICE || t->get_memory_type() == magmadnn::CUDA_MANAGED) t->get_memory_manager()->sync();
	else if (t->get_memory_type() == magmadnn::MANAGED) t->get_memory_manager()->sync(true);
	#endif
}