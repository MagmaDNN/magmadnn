/**
 * @file testing_compute_graph.cpp
 * @author Daniel Nichols
 * @version 0.0.1
 * @date 2019-02-18
 * 
 * @copyright Copyright (c) 2019
 */
#include "skepsi.h"

using namespace skepsi;

void test_add(memory_t mem_type, unsigned int size);
const char* get_memory_type_name(memory_t mem);

int main(int argc, char **argv) {

	test_add(HOST, 50);
	#ifdef _HAS_CUDA_
	test_add(DEVICE, 50);
	test_add(MANAGED, 50);
	test_add(CUDA_MANAGED, 50);
	#endif
    
    return 0;
}

void test_add(memory_t mem_type, unsigned int size) {
	float left = 4;
	float right = 6;
	float total = left + right;

	printf("testing %s add...  ", get_memory_type_name(mem_type));

	tensor<float> *t0 = new tensor<float> ({size, size}, {CONSTANT, {left}}, mem_type);
    tensor<float> *t1 = new tensor<float> ({size, size}, {CONSTANT, {right}}, mem_type);

	variable<float> *v0 = new variable<float> ("t0", t0);
	variable<float> *v1 = new variable<float> ("t1", t1);

	auto sum = add_nocopy<float> (v0, v1);

	tensor<float> *fin = sum.eval();

	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			assert( fin->get({i,j}) == total );
		}
	}

	delete t0;
	delete t1;
	delete v0;
	delete v1;

	printf("Success!\n");
}

const char* get_memory_type_name(memory_t mem) {
	switch (mem) {
		case HOST: 			return "HOST";
		#ifdef _HAS_CUDA_
		case DEVICE: 		return "DEVICE";
		case MANAGED: 		return "MANAGED";
		case CUDA_MANAGED: 	return "CUDA_MANAGED";
		#endif
		default: 			return "UNDEFINED_MEMORY_TYPE";
	}
}
