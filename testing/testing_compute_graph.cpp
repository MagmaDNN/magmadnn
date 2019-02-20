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
	float val0 = 4;
	float val1 = 6;
	float val2 = 9;
	float total = val0 + val1 + val2;

	printf("testing %s add...  ", get_memory_type_name(mem_type));

	tensor<float> *t0 = new tensor<float> ({size, size}, {CONSTANT, {val0}}, mem_type);
    tensor<float> *t1 = new tensor<float> ({size, size}, {CONSTANT, {val1}}, mem_type);
	tensor<float> *t2 = new tensor<float> ({size, size}, {CONSTANT, {val2}}, mem_type);

	op::variable<float> *v0 = op::var("t0", t0);
	op::variable<float> *v1 = op::var("t1", t1);
	op::variable<float> *v2 = op::var("t2", t2);

	// adds into v1
	auto sum = op::add(v0, op::add(v1, v2));

	tensor<float> *fin = sum->eval();

	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			assert( fin->get({i,j}) == total );
		}
	}

	delete t0;
	delete t1;
	delete t2;

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
