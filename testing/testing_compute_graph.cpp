/**
 * @file testing_compute_graph.cpp
 * @author Daniel Nichols
 * @version 0.0.1
 * @date 2019-02-18
 * 
 * @copyright Copyright (c) 2019
 */
#include "magmadnn.h"
#include "utilities.h"

using namespace magmadnn;

void test_add(memory_t mem_type, unsigned int size);
void test_sum(memory_t mem_type, unsigned int size);
void test_matmul(memory_t mem_type, unsigned int size);
void test_affine(memory_t mem_type, unsigned int size);
void test_sigmoid(memory_t mem_type, unsigned int size);
void test_tanh(memory_t mem_type, unsigned int size);

int main(int argc, char **argv) {
	magmadnn_init();

	// test add
	test_for_all_mem_types(test_add, 50);

	// test sum
	test_for_all_mem_types(test_sum, 6);

	// test matmul
	test_matmul(HOST, 50);
	#if defined(_HAS_CUDA_)
	test_matmul(DEVICE, 50);
	test_matmul(MANAGED, 50);
	test_matmul(CUDA_MANAGED, 50);
	#endif

	// test affine transformation
	test_affine(HOST, 50);
	#if defined(_HAS_CUDA_)
	test_affine(DEVICE, 50);
	test_affine(MANAGED, 50);
	test_affine(CUDA_MANAGED, 50);
	#endif

	// test sigmoid
	test_sigmoid(HOST, 50);
	#if defined(_HAS_CUDA_)
	test_sigmoid(DEVICE, 50);
	test_sigmoid(MANAGED, 50);
	test_sigmoid(CUDA_MANAGED, 50);
	#endif

	// test tanh
	test_tanh(HOST, 50);
	#if defined(_HAS_CUDA_)
	test_tanh(DEVICE, 50);
	test_tanh(MANAGED, 50);
	test_tanh(CUDA_MANAGED, 50);
	#endif
    
	magmadnn_finalize();
    return 0;
}

void test_add(memory_t mem_type, unsigned int size) {
	float val0 = 4;
	float val1 = 6;
	float val2 = 9;
	float total = val0 + val1 + val2;

	printf("testing %s add...  ", get_memory_type_name(mem_type));

	Tensor<float> *t0 = new Tensor<float> ({size, size}, {CONSTANT, {val0}}, mem_type);
    Tensor<float> *t1 = new Tensor<float> ({size, size}, {CONSTANT, {val1}}, mem_type);
	Tensor<float> *t2 = new Tensor<float> ({size, size}, {CONSTANT, {val2}}, mem_type);

	op::Variable<float> *v0 = op::var("t0", t0);
	op::Variable<float> *v1 = op::var("t1", t1);
	op::Variable<float> *v2 = op::var("t2", t2);

	// adds into v1
	auto sum = op::add(v0, op::add(v1, v2));

	Tensor<float> *fin = sum->eval();

	#if defined(_HAS_CUDA_)
	if (mem_type == DEVICE || mem_type == CUDA_MANAGED) fin->get_memory_manager()->sync();
	if (mem_type == MANAGED) fin->get_memory_manager()->sync(true);
	#endif

	for (int i = 0; i < (int)size; i++) {
		for (int j = 0; j < (int)size; j++) {
			assert( fin->get({i,j}) == total );
		}
	}

	delete t0;
	delete t1;
	delete t2;
	delete sum;

	show_success();
}

void test_sum(memory_t mem_type, unsigned int size) {
	float val0 = 1.5, val1 = 2.0, val2 = -1.2, val3 = 3.275;
	float total = val0 + val1 + val2 + val3;

	printf("Testing %s sum...  ", get_memory_type_name(mem_type));

	op::Variable<float> *v0 = op::var<float>("v0", {size, size, size}, {CONSTANT, {val0}}, mem_type);
	op::Variable<float> *v1 = op::var<float>("v1", {size, size, size}, {CONSTANT, {val1}}, mem_type);
	op::Variable<float> *v2 = op::var<float>("v2", {size, size, size}, {CONSTANT, {val2}}, mem_type);
	op::Variable<float> *v3 = op::var<float>("v3", {size, size, size}, {CONSTANT, {val3}}, mem_type);

	op::Operation<float> *sum = op::sum<float>({v0, v1, v2, v3});
	Tensor<float> *fin = sum->eval();

	#if defined(_HAS_CUDA_)
	if (mem_type == DEVICE || mem_type == CUDA_MANAGED) fin->get_memory_manager()->sync();
	if (mem_type == MANAGED) fin->get_memory_manager()->sync(true);
	#endif

	for (int x = 0; x < size; x++) {
		for (int y = 0; y < size; y++) {
			for (int z = 0; z < size; z++) {
				assert( fequal(fin->get({x,y,z}), total) );
			}
		}
	}

	delete sum;

	show_success();
}

void test_matmul(memory_t mem_type, unsigned int size) {
	unsigned int m = size;
	unsigned int n = size;
	unsigned int p = size+5;
	float val = 5;

	printf("Testing %s matmul...  ", get_memory_type_name(mem_type));

	Tensor<float> *t0 = new Tensor<float> ({m,n}, {ZERO, {}}, mem_type);
	Tensor<float> *t1 = new Tensor<float> ({n,p}, {CONSTANT, {5}}, mem_type);

	/* make t0 identity matrix */
	for (int i = 0; i < (int) m; i++)
		for (int j = 0; j < (int) n; j++)
			if (i==j) t0->set({i,j}, 1);

	op::Variable<float> *v0 = op::var("t0", t0);
	op::Variable<float> *v1 = op::var("t1", t1);

	auto prod = op::matmul(v0, v1);

	Tensor<float> *fin = prod->eval();

	#if defined(_HAS_CUDA_)
	if (mem_type == DEVICE || mem_type == CUDA_MANAGED) fin->get_memory_manager()->sync();
	if (mem_type == MANAGED) fin->get_memory_manager()->sync(true);
	#endif

	for (int i = 0; i < (int) m; i++) {
		for (int j = 0; j < (int) p; j++) {
			assert( fin->get({i,j}) == val );
		}
	}

	delete t0;
	delete t1;
	delete prod;

	show_success();
}

void test_affine(memory_t mem_type, unsigned int size) {
	unsigned int m = size;
	unsigned int n = size;
	unsigned int p = size+5;
	float val = 5;
	float b = 12.5;

	printf("Testing %s affine...  ", get_memory_type_name(mem_type));

	Tensor<float> *t0 = new Tensor<float> ({m,n}, {ZERO, {}}, mem_type);
	Tensor<float> *t1 = new Tensor<float> ({n,p}, {CONSTANT, {5}}, mem_type);
	Tensor<float> *t2 = new Tensor<float> ({m,p}, {CONSTANT, {b}}, mem_type);

	/* make t0 identity matrix */
	for (int i = 0; i < (int) m; i++)
		for (int j = 0; j < (int) n; j++)
			if (i==j) t0->set({i,j}, 1);

	op::Variable<float> *v0 = op::var("t0", t0);
	op::Variable<float> *v1 = op::var("t1", t1);
	op::Variable<float> *v2 = op::var("t2", t2);

	auto aff = op::add(op::matmul(v0, v1), v2);

	Tensor<float> *fin = aff->eval();

	#if defined(_HAS_CUDA_)
	if (mem_type == DEVICE || mem_type == CUDA_MANAGED) fin->get_memory_manager()->sync();
	if (mem_type == MANAGED) fin->get_memory_manager()->sync(true);
	#endif

	for (int i = 0; i < (int) m; i++) {
		for (int j = 0; j < (int) p; j++) {
			assert( fin->get({i,j}) == (val+b) );
		}
	}

	delete t0;
	delete t1;
	delete t2;
	delete aff;

	show_success();
}

void test_sigmoid(memory_t mem_type, unsigned int size) {

	printf("Testing %s sigmoid...  ", get_memory_type_name(mem_type));

	Tensor<float> *t0 = new Tensor<float> ({size, size}, {CONSTANT, {-7}}, mem_type);

	auto v0 = op::var("t0", t0);

	auto sig = op::sigmoid(v0);

	auto fin = sig->eval();

	#if defined(_HAS_CUDA_)
	if (mem_type == DEVICE || mem_type == CUDA_MANAGED) fin->get_memory_manager()->sync();
	else if (mem_type == MANAGED) fin->get_memory_manager()->sync(true);
	#endif

	for (unsigned int i = 0; i < fin->get_size(); i++) {
		assert( fabs(fin->get(i) - (-0.875)) < 1E-8 );
	}

	delete t0;
	delete sig;

	show_success();
}

void test_tanh(memory_t mem_type, unsigned int size) {

	float val = 5.0;

	printf("Testing %s tanh...  ", get_memory_type_name(mem_type));

	Tensor<float> *t0 = new Tensor<float> ({size, size}, {CONSTANT, {val}}, mem_type);

	auto v0 = op::var("t0", t0);

	auto fin_op = op::tanh(v0);

	auto fin = fin_op->eval();

	#if defined(_HAS_CUDA_)
	if (mem_type == DEVICE || mem_type == CUDA_MANAGED) fin->get_memory_manager()->sync();
	else if (mem_type == MANAGED) fin->get_memory_manager()->sync(true);
	#endif

	for (unsigned int i = 0; i < fin->get_size(); i++) {
		assert( fabs(fin->get(i) - tanh(val)) < 1E-6 );
	}

	delete t0;
	delete fin_op;
	delete fin;

	show_success();
}

