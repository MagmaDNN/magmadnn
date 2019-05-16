/**
 * @file testing_grad.cpp
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-05-14
 * 
 * @copyright Copyright (c) 2019
 */

#include "skepsi.h"
#include "utilities.h"

using namespace skepsi;

void test_simple_grad(memory_t mem, unsigned int size);
void test_full_grad(memory_t mem, unsigned int size);
void test_optimize(memory_t mem, unsigned int size);

int main(int argc, char **argv) {

    test_simple_grad(HOST, 20);
    #if defined(_HAS_CUDA_)
    test_simple_grad(CUDA, 20);
    test_simple_grad(MANAGED, 20);
    test_simple_grad(CUDA_MANAGED, 20);
    #endif

    test_full_grad(HOST, 20);
    #if defined(_HAS_CUDA_)
    test_full_grad(CUDA, 20);
    test_full_grad(MANAGED, 20);
    test_full_grad(CUDA_MANAGED, 20);
    #endif

    test_optimize(HOST, 20);
    #if defined(_HAS_CUDA_)
    test_optimize(CUDA, 20);
    test_optimize(MANAGED, 20);
    test_optimize(CUDA_MANAGED, 20);
    #endif

    return 0;
}

void test_simple_grad(memory_t mem, unsigned int size) {
    printf("Testing simple grad on %s...  ", get_memory_type_name(mem));

    op::Variable<float>* x = op::var<float>("x", {size, size}, {GLOROT, {0.0,1.0}}, mem);


    show_success();
}

void test_full_grad(memory_t mem, unsigned int size) {
    printf("Testing full grad on %s...  ", get_memory_type_name(mem));

    show_success();
}

void test_optimize(memory_t mem, unsigned int size) {
    printf("Testing optimization on %s...  ", get_memory_type_name(mem));

    show_success();
}