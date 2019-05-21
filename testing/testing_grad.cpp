/**
 * @file testing_grad.cpp
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-05-14
 * 
 * @copyright Copyright (c) 2019
 */

#include "magmadnn.h"
#include "utilities.h"

using namespace magmadnn;

void test_simple_grad(memory_t mem, unsigned int size);
void test_full_grad(memory_t mem, unsigned int size);
void test_optimize(memory_t mem, unsigned int size);

int main(int argc, char **argv) {

    
    test_for_all_mem_types(test_simple_grad, 20);
    test_for_all_mem_types(test_full_grad, 20);
    test_for_all_mem_types(test_optimize, 20);

    return 0;
}

void test_simple_grad(memory_t mem, unsigned int size) {
    printf("Testing simple grad on %s...  ", get_memory_type_name(mem));

    /* try to differentiate AX + B wrt to X and B */

    op::Variable<float> *x = op::var<float> ("X", {size, size}, {IDENTITY, {}}, mem);
    op::Variable<float> *a = op::var<float> ("A", {size, size}, {CONSTANT, {5.0}}, mem);
    op::Variable<float> *b = op::var<float> ("B", {size, size}, {CONSTANT, {-1.0}}, mem);

    op::Operation<float> *affine = op::add(op::matmul(a, x), b);

    op::GradTable<float> table;
    magmadnn_error_t err = op::get_grad_table({x,b}, affine, table);
    
    assert( err == 0 );

    op::Operation<float> *affine_wrt_x = table.get(x);
    internal::debugf("affine_wrt_x = %s .\n", affine_wrt_x->to_string().c_str());
    Tensor<float> *res_x = affine_wrt_x->eval();


    op::Operation<float> *affine_wrt_b = table.get(b);
    internal::debugf("affine_wrt_b = %s .\n", affine_wrt_b->to_string().c_str());
    Tensor<float> *res_b = affine_wrt_b->eval();


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