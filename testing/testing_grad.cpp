/**
 * @file testing_grad.cpp
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-05-14
 * 
 * @copyright Copyright (c) 2019
 */

#include <cmath>
#include "magmadnn.h"
#include "utilities.h"
#include "utilities_internal.h"

using namespace magmadnn;

void test_simple_grad(memory_t mem, unsigned int size);
void test_full_grad(memory_t mem, unsigned int size);
void test_optimize(memory_t mem, unsigned int size);

int main(int argc, char **argv) {
    magmadnn_init();
    
    test_for_all_mem_types(test_simple_grad, 20);
    test_for_all_mem_types(test_full_grad, 10);
    test_for_all_mem_types(test_optimize, 20);

    magmadnn_finalize();
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

    sync(res_x);

    for (unsigned int i = 0; i < res_x->get_size(); i++) {
        assert( fequal(res_x->get(i), 5.0) );
    }

    op::Operation<float> *affine_wrt_b = table.get(b);
    internal::debugf("affine_wrt_b = %s .\n", affine_wrt_b->to_string().c_str());
    Tensor<float> *res_b = affine_wrt_b->eval();

    sync(res_b);

    for (unsigned int i = 0; i < res_b->get_size(); i++) {
        assert( fequal(res_b->get(i), 1.0) );
    }

    show_success();
}

void test_full_grad(memory_t mem, unsigned int size) {
    printf("Testing full grad on %s...  ", get_memory_type_name(mem));

    /* compute the grad of sigmoid( 1 - x ) */
    float val = 9.0f;
    float s_x = (1.0f - val) / (1.0f + std::fabs(1.0f - val));
    float out = -1.0f * (s_x) * (1 - s_x);

    op::Operation<float> *one = op::scalar<float> ("1.0", 1.0f, mem);
    op::Operation<float> *x = op::var<float> ("x", {size, size}, {CONSTANT, {val}}, mem);
    op::Operation<float> *expr = op::sigmoid( op::add(one, op::negative(x)) );

    //internal::print_compute_graph(expr, true);


    Tensor<float> *forward = expr->eval();

    sync(forward);

    op::GradTable<float> table;
    magmadnn_error_t err = op::get_grad_table({x}, expr, table);

    assert( err == 0 );

    op::Operation<float> *d_expr_wrt_x = table.get(x);

    internal::print_compute_graph(d_expr_wrt_x, true);

    Tensor<float> *fin = d_expr_wrt_x->eval();

    sync(fin);

    for (unsigned int i = 0; i < fin->get_size(); i++) {
        //printf("%.4g %.4g\n", fin->get(i), out);
        if (!fequal(fin->get(i), out)) printf("bad vals: %.5g %.5g\n", fin->get(i), out);
        assert( fequal(fin->get(i), out) );
    }

    show_success();
}

void test_optimize(memory_t mem, unsigned int size) {
    printf("Testing optimization on %s...  ", get_memory_type_name(mem));

    show_success();
}