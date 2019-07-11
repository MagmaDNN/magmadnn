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

    op::Variable<float> *x = op::var<float>("X", {size, size}, {IDENTITY, {}}, mem);
    op::Variable<float> *a = op::var<float>("A", {size, size}, {CONSTANT, {5.0}}, mem);
    op::Variable<float> *b = op::var<float>("B", {size, size}, {CONSTANT, {-1.0}}, mem);

    op::Operation<float> *affine = op::add(op::matmul(a, x), b);

    op::GradTable<float> table;
    magmadnn_error_t err = op::get_grad_table({x, b}, affine, table);

    MAGMADNN_TEST_ASSERT_DEFAULT(err == 0, "\"err == 0\" failed");

    Tensor<float> *affine_wrt_x = table.get(x);

    sync(affine_wrt_x);

    for (unsigned int i = 0; i < affine_wrt_x->get_size(); i++) {
        if (!fequal(affine_wrt_x->get(i), 5.0f)) printf("bad vals: %.5g %.5g\n", affine_wrt_x->get(i), 5.0f);
        MAGMADNN_TEST_ASSERT_FEQUAL_DEFAULT(affine_wrt_x->get(i), 5.0);
    }

    Tensor<float> *affine_wrt_b = table.get(b);

    sync(affine_wrt_b);

    for (unsigned int i = 0; i < affine_wrt_b->get_size(); i++) {
        MAGMADNN_TEST_ASSERT_FEQUAL_DEFAULT(affine_wrt_b->get(i), 1.0);
    }

    delete affine;

    show_success();
}

void test_full_grad(memory_t mem, unsigned int size) {
    printf("Testing full grad on %s...  ", get_memory_type_name(mem));

    /* compute the grad of sigmoid( 1 - x ) */
    float val = 9.0f;
    float s_x = (1.0f - val) / (1.0f + std::fabs(1.0f - val));
    float out = -1.0f * (s_x) * (1 - s_x);

    op::Operation<float> *one = op::scalar<float>("1.0", 1.0f, mem);
    op::Operation<float> *x = op::var<float>("x", {size, size}, {CONSTANT, {val}}, mem);
    op::Operation<float> *expr = op::sigmoid(op::add(one, op::negative(x)), true, true);

    Tensor<float> *forward = expr->eval();

    sync(forward);

    op::GradTable<float> table;
    magmadnn_error_t err = op::get_grad_table({x}, expr, table);

    MAGMADNN_TEST_ASSERT_DEFAULT(err == 0, "\"err == 0\" failed");

    Tensor<float> *d_expr_wrt_x = table.get(x);

    sync(d_expr_wrt_x);

    for (unsigned int i = 0; i < d_expr_wrt_x->get_size(); i++) {
        if (!fequal(d_expr_wrt_x->get(i), out)) std::printf("bad vals: %.5g %.5g\n", d_expr_wrt_x->get(i), out);
        MAGMADNN_TEST_ASSERT_FEQUAL_DEFAULT(d_expr_wrt_x->get(i), out);
    }

    delete expr;

    show_success();
}

void test_optimize(memory_t mem, unsigned int size) {
    printf("Testing optimization on %s...  ", get_memory_type_name(mem));

    /* optimizing x^2 + c */
    double x_val = 9.0;
    double c_val = -5.0;
    unsigned int n_iter = 50;

    op::Operation<double> *x = op::var<double>("x", {size}, {CONSTANT, {x_val}}, mem);
    op::Operation<double> *c = op::var<double>("c", {size}, {CONSTANT, {c_val}}, mem);
    op::Operation<double> *expr = op::add(op::pow(x, 2), c);

    optimizer::GradientDescent<double> optim(0.2f);

    for (unsigned int i = 0; i < n_iter; i++) {
        expr->eval(true);

        optim.minimize(expr, {x});
    }

    Tensor<double> *x_tensor = x->get_output_tensor();
    sync(x_tensor);

    for (unsigned int i = 0; i < size; i++) {
        MAGMADNN_TEST_ASSERT_FEQUAL_DEFAULT(x_tensor->get(i), 0.0f);
    }

    delete expr;

    show_success();
}