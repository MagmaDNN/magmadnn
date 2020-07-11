/**
 * @file testing_compute_graph.cpp
 * @author Daniel Nichols
 * @version 1.0
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
void test_transpose(memory_t mem_type, unsigned int size);
void test_log(memory_t mem_type, unsigned int size);
void test_product(memory_t mem_type, unsigned int size);
void test_scalarproduct(memory_t mem_type, unsigned int size);
void test_softmax(memory_t mem_type, unsigned int size);
void test_sumreduce(memory_t mem_type, unsigned int);
void test_affine(memory_t mem_type, unsigned int size);
void test_sigmoid(memory_t mem_type, unsigned int size);
void test_tanh(memory_t mem_type, unsigned int size);
void test_conv2d(memory_t mem_type, unsigned int size);
void test_pooling(memory_t mem_type, unsigned int size);
void test_batchnorm(memory_t mem_type, unsigned int size);
void test_crossentropy(memory_t mem_type, unsigned int size);
void test_meansquarederror(memory_t mem_type, unsigned int size);

int main(int argc, char **argv) {
    magmadnn_init();

    // test add
    test_for_all_mem_types(test_add, 50);
    test_for_all_mem_types(test_sum, 6);
    test_for_all_mem_types(test_matmul, 50);
    test_for_all_mem_types(test_transpose, 100);
    test_for_all_mem_types(test_log, 5);
    test_for_all_mem_types(test_product, 50);
    test_for_all_mem_types(test_scalarproduct, 10);
    test_for_all_mem_types(test_softmax, 10);
    test_for_all_mem_types(test_sumreduce, 10);
    test_for_all_mem_types(test_affine, 50);
    test_for_all_mem_types(test_sigmoid, 50);
    test_for_all_mem_types(test_tanh, 50);

#if defined(MAGMADNN_HAVE_CUDA)
    test_conv2d(DEVICE, 30);
    test_conv2d(MANAGED, 30);
    test_conv2d(CUDA_MANAGED, 30);

    test_pooling(DEVICE, 30);
    test_pooling(MANAGED, 30);
    test_pooling(CUDA_MANAGED, 30);

    test_batchnorm(DEVICE, 30);
    test_batchnorm(MANAGED, 30);
    test_batchnorm(CUDA_MANAGED, 30);
#endif

    test_for_all_mem_types(test_crossentropy, 10);
    // test_meansquarederror(HOST, 10);
    test_for_all_mem_types(test_meansquarederror, 10);

    magmadnn_finalize();
    return 0;
}

void test_add(memory_t mem_type, unsigned int size) {
    float val0 = 4;
    float val1 = 6;
    float val2 = 9;
    float total = val0 + val1 + val2;

    printf("testing %s add...  ", get_memory_type_name(mem_type));

    Tensor<float> *t0 = new Tensor<float>({size, size}, {CONSTANT, {val0}}, mem_type);
    Tensor<float> *t1 = new Tensor<float>({size, size}, {CONSTANT, {val1}}, mem_type);
    Tensor<float> *t2 = new Tensor<float>({size, size}, {CONSTANT, {val2}}, mem_type);

    op::Variable<float> *v0 = op::var("t0", t0);
    op::Variable<float> *v1 = op::var("t1", t1);
    op::Variable<float> *v2 = op::var("t2", t2);

    // adds into v1
    auto sum = op::add(v0, op::add(v1, v2));

    Tensor<float> *fin = sum->eval();

#if defined(MAGMADNN_HAVE_CUDA)
    if (mem_type == DEVICE || mem_type == CUDA_MANAGED) fin->get_memory_manager()->sync();
    if (mem_type == MANAGED) fin->get_memory_manager()->sync(true);
#endif

    for (int i = 0; i < (int) size; i++) {
        for (int j = 0; j < (int) size; j++) {
            MAGMADNN_TEST_ASSERT_DEFAULT(fin->get({i, j}) == total, "\"fin->get({i, j}) == total\" failed");
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

#if defined(MAGMADNN_HAVE_CUDA)
    if (mem_type == DEVICE || mem_type == CUDA_MANAGED) fin->get_memory_manager()->sync();
    if (mem_type == MANAGED) fin->get_memory_manager()->sync(true);
#endif

    for (int x = 0; x < (int) size; x++) {
        for (int y = 0; y < (int) size; y++) {
            for (int z = 0; z < (int) size; z++) {
                MAGMADNN_TEST_ASSERT_FEQUAL_DEFAULT(fin->get({x, y, z}), total);
            }
        }
    }

    delete sum;

    show_success();
}

void test_matmul(memory_t mem_type, unsigned int size) {
    unsigned int m = size;
    unsigned int n = size;
    unsigned int p = size + 5;
    float val = 5;

    printf("Testing %s matmul...  ", get_memory_type_name(mem_type));

    Tensor<float> *t0 = new Tensor<float>({m, n}, {ZERO, {}}, mem_type);
    Tensor<float> *t1 = new Tensor<float>({n, p}, {CONSTANT, {5}}, mem_type);

    /* make t0 identity matrix */
    for (int i = 0; i < (int) m; i++)
        for (int j = 0; j < (int) n; j++)
            if (i == j) t0->set({i, j}, 1);

    op::Variable<float> *v0 = op::var("t0", t0);
    op::Variable<float> *v1 = op::var("t1", t1);

    auto prod = op::matmul(v0, v1);

    Tensor<float> *fin = prod->eval();

#if defined(MAGMADNN_HAVE_CUDA)
    if (mem_type == DEVICE || mem_type == CUDA_MANAGED) fin->get_memory_manager()->sync();
    if (mem_type == MANAGED) fin->get_memory_manager()->sync(true);
#endif

    for (int i = 0; i < (int) m; i++) {
        for (int j = 0; j < (int) p; j++) {
            MAGMADNN_TEST_ASSERT_DEFAULT(fin->get({i, j}) == val, "\"fin->get({i, j}) == val\" failed");
        }
    }

    delete t0;
    delete t1;
    delete prod;

    show_success();
}

void test_transpose(memory_t mem, unsigned int size) {
    size = 6;
    printf("Testing %s transpose...  ", get_memory_type_name(mem));

    Tensor<float> *x = new Tensor<float>({size, size / 2}, {GLOROT, {0.0f, 1.0f}}, mem);

    op::Operation<float> *x_var = op::var("x_var", x);

    op::Operation<float> *trans_var = op::transpose(x_var);
    Tensor<float> *trans = trans_var->eval();

    sync(trans);

    MAGMADNN_TEST_ASSERT_DEFAULT(trans->get_size() == x->get_size(), "\"trans->get_size() == x->get_size()\" failed");
    MAGMADNN_TEST_ASSERT_DEFAULT(trans->get_shape(0) == x->get_shape(1),
                                 "\"trans->get_shape(0) == x->get_shape(1)\" failed");
    MAGMADNN_TEST_ASSERT_DEFAULT(trans->get_shape(1) == x->get_shape(0),
                                 "\"trans->get_shape(1) == x->get_shape(0)\" failed");

    for (unsigned int i = 0; i < size / 2; i++) {
        for (unsigned int j = 0; j < size; j++) {
            MAGMADNN_TEST_ASSERT_FEQUAL_DEFAULT(trans->get({i, j}), x->get({j, i}));
        }
    }

    show_success();
}

void test_log(memory_t mem_type, unsigned int size) {
    printf("Testing %s log..   ", get_memory_type_name(mem_type));

    Tensor<float> *t = new Tensor<float>({2, 3}, {ZERO, {}}, mem_type);
    t->set({0, 0}, 1.4f);
    t->set({0, 1}, 2.6f);
    t->set({0, 2}, 0.3f);
    t->set({1, 0}, 0.5f);
    t->set({1, 1}, 8.4f);
    t->set({1, 2}, 12.8f);

    op::Operation<float> *x = op::var<float>("x", t);
    op::Operation<float> *out = op::log(x);

    Tensor<float> *output = out->eval();

    sync(output);

    Tensor<float> *grad = new Tensor<float>({2, 3}, {ONE, {}}, mem_type);
    grad->set({0, 0}, 1.0f);
    grad->set({0, 1}, 2.0f);
    grad->set({0, 2}, 3.0f);
    grad->set({1, 0}, 4.0f);
    grad->set({1, 1}, 5.0f);
    grad->set({1, 2}, 6.0f);
    Tensor<float> *d_log_wrt_x = out->grad(NULL, x, grad);
    sync(grad);
    sync(d_log_wrt_x);

    show_success();
}

void test_product(memory_t mem_type, unsigned int size) {
    float val0 = 4;
    float val1 = 6;
    float val2 = 9;
    float total = val0 * val1 * val2;

    printf("testing %s product...  ", get_memory_type_name(mem_type));

    Tensor<float> *t0 = new Tensor<float>({size, size}, {CONSTANT, {val0}}, mem_type);
    Tensor<float> *t1 = new Tensor<float>({size, size}, {CONSTANT, {val1}}, mem_type);
    Tensor<float> *t2 = new Tensor<float>({size, size}, {CONSTANT, {val2}}, mem_type);

    op::Variable<float> *v0 = op::var("t0", t0);
    op::Variable<float> *v1 = op::var("t1", t1);
    op::Variable<float> *v2 = op::var("t2", t2);

    op::Operation<float> *out = op::product(v0, op::product(v1, v2));

    Tensor<float> *output = out->eval();

#if defined(MAGMADNN_HAVE_CUDA)
    if (mem_type == DEVICE || mem_type == CUDA_MANAGED) output->get_memory_manager()->sync();
    if (mem_type == MANAGED) output->get_memory_manager()->sync(true);
#endif

    for (unsigned int i = 0; i < size; i++) {
        for (unsigned int j = 0; j < size; j++) {
            MAGMADNN_TEST_ASSERT_DEFAULT(output->get({i, j}) == total, "\"output->get({i, j}) == total\" failed");
        }
    }

    Tensor<float> *grad = new Tensor<float>({size, size}, {ONE, {}}, mem_type);
    Tensor<float> *d_product_wrt_x = out->grad(NULL, v0, grad);
    sync(grad);
    sync(d_product_wrt_x);

    for (unsigned int i = 0; i < d_product_wrt_x->get_size(); i++) {
        MAGMADNN_TEST_ASSERT_FEQUAL_DEFAULT(d_product_wrt_x->get(i), 54.0f);
    }

    show_success();
}

void test_scalarproduct(memory_t mem_type, unsigned int size) {
    float alpha = 1.5f;
    float val = 50.0f;

    printf("Testing %s scalarproduct...  ", get_memory_type_name(mem_type));

    op::Variable<float> *x = op::var<float>("x", {size, size}, {CONSTANT, {val}}, mem_type);
    op::Operation<float> *prod = op::scalarproduct(alpha, x);

    Tensor<float> *fin = prod->eval();

#if defined(MAGMADNN_HAVE_CUDA)
    if (mem_type == DEVICE || mem_type == CUDA_MANAGED) fin->get_memory_manager()->sync();
    if (mem_type == MANAGED) fin->get_memory_manager()->sync(true);
#endif

    MAGMADNN_TEST_ASSERT_DEFAULT(fin->get_shape(0) == size, "\"fin->get_shape(0) == size\" failed");
    MAGMADNN_TEST_ASSERT_DEFAULT(fin->get_shape(1) == size, "\"fin->get_shape(1) == size\" failed");
    for (unsigned int i = 0; i < size; i++) {
        for (unsigned int j = 0; j < size; j++) {
            // printf("%.4g %.4g\n", fin->get({(int)i,(int)j}), alpha * val);
            MAGMADNN_TEST_ASSERT_FEQUAL_DEFAULT(fin->get({(int) i, (int) j}), alpha * val);
        }
    }

    delete prod;

    show_success();
}

void test_softmax(memory_t mem_type, unsigned int size) {
    printf("Testing %s softmax...  ", get_memory_type_name(mem_type));

    float val1 = 0.09003057f;
    float val2 = 0.24472848f;
    float val3 = 0.66524094f;

    Tensor<float> *t = new Tensor<float>({2, 3}, {ZERO, {}}, mem_type);
    t->set({0, 0}, 1.0f);
    t->set({0, 1}, 2.0f);
    t->set({0, 2}, 3.0f);
    t->set({1, 0}, 3.0f);
    t->set({1, 1}, 2.0f);
    t->set({1, 2}, 1.0f);

    op::Operation<float> *x = op::var<float>("x", t);
    op::Operation<float> *out = op::softmax(x);

    Tensor<float> *output = out->eval();

    sync(output);

    for (unsigned int i = 0; i < 6; i++) {
        if (i == 0 || i == 5)
            MAGMADNN_TEST_ASSERT_DEFAULT(output->get(i) - val1 < 1E-6, "\"output->get(i) - val1 < 1E-6\" failed");
        else if (i == 1 || i == 4)
            MAGMADNN_TEST_ASSERT_DEFAULT(output->get(i) - val2 < 1E-6, "\"output->get(i) - val2 < 1E-6\" failed");
        else
            MAGMADNN_TEST_ASSERT_DEFAULT(output->get(i) - val3 < 1E-6, "\"output->get(i) - val3 < 1E-6\" failed");
    }

    Tensor<float> *grad = new Tensor<float>({2, 3}, {ONE, {}}, mem_type);
    grad->set({0, 0}, 0.9f);
    grad->set({0, 1}, 0.3f);
    grad->set({0, 2}, -0.6f);
    grad->set({1, 0}, -0.9f);
    grad->set({1, 1}, 0.87f);
    grad->set({1, 2}, -0.1415f);
    Tensor<float> *d_softmax_wrt_x = out->grad(NULL, x, grad);
    sync(grad);
    sync(d_softmax_wrt_x);

    show_success();
}

void test_sumreduce(memory_t mem_type, unsigned int size) {
    printf("Testing %s sumreduce...  ", get_memory_type_name(mem_type));

    Tensor<float> *t = new Tensor<float>({2, 3}, {ZERO, {}}, mem_type);
    /* [ [1,2,3], [3,2,1] ] */
    t->set({0, 0}, 1);
    t->set({0, 1}, 2);
    t->set({0, 2}, 3);
    t->set({1, 0}, 3);
    t->set({1, 1}, 2);
    t->set({1, 2}, 1);
    op::Operation<float> *v = op::var<float>("x", t);

    op::Operation<float> *col_sums_o = op::reducesum(v, 0);
    op::Operation<float> *row_sums_o = op::reducesum(v, 1);

    Tensor<float> *col_sums = col_sums_o->eval();
    Tensor<float> *row_sums = row_sums_o->eval();

    sync(col_sums);
    sync(row_sums);

    for (unsigned int i = 0; i < col_sums->get_size(); i++) {
        if (!fequal(col_sums->get(i), 4.0f)) {
            printf("Bad vals : %.3g %.3g\n", col_sums->get(i), 4.0f);
        }
        MAGMADNN_TEST_ASSERT_FEQUAL_DEFAULT(col_sums->get(i), 4.0f);
    }

    for (unsigned int i = 0; i < row_sums->get_size(); i++) {
        if (!fequal(row_sums->get(i), 6.0f)) {
            printf("Bad vals : %.3g %.3g\n", row_sums->get(i), 6.0f);
        }
        MAGMADNN_TEST_ASSERT_FEQUAL_DEFAULT(row_sums->get(i), 6.0f);
    }

    /* test the gradient computation */
    Tensor<float> *grad = new Tensor<float>({1, 3}, {ONE, {}}, mem_type);
    Tensor<float> *d_col_sums_wrt_x = col_sums_o->grad(NULL, v, grad);

    sync(d_col_sums_wrt_x);

    for (unsigned int i = 0; i < d_col_sums_wrt_x->get_shape(0); i++) {
        for (unsigned int j = 0; j < d_col_sums_wrt_x->get_shape(1); j++) {
            MAGMADNN_TEST_ASSERT_FEQUAL_DEFAULT(d_col_sums_wrt_x->get({i, j}), 1.0f);
        }
    }

    show_success();
}

void test_affine(memory_t mem_type, unsigned int size) {
    unsigned int m = size;
    unsigned int n = size;
    unsigned int p = size + 5;
    float val = 5;
    float b = 12.5;

    printf("Testing %s affine...  ", get_memory_type_name(mem_type));

    Tensor<float> *t0 = new Tensor<float>({m, n}, {ZERO, {}}, mem_type);
    Tensor<float> *t1 = new Tensor<float>({n, p}, {CONSTANT, {5}}, mem_type);
    Tensor<float> *t2 = new Tensor<float>({m, p}, {CONSTANT, {b}}, mem_type);

    /* make t0 identity matrix */
    for (int i = 0; i < (int) m; i++)
        for (int j = 0; j < (int) n; j++)
            if (i == j) t0->set({i, j}, 1);

    op::Variable<float> *v0 = op::var("t0", t0);
    op::Variable<float> *v1 = op::var("t1", t1);
    op::Variable<float> *v2 = op::var("t2", t2);

    auto aff = op::add(op::matmul(v0, v1), v2);

    Tensor<float> *fin = aff->eval();

    sync(fin);

    for (int i = 0; i < (int) m; i++) {
        for (int j = 0; j < (int) p; j++) {
            MAGMADNN_TEST_ASSERT_DEFAULT(fin->get({i, j}) == (val + b), "\"fin->get({i, j}) == (val + b)\" failed");
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

    Tensor<float> *t0 = new Tensor<float>({size, size}, {CONSTANT, {-7}}, mem_type);

    auto v0 = op::var("t0", t0);

    auto sig = op::sigmoid(v0, true, true);

    auto fin = sig->eval();

    sync(fin);

    for (unsigned int i = 0; i < fin->get_size(); i++) {
        MAGMADNN_TEST_ASSERT_DEFAULT(fabs(fin->get(i) - (-0.875)) < 1E-8,
                                     "\"fabs(fin->get(i) - (-0.875)) < 1E-8\" failed");
    }

    Tensor<float> *grad = new Tensor<float>({size, size}, {ONE, {}}, mem_type);
    Tensor<float> *d_sigmoid_wrt_x = sig->grad(NULL, v0, grad);
    sync(grad);
    sync(d_sigmoid_wrt_x);

    for (unsigned int i = 0; i < d_sigmoid_wrt_x->get_size(); i++) {
        MAGMADNN_TEST_ASSERT_DEFAULT(fabs(d_sigmoid_wrt_x->get(i) - (-1.640625)) < 1E-8,
                                     "\"fabs(d_sigmoid_wrt_x->get(i) - (-1.640625)) < 1E-8\" failed");
    }

    show_success();
}

void test_tanh(memory_t mem_type, unsigned int size) {
    float val = 5.0;

    printf("Testing %s tanh...  ", get_memory_type_name(mem_type));

    Tensor<float> *t0 = new Tensor<float>({size, size}, {CONSTANT, {val}}, mem_type);

    auto v0 = op::var("t0", t0);

    auto fin_op = op::tanh(v0);

    auto fin = fin_op->eval();

#if defined(MAGMADNN_HAVE_CUDA)
    if (mem_type == DEVICE || mem_type == CUDA_MANAGED)
        fin->get_memory_manager()->sync();
    else if (mem_type == MANAGED)
        fin->get_memory_manager()->sync(true);
#endif

    for (unsigned int i = 0; i < fin->get_size(); i++) {
        MAGMADNN_TEST_ASSERT_DEFAULT(fabs(fin->get(i) - tanh(val)) < 1E-6,
                                     "\"fabs(fin->get(i) - tanh(val)) < 1E-6\" failed");
    }

    delete t0;
    delete fin_op;
    delete fin;

    show_success();
}

void test_conv2d(memory_t mem_type, unsigned int size) {
    printf("Testing %s conv2d...  ", get_memory_type_name(mem_type));

    /* basic convolution2d test */
    unsigned int batch_size = 5;
    unsigned int channels = 3;
    unsigned int h = 5;
    unsigned int w = 5;

    unsigned int in_channels = channels;
    unsigned int out_channels = 2;

    int pad_h = 1;
    int pad_w = 1;
    int vertical_stride = 1;
    int horizontal_stride = 1;
    int dilation_h = 1;
    int dilation_w = 1;
    bool use_cross_correlation = true;

    unsigned int filter_h = 3;
    unsigned int filter_w = 3;

    op::Operation<float> *x = op::var<float>("data", {batch_size, channels, h, w}, {GLOROT, {0.0f, 1.0f}}, mem_type);
    op::Operation<float> *filter =
        op::var<float>("filter", {out_channels, in_channels, filter_h, filter_w}, {GLOROT, {0.0f, 1.0f}}, mem_type);

    op::Operation<float> *conv = op::conv2dforward(x, filter, pad_h, pad_w, vertical_stride, horizontal_stride,
                                                   dilation_h, dilation_w, use_cross_correlation);

    Tensor<float> *out = conv->eval();

    sync(out);

    /* formula from:
     * https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnGetConvolution2dForwardOutputDim
     */
    unsigned int expected_h = 1 + (h + 2 * pad_h - (((filter_h - 1) * dilation_h) + 1)) / vertical_stride;
    unsigned int expected_w = 1 + (w + 2 * pad_w - (((filter_w - 1) * dilation_w) + 1)) / horizontal_stride;

    MAGMADNN_TEST_ASSERT_DEFAULT(out->get_shape().size() == 4, "\"out->get_shape().size() == 4\" failed");
    MAGMADNN_TEST_ASSERT_DEFAULT(out->get_shape(0) == batch_size, "\"out->get_shape(0) == batch_size\" failed");
    MAGMADNN_TEST_ASSERT_DEFAULT(out->get_shape(1) == out_channels, "\"out->get_shape(1) == out_channels\" failed");
    MAGMADNN_TEST_ASSERT_DEFAULT(out->get_shape(2) == expected_h, "\"out->get_shape(2) == expected_h\" failed");
    MAGMADNN_TEST_ASSERT_DEFAULT(out->get_shape(3) == expected_w, "\"out->get_shape(3) == expected_w\" failed");

    delete conv;

    show_success();
}

void test_pooling(memory_t mem_type, unsigned int size) {
    printf("Testing %s pooling...  ", get_memory_type_name(mem_type));

    /* basic pooling test */
    unsigned int batch_size = 5;
    unsigned int channels = 3;
    unsigned int h = size;
    unsigned int w = size;

    int filter_h = 2;
    int filter_w = 2;
    int pad_h = 1;
    int pad_w = 1;
    int vertical_stride = 2;
    int horizontal_stride = 2;

    op::Operation<float> *x = op::var<float>("data", {batch_size, channels, h, w}, {GLOROT, {0.0f, 1.0f}}, mem_type);
    op::Operation<float> *pool =
        op::pooling(x, filter_h, filter_w, pad_h, pad_w, vertical_stride, horizontal_stride, MAX_POOL);

    Tensor<float> *out = pool->eval();

    sync(out);

    /* formula from:
     * https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnGetPoolingNdForwardOutputDim */
    unsigned int expected_h = 1 + (h + 2 * pad_h - filter_h) / vertical_stride;
    unsigned int expected_w = 1 + (w + 2 * pad_w - filter_w) / horizontal_stride;

    MAGMADNN_TEST_ASSERT_DEFAULT(out->get_shape().size() == 4, "\"out->get_shape().size() == 4\" failed");
    MAGMADNN_TEST_ASSERT_DEFAULT(out->get_shape(0) == batch_size, "\"out->get_shape(0) == batch_size\" failed");
    MAGMADNN_TEST_ASSERT_DEFAULT(out->get_shape(1) == channels, "\"out->get_shape(1) == channels\" failed");
    MAGMADNN_TEST_ASSERT_DEFAULT(out->get_shape(2) == expected_h, "\"out->get_shape(2) == expected_h\" failed");
    MAGMADNN_TEST_ASSERT_DEFAULT(out->get_shape(3) == expected_w, "\"out->get_shape(3) == expected_w\" failed");

    /* testing grad */
    Tensor<float> *grad = new Tensor<float>(out->get_shape(), {ONE, {}}, mem_type);
    Tensor<float> *d_flatten_wrt_x = pool->grad(NULL, x, grad);
    sync(grad);
    sync(d_flatten_wrt_x);

    delete pool;

    show_success();
}

void test_batchnorm(memory_t mem_type, unsigned int size) {
    printf("Testing %s batchnorm...  ", get_memory_type_name(mem_type));

    /* basic batch norm test */
    unsigned int batch_size = 5;
    unsigned int channels = 3;
    unsigned int h = size;
    unsigned int w = size;

    op::Operation<float> *x = op::var<float>("data", {batch_size, channels, h, w}, {UNIFORM, {3.0f, 6.0f}}, mem_type);
    op::Operation<float> *batchnorm = op::batchnorm(x);

    Tensor<float> *out = batchnorm->eval();
    sync(out);

    /* testing grad */
    Tensor<float> *grad = new Tensor<float>(out->get_shape(), {ONE, {}}, mem_type);
    Tensor<float> *d_flatten_wrt_x = batchnorm->grad(NULL, x, grad);
    sync(grad);
    sync(d_flatten_wrt_x);

    show_success();
}

void test_crossentropy(memory_t mem_type, unsigned int size) {
    printf("Testing %s crossentropy...  ", get_memory_type_name(mem_type));

    Tensor<double> *actual = new Tensor<double>({3, 3}, {ZERO, {}}, mem_type);
    Tensor<double> *predicted = new Tensor<double>({3, 3}, {ZERO, {}}, mem_type);

    double expected_loss = -(log(0.5 + 1E-8) + log(0.5 + 1E-8) + log(0.1 + 1E-8));

    /* 	0 0 1
            0 0 1
            1 0 0 */
    actual->set({0, 2}, 1.0);
    actual->set({1, 2}, 1.0);
    actual->set({2, 0}, 1.0);

    /* 	0.1 0.4 0.5
            0.2 0.3 0.5
            0.1	0.3	0.6 */
    predicted->set({0, 0}, 0.1);
    predicted->set({0, 1}, 0.4);
    predicted->set({0, 2}, 0.5);

    predicted->set({1, 0}, 0.2);
    predicted->set({1, 1}, 0.3);
    predicted->set({1, 2}, 0.5);

    predicted->set({2, 0}, 0.1);
    predicted->set({2, 1}, 0.3);
    predicted->set({2, 2}, 0.6);

    sync(actual);
    sync(predicted);

    op::Operation<double> *actual_p = op::var("actual", actual);
    op::Operation<double> *predicted_p = op::var("predicted", predicted);

    op::Operation<double> *loss_p = op::crossentropy<double>(actual_p, predicted_p);
    Tensor<double> *out = loss_p->eval();

    sync(out);

    double loss = out->get(0);

    MAGMADNN_TEST_ASSERT_FEQUAL_DEFAULT(loss, expected_loss);

    show_success();
}

void test_meansquarederror(memory_t mem_type, unsigned int size) {
    printf("Testing %s meansquarederror...  ", get_memory_type_name(mem_type));

    op::Operation<float> *truth = op::var<float>("truth", {size}, {GLOROT, {0.0f, 1.0f}}, mem_type);
    op::Operation<float> *predicted = op::var<float>("predicted", {size}, {GLOROT, {0.0f, 1.0f}}, mem_type);

    op::Operation<float> *mse_loss = op::meansquarederror(truth, predicted);
    Tensor<float> *loss_tensor = mse_loss->eval();

    // delete mse_loss;
    // return;

    sync(loss_tensor);

    float expected_loss = 0.0f;
    float diff;
    for (unsigned int i = 0; i < size; i++) {
        diff = truth->get_output_tensor()->get(i) - predicted->get_output_tensor()->get(i);
        expected_loss += diff * diff;
    }
    expected_loss /= (float) size;

    MAGMADNN_TEST_ASSERT_FEQUAL_DEFAULT(loss_tensor->get(0), expected_loss);

    delete mse_loss;

    show_success();
}
