/**
 * @file testing_math.cpp
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-06-10
 *
 * @copyright Copyright (c) 2019
 *
 */

#include "magmadnn.h"
#include "utilities.h"
// #include "mat"

using namespace magmadnn;

void test_matmul(memory_t mem, unsigned int size);
void test_pow(memory_t mem, unsigned int size);
void test_relu(memory_t mem, unsigned int size);
void test_crossentropy(memory_t mem, unsigned int size);
void test_reduce_sum(memory_t mem, unsigned int size);
void test_argmax(memory_t mem, unsigned int size);
void test_bias_add(memory_t mem, unsigned int size);
void test_sum(memory_t mem, unsigned int size);
void test_concat(memory_t mem, unsigned int size);
void test_tile(memory_t mem, unsigned int size);
void test_conv2d(memory_t mem, unsigned int size);
void test_conv2d_grad(memory_t mem, unsigned int size);

template <typename T>
void compare_tensor(Tensor<T> *a, Tensor<T> *b, bool print);

int main(int argc, char **argv) {
    magmadnn_init();

    test_for_all_mem_types(test_matmul, 50);
    test_for_all_mem_types(test_pow, 15);
    printf("warning: skipping math tests\n");
    // test_for_all_mem_types(test_relu, 50);
    test_for_all_mem_types(test_crossentropy, 10);
    // test_for_all_mem_types(test_reduce_sum, 10);
    // test_for_all_mem_types(test_argmax, 10);

    // test_argmax(HOST, 10);
    test_conv2d(HOST, 30);
    // test_conv2d_grad(HOST, 30);

    test_for_all_mem_types(test_bias_add, 15);
    test_for_all_mem_types(test_sum, 5);
    test_for_all_mem_types(test_concat, 4);
    // test_for_all_mem_types(test_tile, 4);

    magmadnn_finalize();
}

void test_matmul(memory_t mem, unsigned int size) {
    printf("Testing %s matmul...  ", get_memory_type_name(mem));

    Tensor<float> *A = new Tensor<float>({size, size / 2}, {CONSTANT, {1.0f}}, mem);
    Tensor<float> *B = new Tensor<float>({size, size - 5}, {CONSTANT, {6.0f}}, mem);
    Tensor<float> *C = new Tensor<float>({size / 2, size - 5}, {ZERO, {}}, mem);

    math::matmul(1.0f, true, A, false, B, 1.0f, C);

    sync(C);

    for (unsigned int i = 0; i < size / 2; i++) {
        for (unsigned int j = 0; j < size - 5; j++) {
            MAGMADNN_TEST_ASSERT_FEQUAL_DEFAULT(C->get({i, j}), 300.0f);
        }
    }

    delete A;
    delete B;
    delete C;
    show_success();
}

void test_pow(memory_t mem, unsigned int size) {
    printf("Testing %s pow...  ", get_memory_type_name(mem));

    float val = 3.0f;

    Tensor<float> *x = new Tensor<float>({size, size}, {CONSTANT, {val}}, mem);
    Tensor<float> *out = new Tensor<float>({size, size}, {NONE, {}}, mem);

    math::pow(x, 3, out);

    sync(out);

    for (unsigned int i = 0; i < size * size; i++) {
        MAGMADNN_TEST_ASSERT_FEQUAL_DEFAULT(out->get(i), 3.0f * 3.0f * 3.0f);
    }

    show_success();
}

void test_relu(memory_t mem, unsigned int size) {
    printf("Testing %s relu...  ", get_memory_type_name(mem));

    Tensor<float> *x = new Tensor<float>({size}, {UNIFORM, {-1.0f, 1.0f}}, mem);
    Tensor<float> *relu_out = new Tensor<float>({size}, {NONE, {}}, mem);
    Tensor<float> *grad = new Tensor<float>({size}, {UNIFORM, {0.0f, 1.0f}}, mem);
    Tensor<float> *relu_grad = new Tensor<float>({size}, {NONE, {}}, mem);

    if (mem == HOST) {
        math::relu(x, relu_out);
    }
#if defined(MAGMADNN_HAVE_CUDA)
    else {
        math::relu_cudnn_settings_t settings;
        cudnnErrchk(cudnnCreateActivationDescriptor(&settings.descriptor));
        cudnnErrchk(
            cudnnSetActivationDescriptor(settings.descriptor, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 1.0));
        math::relu_device(x, relu_out, settings);
        cudnnErrchk(cudnnDestroyActivationDescriptor(settings.descriptor));
    }
#endif

    sync(relu_out);

    float x_val;
    for (unsigned int i = 0; i < size; i++) {
        x_val = x->get(i);
        MAGMADNN_TEST_ASSERT_FEQUAL_DEFAULT(relu_out->get(i), (x_val > 0) ? x_val : 0.0f);
    }

    if (mem == HOST) {
        math::relu_grad(x, relu_out, grad, relu_grad);
    }
#if defined(MAGMADNN_HAVE_CUDA)
    else {
        math::relu_cudnn_settings_t settings;
        cudnnErrchk(cudnnCreateActivationDescriptor(&settings.descriptor));
        cudnnErrchk(
            cudnnSetActivationDescriptor(settings.descriptor, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 1.0));
        math::relu_grad_device(x, relu_out, grad, relu_grad, settings);
        cudnnErrchk(cudnnDestroyActivationDescriptor(settings.descriptor));
    }
#endif

    sync(relu_grad);

    for (unsigned int i = 0; i < size; i++) {
        x_val = x->get(i);
        MAGMADNN_TEST_ASSERT_FEQUAL_DEFAULT(relu_grad->get(i), (x_val > 0) ? grad->get(i) : 0.0f);
    }

    delete x;
    delete relu_out;
    delete grad;
    delete relu_grad;

    show_success();
}

void test_crossentropy(memory_t mem, unsigned int size) {
    printf("Testing %s crossentropy...  ", get_memory_type_name(mem));

    unsigned int n_samples = size;
    unsigned int n_classes = size;

    Tensor<float> *ground_truth = new Tensor<float>({n_samples, n_classes}, {IDENTITY, {}}, mem);
    Tensor<float> *predicted = new Tensor<float>({n_samples, n_classes}, {DIAGONAL, {0.2f}}, mem);
    Tensor<float> *out = new Tensor<float>({1}, {NONE, {}}, mem);

    math::crossentropy(predicted, ground_truth, out);

    sync(out);

    show_success();
}

void test_reduce_sum(memory_t mem, unsigned int size) {
    printf("Testing %s reduce_sum...  ", get_memory_type_name(mem));

    Tensor<float> x({size, size}, {IDENTITY, {}}, mem);
    Tensor<float> reduced({size}, {CONSTANT, {5.0f}}, mem);

    if (mem == HOST) {
        Tensor<float> ones({size}, {ONE, {}}, mem);
        math::reduce_sum(&x, 1, &ones, &reduced);
    }
#if defined(MAGMADNN_HAVE_CUDA)
    else {
        math::reduce_sum_cudnn_settings_t settings;

        cudnnCreateReduceTensorDescriptor(&settings.descriptor);
        cudnnSetReduceTensorDescriptor(settings.descriptor, CUDNN_REDUCE_TENSOR_ADD, CUDNN_DATA_FLOAT,
                                       CUDNN_NOT_PROPAGATE_NAN, CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_32BIT_INDICES);
        cudnnGetReductionWorkspaceSize(internal::MAGMADNN_SETTINGS->cudnn_handle, settings.descriptor,
                                       x.get_cudnn_tensor_descriptor(), reduced.get_cudnn_tensor_descriptor(),
                                       &settings.workspace_size);
        cudaErrchk(cudaMalloc((void **) &settings.workspace, settings.workspace_size * sizeof(float)));

        math::reduce_sum_device(&x, 1, &reduced, settings);
    }
#endif

    sync(&reduced);

    for (unsigned int i = 0; i < size; i++) {
        MAGMADNN_TEST_ASSERT_FEQUAL_DEFAULT(reduced.get(i), 1.0f);
    }

    show_success();
}

void test_argmax(memory_t mem, unsigned int size) {
    printf("Testing %s argmax...   ", get_memory_type_name(mem));

    Tensor<float> x({size, size}, {IDENTITY, {}}, mem);
    Tensor<float> out_0({size}, mem);
    Tensor<float> out_1({size}, mem);
    math::argmax(&x, 0, &out_0);
    math::argmax(&x, 1, &out_1);

    sync(&out_0);
    sync(&out_1);

    for (unsigned int i = 0; i < size; i++) {
        MAGMADNN_TEST_ASSERT_FEQUAL_DEFAULT(out_0.get(i), (float) i);
        MAGMADNN_TEST_ASSERT_FEQUAL_DEFAULT(out_1.get(i), (float) i);
    }

    show_success();
}

void test_bias_add(memory_t mem, unsigned int size) {
    printf("Testing %s bias_add...  ", get_memory_type_name(mem));

    Tensor<float> x({size, size / 2}, {UNIFORM, {-1.0f, 1.0f}}, mem);
    Tensor<float> bias({size}, {UNIFORM, {0.0f, 1.0f}}, mem);
    Tensor<float> out({size, size / 2}, {NONE, {}}, mem);

    math::bias_add(&x, &bias, &out);

    sync(&out);

    for (unsigned int i = 0; i < out.get_shape(0); i++) {
        for (unsigned int j = 0; j < out.get_shape(1); j++) {
            MAGMADNN_TEST_ASSERT_FEQUAL_DEFAULT(out.get({i, j}), x.get({i, j}) + bias.get({i}));
            MAGMADNN_TEST_ASSERT_FEQUAL_DEFAULT(out.get({i, j}), x.get({i, j}) + bias.get({i}));
        }
    }

    show_success();
}

void test_sum(memory_t mem, unsigned int size) {
    printf("Testing %s sum...  ", get_memory_type_name(mem));

    Tensor<float> a({size, size}, {UNIFORM, {-1.0f, 1.0f}}, mem);
    Tensor<float> b({size, size}, {UNIFORM, {-1.0f, 1.0f}}, mem);
    Tensor<float> c({size, size}, {UNIFORM, {-1.0f, 1.0f}}, mem);
    Tensor<float> out({size, size}, {ZERO, {}}, mem);

    math::sum({&a, &b, &c}, &out);

    std::vector<Tensor<float> *> tensors = {&a, &b, &c};
    Tensor<float> actual_out({size, size}, {ZERO, {}}, mem);

    for (unsigned int i = 0; i < size * size; i++) {
        float sum = 0.0f;
        for (const auto &t : tensors) {
            sum += t->get(i);
        }
        actual_out.set(i, sum);
    }

    sync(&out);
    sync(&actual_out);

    for (unsigned int i = 0; i < size * size; i++) {
        MAGMADNN_TEST_ASSERT_FEQUAL_DEFAULT(out.get(i), actual_out.get(i));
    }

    show_success();
}

void test_concat(memory_t mem, unsigned int size) {
    printf("Testing %s concat...  ", get_memory_type_name(mem));

    Tensor<float> *A = new Tensor<float>({size, size / 2, size * 2}, {CONSTANT, {1.0f}}, mem);
    Tensor<float> *B = new Tensor<float>({size, size, size * 2}, {CONSTANT, {2.0f}}, mem);
    Tensor<float> *C = new Tensor<float>({size, size * 3 / 2, size * 2});

    math::concat(A, B, C, 1);
    sync(C);

    for (unsigned int i = 0; i < size; i++) {
        for (unsigned int j = 0; j < size * 3 / 2; j++) {
            for (unsigned int k = 0; k < size * 2; k++) {
                if (j < size / 2)
                    MAGMADNN_TEST_ASSERT_FEQUAL_DEFAULT(C->get({i, j, k}), 1.0f);
                else
                    MAGMADNN_TEST_ASSERT_FEQUAL_DEFAULT(C->get({i, j, k}), 2.0f);
            }
        }
    }
    printf("Testing %s tile...  ", get_memory_type_name(mem));

    Tensor<float> *D = new Tensor<float>({size, 1, size * 2}, {CONSTANT, {2.0f}}, mem);
    Tensor<float> *E = new Tensor<float>({size, size, size * 2});

    math::tile(D, E, size, 1);
    sync(E);

    for (unsigned int i = 0; i < size; i++) {
        for (unsigned int j = 0; j < size; j++) {
            for (unsigned int k = 0; k < size * 2; k++) {
                MAGMADNN_TEST_ASSERT_DEFAULT(E->get({i, j, k}) == 2.0f, "\"E->get({i, j, k}) == 2.0f\" failed");
            }
        }
    }

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

#ifdef MAGMADNN_HAVE_CUDA
    if (mem_type == HOST) {
        op::Operation<float> *x_dev =
            op::var<float>("data", {batch_size, channels, h, w}, {GLOROT, {0.0f, 1.0f}}, DEVICE);
        op::Operation<float> *filter_dev =
            op::var<float>("filter", {out_channels, in_channels, filter_h, filter_w}, {GLOROT, {0.0f, 1.0f}}, DEVICE);

        op::Operation<float> *conv_dev =
            op::conv2dforward(x_dev, filter_dev, pad_h, pad_w, vertical_stride, horizontal_stride, dilation_h,
                              dilation_w, use_cross_correlation);

        Tensor<float> *out_dev = conv_dev->eval();

        compare_tensor(out_dev, out, false);
    }
#endif

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

template <typename T>
void test_conv2d_grad(memory_t mem_type, unsigned int size) {
    printf("Testing %s conv2d_grad...  ", get_memory_type_name(mem_type));

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

    unsigned int grad_h = 1 + (h + 2 * pad_h - (((filter_h - 1) * dilation_h) + 1)) / vertical_stride;
    unsigned int grad_w = 1 + (w + 2 * pad_w - (((filter_w - 1) * dilation_w) + 1)) / horizontal_stride;

    Tensor<T> *out = new Tensor<T>({out_channels, batch_size, filter_h, filter_w}, {NONE, {}}, HOST);
    Tensor<T> *input = new Tensor<T>({batch_size, channels, h, w}, {GLOROT, {}}, HOST);
    Tensor<T> *grad = new Tensor<T>({batch_size, out_channels, grad_h, grad_w}, {GLOROT, {}}, HOST);

    ::magmadnn::math::conv2d_grad_filter(input, grad, out, pad_h, pad_w, vertical_stride, horizontal_stride, dilation_h,
                                         dilation_w);
#if defined(MAGMADNN_HAVE_CUDA)
    //                 print("Testing CPU conv requires GPU.\n");
    //                 return;
    Tensor<T> *gpu_test = new Tensor<T>(out->get_shape(), {NONE, {}}, DEVICE);
    Tensor<T> *gpu_input = new Tensor<T>(input->get_shape(), {NONE, {}}, DEVICE);
    Tensor<T> *gpu_grad = new Tensor<T>(grad->get_shape(), {NONE, {}}, DEVICE);
    // *input = *(this->input_tensor);
    gpu_input->copy_from(*(input));
    gpu_grad->copy_from(*(grad));
    // this->cudnn_settings.handle = this->get_cudnn_handle();
    // ::magmadnn::math::conv2d_grad_filter_device(input, gpu_grad, gpu_test, this->cudnn_settings);
    // if (!this->get_async()) cudaStreamSynchronize(this->get_custream());
    printf("conv backward grad: ");
    compare_tensor(gpu_test, out, true);
    // compare_tensor(input, this->input_tensor);

#endif

    sync(out);

    // delete conv;

    show_success();
}

template <typename T>
void compare_tensor(Tensor<T> *a, Tensor<T> *b, bool print) {
    std::vector<unsigned int> shape = a->get_shape();
    if (a->get_shape().size() != b->get_shape().size()) {
        printf("shapes don't match\n");
        return;
    }

    double norm_difference, l2norm_a = 0, l2norm_b = 0;

    for (int i = 0; i < shape.size(); i++) {
        if (a->get_shape()[i] != b->get_shape()[i]) {
            printf("shape sizes don't match\n");
            return;
        }
    }

    float max = 0;
    if (a->get_shape().size() == 4) {
        if (print) printf("\n[");
        for (int n = 0; n < shape[0]; n++) {
            if (print) printf("[");
            for (int c = 0; c < shape[1]; c++) {
                if (print) printf("[");
                for (int h = 0; h < shape[2]; h++) {
                    if (print) printf("[");
                    for (int w = 0; w < shape[3]; w++) {
                        float diff = a->get({n, c, h, w}) - b->get({n, c, h, w});
                        l2norm_a += a->get({n, c, h, w}) * a->get({n, c, h, w});
                        l2norm_b += b->get({n, c, h, w}) * b->get({n, c, h, w});

                        if (diff < 0) diff = -diff;

                        if (diff > max) max = diff;
                        if (print) {
                            int color = 37;
                            if (diff > 0) color = 32;
                            if (diff > 0.00001) color = 33;
                            if (diff > 0.00005) color = 31;
                            printf("\033[1;%im%f\033[0m ", color, diff);
                        }
                    }
                    if (print) {
                        if (h == shape[3] - 1)
                            printf("]");
                        else
                            printf("]\n   ");
                    }
                }
                if (print) {
                    if (c == shape[1] - 1)
                        printf("]");
                    else
                        printf("]\n  ");
                }
            }
            if (print) {
                if (n == shape[0] - 1)
                    printf("]");
                else
                    printf("]\n\n ");
            }
        }
        if (print) printf("]\n");
    }

    l2norm_a = sqrt(l2norm_a);
    l2norm_b = sqrt(l2norm_b);

    norm_difference = l2norm_a - l2norm_b;
    if (norm_difference < 0) norm_difference = -norm_difference;

    printf("max error: %lf ", max);
    printf("l2norm: %lf ", norm_difference);
}
