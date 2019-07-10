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

using namespace magmadnn;

void test_matmul(memory_t mem, unsigned int size);
void test_pow(memory_t mem, unsigned int size);
void test_relu(memory_t mem, unsigned int size);
void test_crossentropy(memory_t mem, unsigned int size);
void test_reduce_sum(memory_t mem, unsigned int size);
void test_argmax(memory_t mem, unsigned int size);
void test_bias_add(memory_t mem, unsigned int size);
void test_concat(memory_t mem, unsigned int size);
void test_tile(memory_t mem, unsigned int size);

int main(int argc, char **argv) {
    magmadnn_init();

    test_for_all_mem_types(test_matmul, 50);
    test_for_all_mem_types(test_pow, 15);
    test_for_all_mem_types(test_relu, 50);
    test_for_all_mem_types(test_crossentropy, 10);
    test_for_all_mem_types(test_reduce_sum, 10);
    // test_for_all_mem_types(test_argmax, 10);

    test_argmax(HOST, 10);

    test_for_all_mem_types(test_bias_add, 15);
    test_for_all_mem_types(test_concat, 4);
    test_for_all_mem_types(test_tile, 4);

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
            assert(fequal(C->get({i, j}), 300.0f));
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
        assert(fequal(out->get(i), 3.0f * 3.0f * 3.0f));
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
#if defined(_HAS_CUDA_)
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
        assert(fequal(relu_out->get(i), (x_val > 0) ? x_val : 0.0f));
    }

    if (mem == HOST) {
        math::relu_grad(x, relu_out, grad, relu_grad);
    }
#if defined(_HAS_CUDA_)
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
        assert(fequal(relu_grad->get(i), (x_val > 0) ? grad->get(i) : 0.0f));
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
#if defined(_HAS_CUDA_)
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
        assert(fequal(reduced.get(i), 1.0f));
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
        assert(fequal(out_0.get(i), (float) i));
        assert(fequal(out_1.get(i), (float) i));
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
            assert(fequal(out.get({i, j}), x.get({i, j}) + bias.get({i})));
        }
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
                    assert(fequal(C->get({i, j, k}), 1.0f));
                else
                    assert(fequal(C->get({i, j, k}), 2.0f));
            }
        }
    }

    show_success();
}

void test_tile(memory_t mem, unsigned int size) {
    printf("Testing %s tile...  ", get_memory_type_name(mem));

    Tensor<float> *D = new Tensor<float>({size, 1, size * 2}, {CONSTANT, {2.0f}}, mem);
    Tensor<float> *E = new Tensor<float>({size, size, size * 2});

    math::tile(D, E, size, 1);
    sync(E);

    for (unsigned int i = 0; i < size; i++) {
        for (unsigned int j = 0; j < size; j++) {
            for (unsigned int k = 0; k < size * 2; k++) {
                assert(E->get({i, j, k}) == 2.0f);
            }
        }
    }

    show_success();
}