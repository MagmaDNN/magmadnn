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
void test_relu(memory_t mem, unsigned int size);
void test_crossentropy(memory_t mem, unsigned int size);
void test_reduce_sum(memory_t mem, unsigned int size);
void test_argmax(memory_t mem, unsigned int size);
void test_bias_add(memory_t mem, unsigned int size);
void test_sum(memory_t mem, unsigned int size);
void test_concat(memory_t mem, unsigned int size);
void test_tile(memory_t mem, unsigned int size);

int main(int argc, char **argv) {
    magmadnn_init();

    test_for_all_mem_types(test_matmul, 50);
    test_for_all_mem_types(test_relu, 50);
    test_for_all_mem_types(test_crossentropy, 10);
    test_for_all_mem_types(test_reduce_sum, 10);
    // test_for_all_mem_types(test_argmax, 10);

    test_argmax(HOST, 10);

    test_for_all_mem_types(test_bias_add, 15);
    test_for_all_mem_types(test_sum, 5);
    test_for_all_mem_types(test_concat, 4);
    test_for_all_mem_types(test_tile, 4);

    magmadnn_finalize();
}

void test_matmul(memory_t mem, unsigned int size) {
    printf("Testing %s matmul...  ", get_memory_type_name(mem));

    Tensor A({size, size / 2}, FLOAT, {CONSTANT, {1.0f}}, mem);
    Tensor B({size, size - 5}, FLOAT, {CONSTANT, {6.0f}}, mem);
    Tensor C({size / 2, size - 5}, FLOAT, {ZERO, {}}, mem);

    math::matmul<float>(1.0f, true, A, false, B, 1.0f, C);

    sync(C);

    for (unsigned int i = 0; i < size / 2; i++) {
        for (unsigned int j = 0; j < size - 5; j++) {
            MAGMADNN_TEST_ASSERT_FEQUAL_DEFAULT(C.get<float>({i, j}), 300.0f);
        }
    }

    show_success();
}

void test_relu(memory_t mem, unsigned int size) {
    printf("Testing %s relu...  ", get_memory_type_name(mem));

    Tensor x({size}, FLOAT, {UNIFORM, {-1.0f, 1.0f}}, mem);
    Tensor relu_out({size}, FLOAT, {NONE, {}}, mem);
    Tensor grad({size}, FLOAT, {UNIFORM, {0.0f, 1.0f}}, mem);
    Tensor relu_grad({size}, FLOAT, {NONE, {}}, mem);

    if (mem == HOST) {
        math::relu<float>(x, relu_out);
    }
#if defined(_HAS_CUDA_)
    else {
        math::relu_cudnn_settings_t settings;
        cudnnErrchk(cudnnCreateActivationDescriptor(&settings.descriptor));
        cudnnErrchk(
            cudnnSetActivationDescriptor(settings.descriptor, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 1.0));
        math::relu_device<float>(x, relu_out, settings);
        cudnnErrchk(cudnnDestroyActivationDescriptor(settings.descriptor));
    }
#endif

    sync(relu_out);

    float x_val;
    for (unsigned int i = 0; i < size; i++) {
        x_val = x.get<float>(i);
        MAGMADNN_TEST_ASSERT_FEQUAL_DEFAULT(relu_out.get<float>(i), (x_val > 0) ? x_val : 0.0f);
    }

    if (mem == HOST) {
        math::relu_grad<float>(x, relu_out, grad, relu_grad);
    }
#if defined(_HAS_CUDA_)
    else {
        math::relu_cudnn_settings_t settings;
        cudnnErrchk(cudnnCreateActivationDescriptor(&settings.descriptor));
        cudnnErrchk(
            cudnnSetActivationDescriptor(settings.descriptor, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 1.0));
        math::relu_grad_device<float>(x, relu_out, grad, relu_grad, settings);
        cudnnErrchk(cudnnDestroyActivationDescriptor(settings.descriptor));
    }
#endif

    sync(relu_grad);

    for (unsigned int i = 0; i < size; i++) {
        x_val = x.get<float>(i);
        MAGMADNN_TEST_ASSERT_FEQUAL_DEFAULT(relu_grad.get<float>(i), (x_val > 0) ? grad.get<float>(i) : 0.0f);
    }

    show_success();
}

void test_crossentropy(memory_t mem, unsigned int size) {
    printf("Testing %s crossentropy...  ", get_memory_type_name(mem));

    unsigned int n_samples = size;
    unsigned int n_classes = size;

    Tensor ground_truth({n_samples, n_classes}, FLOAT, {IDENTITY, {}}, mem);
    Tensor predicted({n_samples, n_classes}, FLOAT, {DIAGONAL, {0.2f}}, mem);
    Tensor out({1}, FLOAT, {NONE, {}}, mem);

    math::crossentropy<float>(predicted, ground_truth, out);

    sync(out);

    show_success();
}

void test_reduce_sum(memory_t mem, unsigned int size) {
    printf("Testing %s reduce_sum...  ", get_memory_type_name(mem));

    Tensor x({size, size}, FLOAT, {IDENTITY, {}}, mem);
    Tensor reduced({size}, FLOAT, {CONSTANT, {5.0f}}, mem);

    if (mem == HOST) {
        Tensor ones({size}, FLOAT, {ONE, {}}, mem);
        math::reduce_sum<float>(x, 1, ones, reduced);
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

        math::reduce_sum_device<float>(x, 1, reduced, settings);
    }
#endif

    sync(reduced);

    for (unsigned int i = 0; i < size; i++) {
        MAGMADNN_TEST_ASSERT_FEQUAL_DEFAULT(reduced.get<float>(i), 1.0f);
    }

    show_success();
}

void test_argmax(memory_t mem, unsigned int size) {
    printf("Testing %s argmax...   ", get_memory_type_name(mem));

    Tensor x({size, size}, FLOAT, {IDENTITY, {}}, mem);
    Tensor out_0({size}, FLOAT, {NONE, {}}, mem);
    Tensor out_1({size}, FLOAT, {NONE, {}}, mem);
    math::argmax<float>(x, 0, out_0);
    math::argmax<float>(x, 1, out_1);

    sync(out_0);
    sync(out_1);

    for (unsigned int i = 0; i < size; i++) {
        MAGMADNN_TEST_ASSERT_FEQUAL_DEFAULT(out_0.get<float>(i), (float) i);
        MAGMADNN_TEST_ASSERT_FEQUAL_DEFAULT(out_1.get<float>(i), (float) i);
    }

    show_success();
}

void test_bias_add(memory_t mem, unsigned int size) {
    printf("Testing %s bias_add...  ", get_memory_type_name(mem));

    Tensor x({size, size / 2}, FLOAT, {UNIFORM, {-1.0f, 1.0f}}, mem);
    Tensor bias({size}, FLOAT, {UNIFORM, {0.0f, 1.0f}}, mem);
    Tensor out({size, size / 2}, FLOAT, {NONE, {}}, mem);

    if (mem == HOST) {
        math::bias_add<CPU>(x, bias, out);
    } else {
        math::bias_add<GPU>(x, bias, out);
    }

    sync(out);

    for (unsigned int i = 0; i < out.get_shape(0); i++) {
        for (unsigned int j = 0; j < out.get_shape(1); j++) {
            MAGMADNN_TEST_ASSERT_FEQUAL_DEFAULT(out.get<float>({i, j}), x.get<float>({i, j}) + bias.get<float>({i}));
            MAGMADNN_TEST_ASSERT_FEQUAL_DEFAULT(out.get<float>({i, j}), x.get<float>({i, j}) + bias.get<float>({i}));
        }
    }

    show_success();
}

void test_sum(memory_t mem, unsigned int size) {
    printf("Testing %s sum...  ", get_memory_type_name(mem));

    Tensor a({size, size}, FLOAT, {UNIFORM, {-1.0f, 1.0f}}, mem);
    Tensor b({size, size}, FLOAT, {UNIFORM, {-1.0f, 1.0f}}, mem);
    Tensor c({size, size}, FLOAT, {UNIFORM, {-1.0f, 1.0f}}, mem);
    Tensor out({size, size}, FLOAT, {ZERO, {}}, mem);

    math::sum<float>({a, b, c}, out);

    std::vector<Tensor> tensors = {a, b, c};
    Tensor actual_out({size, size}, FLOAT, {ZERO, {}}, mem);

    for (unsigned int i = 0; i < size * size; i++) {
        float sum = 0.0f;
        for (const auto &t : tensors) {
            sum += t.get<float>(i);
        }
        actual_out.set<float>(i, sum);
    }

    sync(out);
    sync(actual_out);

    for (unsigned int i = 0; i < size * size; i++) {
        MAGMADNN_TEST_ASSERT_FEQUAL_DEFAULT(out.get<float>(i), actual_out.get<float>(i));
    }

    show_success();
}

void test_concat(memory_t mem, unsigned int size) {
    printf("Testing %s concat...  ", get_memory_type_name(mem));

    Tensor A({size, size / 2, size * 2}, FLOAT, {CONSTANT, {1.0f}}, mem);
    Tensor B({size, size, size * 2}, FLOAT, {CONSTANT, {2.0f}}, mem);
    Tensor C({size, size * 3 / 2, size * 2});

    if (mem == HOST) {
        math::concat<CPU>(A, B, C, 1);
    } else {
        math::concat<GPU>(A, B, C, 1);
    }
    sync(C);

    for (unsigned int i = 0; i < size; i++) {
        for (unsigned int j = 0; j < size * 3 / 2; j++) {
            for (unsigned int k = 0; k < size * 2; k++) {
                if (j < size / 2)
                    MAGMADNN_TEST_ASSERT_FEQUAL_DEFAULT(C.get<float>({i, j, k}), 1.0f);
                else
                    MAGMADNN_TEST_ASSERT_FEQUAL_DEFAULT(C.get<float>({i, j, k}), 2.0f);
            }
        }
    }

    show_success();
}

void test_tile(memory_t mem, unsigned int size) {
    printf("Testing %s tile...  ", get_memory_type_name(mem));

    Tensor D({size, 1, size * 2}, FLOAT, {CONSTANT, {2.0f}}, mem);
    Tensor E({size, size, size * 2}, FLOAT, {NONE, {}}, mem);

    math::tile<float>(D, E, size, 1);
    sync(E);

    for (unsigned int i = 0; i < size; i++) {
        for (unsigned int j = 0; j < size; j++) {
            for (unsigned int k = 0; k < size * 2; k++) {
                MAGMADNN_TEST_ASSERT_DEFAULT(E.get<float>({i, j, k}) == 2.0f,
                                             "\"E.get<float>({i, j, k}) == 2.0f\" failed");
            }
        }
    }

    show_success();
}