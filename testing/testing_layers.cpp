/**
 * @file testing_layers.cpp
 * @author Daniel Nichols
 * @version 0.0.1
 * @date 2019-03-11
 * 
 * @copyright Copyright (c) 2019
 */
#include <stdio.h>
#include "skepsi.h"
#include "utilities.h"

using namespace skepsi;

void test_input(memory_t mem, unsigned int size);
void test_fullyconnected(memory_t mem, unsigned int size);
void test_activation(memory_t mem, unsigned int size);
void test_layers(memory_t mem, unsigned int size);

int main(int argc, char **argv) {
    
    test_input(HOST, 50);
    #if defined(_HAS_CUDA_)
    test_input(DEVICE, 50);
    test_input(MANAGED, 50);
    test_input(CUDA_MANAGED, 50);
    #endif

    test_fullyconnected(HOST, 15);
    #if defined(_HAS_CUDA_)
    test_fullyconnected(DEVICE, 15);
    test_fullyconnected(MANAGED, 15);
    test_fullyconnected(CUDA_MANAGED, 15);
    #endif

    test_activation(HOST, 15);
    #if defined(_HAS_CUDA_)
    test_activation(DEVICE, 15);
    test_activation(MANAGED, 15);
    test_activation(CUDA_MANAGED, 15);
    #endif

    test_layers(HOST, 15);
    #if defined(_HAS_CUDA_)
    test_layers(DEVICE, 15);
    test_layers(MANAGED, 15);
    test_layers(CUDA_MANAGED, 15);
    #endif

}

void test_input(memory_t mem, unsigned int size) {
    printf("testing %s input...  ", get_memory_type_name(mem));

    Tensor<float> *data_tensor = new Tensor<float> ({size, size}, {IDENTITY, {}}, mem);
    op::Variable<float> *data = op::var("data", data_tensor);

    layer::InputLayer<float> *input_layer = new layer::InputLayer<float>(data);

    op::Operation<float> *output = input_layer->out();
    Tensor<float> *output_tensor = output->eval();

    for (unsigned int i = 0; i < size; i++) {
        for (unsigned int j = 0; j < size; j++) {
            if (i == j)
                assert( output_tensor->get({(int)i,(int)j}) == 1.0);
            else
                assert( output_tensor->get({(int)i,(int)j}) == 0.0);
        }
    }

    delete data_tensor;
    delete output;
    show_success();
}

void test_fullyconnected(memory_t mem, unsigned int size) {
    unsigned int hidden_units = 25;

    printf("testing %s fullyconnected...  ", get_memory_type_name(mem));

    Tensor<float> *data_tensor = new Tensor<float> ({size, size}, {IDENTITY, {}}, mem);
    op::Variable<float> *data = op::var("data", data_tensor);

    layer::FullyConnectedLayer<float> *fc = layer::fullyconnected(data, hidden_units, false);

    op::Operation<float> *output = fc->out();
    Tensor<float> *output_tensor = output->eval();

    assert( output_tensor->get_shape().size() == 2 );
    assert( output_tensor->get_shape(0) == size );
    assert( output_tensor->get_shape(1) == hidden_units );

    delete data_tensor;
    show_success();
}

void test_activation(memory_t mem, unsigned int size) {
    float val = 2.3f;

    printf("testing %s activation...  ", get_memory_type_name(mem));

    Tensor<float> *data_tensor = new Tensor<float> ({size, size}, {CONSTANT, {val}}, mem);
    op::Variable<float> *data = op::var("data", data_tensor);

    layer::ActivationLayer<float> *act = layer::activation(data, layer::TANH);

    op::Operation<float> *output = act->out();
    Tensor<float> *output_tensor = output->eval();

    for (unsigned int i = 0; i < output_tensor->get_size(); i++) {
        assert( abs(output_tensor->get(i) - tanh(val)) <= 1E-8 );
    }

    show_success();
}

void test_layers(memory_t mem, unsigned int size) {
    unsigned int hidden_units = 728;
    unsigned int output_classes = 10;

    printf("testing %s layers...  ", get_memory_type_name(mem));

    Tensor<float> *data_tensor = new Tensor<float> ({size, size}, {UNIFORM, {-1.0, 1.0}}, mem);
    op::Variable<float> *data = op::var("data", data_tensor);

    layer::InputLayer<float> *input = layer::input(data);
    layer::FullyConnectedLayer<float> *fc1 = layer::fullyconnected(input->out(), hidden_units);
    layer::FullyConnectedLayer<float> *fc2 = layer::fullyconnected(fc1->out(), output_classes);
    layer::OutputLayer<float> *output = layer::output(fc2->out());

    op::Operation<float> *out = output->out();
    Tensor<float> *out_tensor = out->eval();

    assert( out_tensor->get_shape(0) == size );
    assert( out_tensor->get_shape(1) == output_classes );

    show_success();
}
