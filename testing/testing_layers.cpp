/**
 * @file testing_layers.cpp
 * @author Daniel Nichols
 * @version 1.0
 * @date 2019-03-11
 * 
 * @copyright Copyright (c) 2019
 */
#include <stdio.h>
#include "magmadnn.h"
#include "utilities.h"

using namespace magmadnn;

void test_input(memory_t mem, unsigned int size);
void test_fullyconnected(memory_t mem, unsigned int size);
void test_activation(memory_t mem, unsigned int size);
void test_dropout(memory_t mem, unsigned int size);
void test_flatten(memory_t mem, unsigned int size);
void test_layers(memory_t mem, unsigned int size);

int main(int argc, char **argv) {
    magmadnn_init();
    
    test_for_all_mem_types(test_input, 50);

    test_for_all_mem_types(test_fullyconnected, 15);

    test_for_all_mem_types(test_activation, 15);

    test_for_all_mem_types(test_dropout, 15);

    test_for_all_mem_types(test_flatten, 15);

    test_for_all_mem_types(test_layers, 15);

    magmadnn_finalize();
    return 0;
}

void test_input(memory_t mem, unsigned int size) {
    printf("testing %s input...  ", get_memory_type_name(mem));

    Tensor<float> *data_tensor = new Tensor<float> ({size, size}, {IDENTITY, {}}, mem);
    op::Variable<float> *data = op::var("data", data_tensor);

    layer::InputLayer<float> *input_layer = new layer::InputLayer<float>(data);

    op::Operation<float> *output = input_layer->out();
    Tensor<float> *output_tensor = output->eval();
    sync(output_tensor);

    for (unsigned int i = 0; i < size; i++) {
        for (unsigned int j = 0; j < size; j++) {
            if (i == j)
                assert( fequal(output_tensor->get({i,j}), 1.0f) );
            else
                assert( fequal(output_tensor->get({i,j}), 0.0f) );
        }
    }

    delete data_tensor;
    delete output;
    show_success();
}

void test_fullyconnected(memory_t mem, unsigned int size) {
    unsigned int batch_size = size;
    unsigned int n_features = size + 2;
    unsigned int hidden_units = size*3/4;

    printf("testing %s fullyconnected...  ", get_memory_type_name(mem));

    op::Operation<double> *input = op::var<double>("input", {batch_size, n_features}, {UNIFORM, {-1.0f, 1.0f}}, mem);
    Tensor<double> *input_tensor = input->get_output_tensor();

    layer::FullyConnectedLayer<double> *fc = layer::fullyconnected(input, hidden_units, true);
    op::Operation<double> *out = fc->out();

    Tensor<double> *out_tensor = out->eval();

    Tensor<double> *weight_tensor = fc->get_weight()->get_output_tensor();
    Tensor<double> *bias_tensor = fc->get_bias()->get_output_tensor();
    Tensor<double> predicted_output ({batch_size, hidden_units}, {NONE,{}}, mem);

    assert( out_tensor->get_shape().size() == 2 );
    assert( out_tensor->get_shape(0) == batch_size );
    assert( out_tensor->get_shape(1) == hidden_units );


    /* calculate the predicted output and compare it to the actual */
    math::matmul(1.0, false, input_tensor, false, weight_tensor, 0.0, &predicted_output);
    math::bias_add(&predicted_output, bias_tensor, &predicted_output);

    sync(out_tensor);
    sync(&predicted_output);

    /* assert predicted output and out are the same */
    for (unsigned int i = 0; i < batch_size; i++) {
        for (unsigned int j = 0; j < hidden_units; j++) {
            assert( fequal(predicted_output.get({i,j}), out_tensor->get({i,j})) );
        }
    }

    /* CHECK THE GRADIENT OF THIS LAYER */
    Tensor<double> grad ({batch_size, hidden_units}, {UNIFORM, {-1.0f, 1.0f}}, mem);
    
    Tensor<double> *grad_wrt_weight = out->grad(NULL, fc->get_weight(), &grad);
    Tensor<double> *grad_wrt_bias = out->grad(NULL, fc->get_bias(), &grad);
    Tensor<double> predicted_grad_wrt_weight ({n_features, hidden_units}, {NONE,{}}, mem);

    math::matmul(1.0, true, input_tensor, false, &grad, 0.0, &predicted_grad_wrt_weight);

    sync(grad_wrt_bias);
    sync(grad_wrt_weight);
    sync(&predicted_grad_wrt_weight);

    /* grad_wrt_weight should be X^T . G */
    for (unsigned int i = 0; i < n_features; i++) {
        for (unsigned int j = 0; j < hidden_units; j++) {
            assert( fequal(grad_wrt_weight->get({i,j}), predicted_grad_wrt_weight.get({i,j})) );
        }
    }
    /* grad_wrt_bias should be row sums of grad */
    for (unsigned int i = 0; i < batch_size; i++) {
        double row_sum = 0.0;
        for (unsigned int j = 0; j < hidden_units; j++) {
            row_sum += grad.get({i, j});
        }
        assert( fabs(row_sum - grad_wrt_bias->get(i)) <= 1E-8 );
    }

    delete fc;

    show_success();
}

void test_activation(memory_t mem, unsigned int size) {
    float val = 2.3f;

    printf("testing %s activation...  ", get_memory_type_name(mem));

    Tensor<float> *data_tensor = new Tensor<float> ({size, size}, {CONSTANT, {val}}, mem);
    op::Variable<float> *data = op::var("data", data_tensor);

    /* create the layer */
    layer::ActivationLayer<float> *act = layer::activation(data, layer::TANH);

    /* the output of the layer */
    op::Operation<float> *output = act->out();
    Tensor<float> *output_tensor = output->eval();

    /* synchronize the memory if managed was being used */
    sync(output_tensor);

    for (unsigned int i = 0; i < output_tensor->get_size(); i++) {
        assert( fabs(output_tensor->get(i) - tanh(val)) <= 1E-8 );
    }

    show_success();
}

void test_dropout(memory_t mem, unsigned int size) {
    float val = 2.4;
    float dropout_rate = 0.2;

    printf("testing %s dropout...  ", get_memory_type_name(mem));

    Tensor<float> *data_tensor = new Tensor<float> ({size, size}, {CONSTANT, {val}}, mem);
    op::Variable<float> *data = op::var("data", data_tensor);

    /* create the layer */
    layer::DropoutLayer<float> *dropout = layer::dropout(data, dropout_rate);

    /* the output of the layer */
    op::Operation<float> *output = dropout->out();
    Tensor<float> *output_tensor = output->eval();

    /* synchronize the memory if managed was being used */
    sync(output_tensor);

    bool exists_zero = false;
    bool exists_nonzero = false;
    for (unsigned int i = 0; i < output_tensor->get_shape(0); i ++) {
        for (unsigned int j = 0; j < output_tensor->get_shape(1); j ++) {
            assert(output_tensor->get({i,j}) == 0.0f || output_tensor->get({i,j}) == val / (1 - dropout_rate));
            if (output_tensor->get({i,j}) == 0.0f) {
                exists_zero = true;
            } if (output_tensor->get({i,j}) == val / (1 - dropout_rate)) {
                exists_nonzero = true;
            }
        } 
    }

    if (!exists_zero || !exists_nonzero) assert(false);

    show_success();
}

void test_flatten(memory_t mem, unsigned int size) {
    printf("testing %s flatten...  ", get_memory_type_name(mem));

    Tensor<float> *data_tensor = new Tensor<float> ({size*5, size, size*3, size*2}, {CONSTANT, {3}}, mem);
    op::Variable<float> *data = op::var("data", data_tensor);

    /* create the layer */
    layer::FlattenLayer<float> *flatten = layer::flatten(data);

    /* the output of the layer */
    op::Operation<float> *output = flatten->out();
    Tensor<float> *output_tensor = output->eval();

    /* synchronize the memory if managed was being used */
    sync(output_tensor);

    assert( output_tensor->get_shape().size() == 2 );
    assert( output_tensor->get_shape(0) == size*5 );
    assert( output_tensor->get_shape(1) == size*(size*3)*(size*2) );

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
    layer::DropoutLayer<float> *dropout = layer::dropout(fc1->out(), 0.1);
    layer::FullyConnectedLayer<float> *fc2 = layer::fullyconnected(dropout->out(), output_classes);
    layer::OutputLayer<float> *output = layer::output(fc2->out());

    op::Operation<float> *out = output->out();
    Tensor<float> *out_tensor = out->eval();

    sync(out_tensor);

    assert( out_tensor->get_shape(0) == size );
    assert( out_tensor->get_shape(1) == output_classes );

    show_success();
}
