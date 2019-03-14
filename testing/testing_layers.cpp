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

using namespace skepsi;

void test_input(memory_t mem, unsigned int size);
void test_fullyconnected(memory_t mem, unsigned int size);
void test_layers(memory_t mem, unsigned int size);
const char* get_memory_type_name(memory_t mem);

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

    test_layers(HOST, 15);
    #if defined(_HAS_CUDA_)
    test_layers(DEVICE, 15);
    test_layers(MANAGED, 15);
    test_layers(CUDA_MANAGED, 15);
    #endif

}

void test_input(memory_t mem, unsigned int size) {
    printf("testing %s input...  ", get_memory_type_name(mem));

    tensor<float> *data_tensor = new tensor<float> ({size, size}, {IDENTITY, {}}, mem);
    op::variable<float> *data = op::var("data", data_tensor);

    layer::input_layer<float> *input_layer = new layer::input_layer<float>(data);

    op::operation<float> *output = input_layer->out();
    tensor<float> *output_tensor = output->eval();

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
    printf("Success!\n");
}

void test_fullyconnected(memory_t mem, unsigned int size) {
    unsigned int hidden_units = 25;

    printf("testing %s fullyconnected...  ", get_memory_type_name(mem));

    tensor<float> *data_tensor = new tensor<float> ({size, size}, {IDENTITY, {}}, mem);
    op::variable<float> *data = op::var("data", data_tensor);

    layer::fullyconnected_layer<float> *fc = layer::fullyconnected(data, hidden_units, false);

    op::operation<float> *output = fc->out();
    tensor<float> *output_tensor = output->eval();

    assert( output_tensor->get_shape().size() == 2 );
    assert( output_tensor->get_shape(0) == size );
    assert( output_tensor->get_shape(1) == hidden_units );

    delete data_tensor;
    printf("Success!\n");
}

void test_layers(memory_t mem, unsigned int size) {
    unsigned int hidden_units = 728;
    unsigned int output_classes = 10;

    printf("testing %s layers...  ", get_memory_type_name(mem));

    tensor<float> *data_tensor = new tensor<float> ({size, size}, {UNIFORM, {-1.0, 1.0}}, mem);
    op::variable<float> *data = op::var("data", data_tensor);

    layer::input_layer<float> *input = layer::input(data);
    layer::fullyconnected_layer<float> *fc1 = layer::fullyconnected(input->out(), hidden_units);
    layer::fullyconnected_layer<float> *fc2 = layer::fullyconnected(fc1->out(), output_classes);
    layer::output_layer<float> *output = layer::output(fc2->out());

    op::operation<float> *out = output->out();
    tensor<float> *out_tensor = out->eval();

    assert( out_tensor->get_shape(0) == size );
    assert( out_tensor->get_shape(1) == output_classes );

    printf("Success!\n");
}

const char* get_memory_type_name(memory_t mem) {
	switch (mem) {
		case HOST: 			return "HOST";
		#if defined(_HAS_CUDA_)
		case DEVICE: 		return "DEVICE";
		case MANAGED: 		return "MANAGED";
		case CUDA_MANAGED: 	return "CUDA_MANAGED";
		#endif
		default: 			return "UNDEFINED_MEMORY_TYPE";
	}
}