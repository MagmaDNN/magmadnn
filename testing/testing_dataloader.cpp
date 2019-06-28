/**
 * @file testing_dataloader.cpp
 * @author Sedrick Keh
 * @version 0.1
 * @date 2019-06-26
 * 
 * @copyright Copyright (c) 2019
 */
#include "magmadnn.h"
#include "utilities.h"
#include <iostream>
using namespace magmadnn;

void test_linear(memory_t mem_type, unsigned int size);

int main(int argc, char **argv) {
	magmadnn_init();

	test_for_all_mem_types(test_linear, 50);
    
	magmadnn_finalize();
    return 0;
}

void test_linear(memory_t mem_type, unsigned int size) {
    printf("Testing %s linear loader...  ", get_memory_type_name(mem_type));
    
    Tensor<float> *x = new Tensor<float> ({size, size}, {UNIFORM, {-1.0f, 1.0f}}, mem_type);
    Tensor<float> *y = new Tensor<float> ({size}, {UNIFORM, {-1.0f, 1.0f}}, mem_type);
    unsigned int batch_size = size/4;
    unsigned int num_samples = x->get_shape(0);
    dataloader::LinearLoader<float> *data = new dataloader::LinearLoader<float>(x, y, batch_size);

    Tensor<float> *x_batch = new Tensor<float> ({num_samples / data->get_num_batches() , size}, mem_type);
    Tensor<float> *y_batch = new Tensor<float> ({num_samples / data->get_num_batches(), 1}, mem_type);

    for (unsigned int i = 0; i < data->get_num_batches(); i ++) {
        data->next(x_batch, y_batch);
        for (unsigned int j = 0; j < batch_size; j ++) {
            for (unsigned int k = 0; k < size; k ++) {
                assert(x_batch->get({j,k}) == x->get({i*batch_size+j, k}));
            }
            assert(y_batch->get(j) == y->get(i*batch_size+j));
        }
    }

    show_success();
}