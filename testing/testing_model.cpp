/**
 * @file testing_model.cpp
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-05-30
 * 
 * @copyright Copyright (c) 2019
 */

#include <cstdio>
#include <vector>
#include "magmadnn.h"
#include "utilities.h"


using namespace magmadnn;


void test_model_MLP(memory_t mem, unsigned int size);


int main(int argc, char **argv) {
    magmadnn_init();

    test_for_all_mem_types(test_model_MLP, 50);

    magmadnn_finalize();
    return 0;
}

void test_model_MLP(memory_t mem, unsigned int size) {
    unsigned int n_features = 6;
    unsigned int n_classes = 5;
    unsigned int n_samples = 10;
    model::metric_t metrics;

    printf("testing %s MLP...  ", get_memory_type_name(mem));

    Tensor<float> x ({n_samples, n_features}, {CONSTANT, {1.0f}}, mem);
    auto var = op::var<float>("x", &x);

    Tensor<float> y ({n_samples, n_classes}, {CONSTANT, {1.0f}}, mem);

    auto input = layer::input<float>(var);
    auto fc1 = layer::fullyconnected<float>(input->out(), n_classes, true);
    auto act1 = layer::activation<float>(fc1->out(), layer::SIGMOID);
    auto output = layer::output<float>(act1->out());

    std::vector<layer::Layer<float> *> layers = {input, fc1, act1, output};

    model::nn_params_t p;
    p.n_epochs = 5;
    p.batch_size = 10;
    model::NeuralNetwork<float> model (layers, optimizer::CROSS_ENTROPY, optimizer::SGD, p);

    /* training routing */
    model.fit(&x, &y, metrics);

    printf("loss: %.5g\n", metrics.loss);

    show_success();
}