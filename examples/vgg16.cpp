/**
 * @file vgg16.cpp
 * @author Daniel Nichols
 * @version 1.0
 * @date 2019-08-14
 *
 * @copyright Copyright (c) 2019
 */
#include <cstdio>
#include <string>

#include "magmadnn.h"

using namespace magmadnn;

magmadnn_error_t load_cifar_batch(uint32_t batch_idx, const std::string& cifar_root, Tensor<float>** data,
                                  Tensor<float>** labels, uint32_t& n_images, uint32_t& image_width,
                                  uint32_t& image_height, uint32_t& n_channels, uint32_t& n_classes,
                                  bool normalize = true);

magmadnn_error_t read_cifar(const std::string& file_name, Tensor<float>** data, Tensor<float>** labels,
                            uint32_t& n_images, uint32_t& image_width, uint32_t& image_height, uint32_t& n_channels,
                            uint32_t& n_classes, bool normalize = true);

int main(int argc, char** argv) {
    magmadnn_init();

    Tensor<float>*host_data, *host_labels, *data, *labels;
    uint32_t n_images, image_width, image_height, n_channels, n_classes;
    magmadnn_error_t err;
    std::vector<std::string> label_names;
    std::string data_set_root;
    uint32_t cur_batch;

    /* process arguments */
    if (argc == 1) {
        data_set_root = "./cifar-10-batches-bin/";
        cur_batch = 1;
    } else if (argc == 2) {
        data_set_root = std::string(argv[1]);
        cur_batch = 1;
    } else if (argc == 3) {
        data_set_root = std::string(argv[1]);
        cur_batch = std::stoul(argv[2]);
    } else {
        std::fprintf(stderr, "usage: %s <cifar-10-root>\n", argv[0]);
        std::exit(1);
    }

    // read data
    err = load_cifar_batch(cur_batch, data_set_root, &host_data, &host_labels, n_images, image_width, image_height,
                           n_channels, n_classes);

    if (err != 0) {
        std::fprintf(stderr, "Error reading Cifar data.\n");
        std::exit(1);
    } else {
        std::printf("data read successfully!\n");
    }

    memory_t dev;
#if defined(MAGMADNN_HAVE_CUDA)
    dev = DEVICE;
#else
    dev = HOST;
#endif

    data = new Tensor<float>(host_data->get_shape(), {NONE, {}}, dev);
    labels = new Tensor<float>(host_labels->get_shape(), {NONE, {}}, dev);

    data->copy_from(*host_data);
    labels->copy_from(*host_labels);
    delete host_data;
    delete host_labels;

    model::nn_params_t params;
    params.batch_size = 128;
    params.n_epochs = 100;
    params.learning_rate = 0.001;
    params.momentum = 0.9;
    params.decaying_factor = 0.01;
   
    auto x_batch = op::var<float>("x_batch", {params.batch_size, 3, image_height, image_width}, {NONE, {}}, dev);

    auto input_layer = layer::input<float>(x_batch);

    /* CHUNK 1 */
    auto conv1 = layer::conv2d<float>(input_layer->out(), {3, 3}, 64, layer::SAME);
    auto act1 = layer::activation<float>(conv1->out(), layer::RELU);
    /* batchnorm */
    auto drop1 = layer::dropout<float>(act1->out(), 0.3);

    auto conv2 = layer::conv2d<float>(drop1->out(), {3, 3}, 64, layer::SAME);
    auto act2 = layer::activation<float>(conv2->out(), layer::RELU);
    /* batchnorm */

    auto pool1 = layer::pooling<float>(act2->out(), {2, 2}, layer::SAME, {1, 1}, MAX_POOL);

    /* CHUNK 2 */
    auto conv3 = layer::conv2d<float>(pool1->out(), {3, 3}, 128, layer::SAME);
    auto act3 = layer::activation<float>(conv3->out(), layer::RELU);
    /* batchnorm */
    auto drop2 = layer::dropout<float>(act3->out(), 0.4);

    auto conv4 = layer::conv2d<float>(drop2->out(), {3, 3}, 128, layer::SAME);
    auto act4 = layer::activation<float>(conv4->out(), layer::RELU);
    /* batchnorm */

    auto pool2 = layer::pooling<float>(act4->out(), {2, 2}, layer::SAME, {1, 1}, MAX_POOL);

    /* CHUNK 3 */
    auto conv5 = layer::conv2d<float>(pool2->out(), {3, 3}, 256, layer::SAME);
    auto act5 = layer::activation<float>(conv5->out(), layer::RELU);
    /* batchnorm */
    auto drop3 = layer::dropout<float>(act5->out(), 0.4);

    auto conv6 = layer::conv2d<float>(drop3->out(), {3, 3}, 256, layer::SAME);
    auto act6 = layer::activation<float>(conv6->out(), layer::RELU);
    /* batchnorm */
    auto drop4 = layer::dropout<float>(act6->out(), 0.4);

    auto conv7 = layer::conv2d<float>(drop4->out(), {3, 3}, 256, layer::SAME);
    auto act7 = layer::activation<float>(conv7->out(), layer::RELU);
    /* batchnorm */

    auto pool3 = layer::pooling<float>(act7->out(), {2, 2}, layer::SAME, {1, 1}, MAX_POOL);

    /* CHUNK 4 */
    auto conv8 = layer::conv2d<float>(pool3->out(), {3, 3}, 512, layer::SAME);
    auto act8 = layer::activation<float>(conv8->out(), layer::RELU);
    /* batchnorm */
    auto drop5 = layer::dropout<float>(act8->out(), 0.4);

    auto conv9 = layer::conv2d<float>(drop5->out(), {3, 3}, 512, layer::SAME);
    auto act9 = layer::activation<float>(conv9->out(), layer::RELU);
    /* batchnorm */
    auto drop6 = layer::dropout<float>(act9->out(), 0.4);

    auto conv10 = layer::conv2d<float>(drop6->out(), {3, 3}, 512, layer::SAME);
    auto act10 = layer::activation<float>(conv10->out(), layer::RELU);
    /* batchnorm */

    auto pool4 = layer::pooling<float>(act10->out(), {2, 2}, layer::SAME, {1, 1}, MAX_POOL);

    /* CHUNK 5 */
    auto conv11 = layer::conv2d<float>(pool4->out(), {3, 3}, 512, layer::SAME);
    auto act11 = layer::activation<float>(conv11->out(), layer::RELU);
    /* batchnorm */
    auto drop7 = layer::dropout<float>(act11->out(), 0.4);

    auto conv12 = layer::conv2d<float>(drop7->out(), {3, 3}, 512, layer::SAME);
    auto act12 = layer::activation<float>(conv12->out(), layer::RELU);
    /* batchnorm */
    auto drop8 = layer::dropout<float>(act12->out(), 0.4);

    auto conv13 = layer::conv2d<float>(drop8->out(), {3, 3}, 512, layer::SAME);
    auto act13 = layer::activation<float>(conv13->out(), layer::RELU);
    /* batchnorm */

    auto pool5 = layer::pooling<float>(act13->out(), {2, 2}, layer::SAME, {1, 1}, MAX_POOL);
    auto drop9 = layer::dropout<float>(pool5->out(), 0.5);

    auto flat = layer::flatten<float>(drop9->out());
    auto fc1 = layer::fullyconnected<float>(flat->out(), 512, false);
    auto act14 = layer::activation<float>(fc1->out(), layer::RELU);
    /* batchnorm */

    auto drop10 = layer::dropout<float>(act14->out(), 0.5);
    auto fc2 = layer::fullyconnected<float>(drop10->out(), n_classes, false);
    auto act15 = layer::activation<float>(fc2->out(), layer::SIGMOID);
    auto output = layer::output<float>(act15->out());

    std::vector<layer::Layer<float>*> layers = {
        input_layer, conv1,  act1,  drop1, conv2,  act2,  pool1, conv3,  act3,   drop2, conv4,  act4,
        pool2,       conv5,  act5,  drop3, conv6,  act6,  drop4, conv7,  act7,   pool3, conv8,  act8,
        drop5,       conv9,  act9,  drop6, conv10, act10, pool4, conv11, act11,  drop7, conv12, act12,
        drop8,       conv13, act13, pool5, drop9,  flat,  fc1,   act14,  drop10, fc2,   act15,  output};

    // model::NeuralNetwork<float> model(layers, optimizer::CROSS_ENTROPY, optimizer::SGD, params);
    model::NeuralNetwork<float> model(layers, optimizer::CROSS_ENTROPY, optimizer::RMSPROP, params);

    std::printf("model created successfully!\n");
    // std::exit(0);

    model::metric_t output_metrics;
    model.fit(data, labels, output_metrics, true);

    delete data;
    delete labels;

    magmadnn_finalize();
}

#define FREAD_CHECK(res, nmemb, ret)           \
    if ((res) != (nmemb)) {                    \
        std::fprintf(stderr, "fread fail.\n"); \
        return ret;                            \
    }

magmadnn_error_t load_cifar_batch(uint32_t batch_idx, const std::string& cifar_root, Tensor<float>** data,
                                  Tensor<float>** labels, uint32_t& n_images, uint32_t& image_width,
                                  uint32_t& image_height, uint32_t& n_channels, uint32_t& n_classes, bool normalize) {
    return read_cifar(cifar_root + "/data_batch_" + std::to_string(batch_idx) + ".bin", data, labels, n_images,
                      image_width, image_height, n_channels, n_classes, normalize);
}

magmadnn_error_t read_cifar(const std::string& file_name, Tensor<float>** data, Tensor<float>** labels,
                            uint32_t& n_images, uint32_t& image_width, uint32_t& image_height, uint32_t& n_channels,
                            uint32_t& n_classes, bool normalize) {
    n_images = 10000; /* magic numbers from cifar10 file format */
    image_width = 32;
    image_height = 32;
    n_channels = 3;
    n_classes = 10;

    *data = new Tensor<float>({n_images, n_channels, image_height, image_width}, {NONE, {}}, HOST);
    *labels = new Tensor<float>({n_images, n_classes}, {ZERO, {}}, HOST);

    FILE* file;
    uint8_t val;
    uint8_t* img;

    file = std::fopen(file_name.c_str(), "r");

    if (file == NULL) {
        std::fprintf(stderr, "could not open file %s for reading.\n", file_name.c_str());
        return 1;
    }

    img = new uint8_t[n_channels * image_width * image_height];
    for (uint32_t i = 0; i < n_images; i++) {
        FREAD_CHECK(fread(&val, sizeof(uint8_t), 1, file), 1, 1);
        (*labels)->set(i * n_classes + (val), 1.0f);

        FREAD_CHECK(fread(img, sizeof(uint8_t), n_channels * image_width * image_height, file),
                    n_channels * image_width * image_height, 1);

        for (uint32_t j = 0; j < n_channels * image_height * image_width; j++) {
            float normalized_val = (img[j] / 128.0f) - 1.0f;

            (*data)->set(i * n_channels * image_height * image_width + j, normalized_val);
        }
    }

    delete img;

    return (magmadnn_error_t) 0;
}
