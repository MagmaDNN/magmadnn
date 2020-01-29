/**
 * @file cifar10_interactive.cpp
 * @author Daniel Nichols
 * @version 1.0
 * @date 2019-07-16
 *
 * @copyright Copyright (c) 2019
 */
#include <cstdint>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include "magmadnn.h"

using namespace magmadnn;

magmadnn_error_t load_cifar_batch(uint32_t batch_idx, const std::string& cifar_root, Tensor<float>** data,
                                  Tensor<float>** labels, uint32_t& n_images, uint32_t& image_width,
                                  uint32_t& image_height, uint32_t& n_channels, uint32_t& n_classes,
                                  bool normalize = true);

magmadnn_error_t read_cifar(const std::string& file_name, Tensor<float>** data, Tensor<float>** labels,
                            uint32_t& n_images, uint32_t& image_width, uint32_t& image_height, uint32_t& n_channels,
                            uint32_t& n_classes, bool normalize = true);

std::vector<std::string> get_class_names(const std::string& file_name);
void print_image(uint32_t idx, const Tensor<float>* data, const Tensor<float>* labels,
                 const std::vector<std::string>& label_names);

void show_image(const Tensor<float>* data, const Tensor<float>* labels, const std::vector<std::string>& label_names);

void train(model::NeuralNetwork<float>& model, Tensor<float>* data, Tensor<float>* labels);

void predict(model::NeuralNetwork<float>& model, Tensor<float>* images, Tensor<float>* labels,
             const std::vector<std::string>& label_names);

void switch_batch(Tensor<float>** data, Tensor<float>** labels, uint32_t& cur_batch, const std::string& cifar_root,
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
    } else if (argc == 2) {
        data_set_root = std::string(argv[1]);
    } else {
        std::cerr << "usage: " << argv[0] << "<?cifar data root>\n";
        std::exit(1);
    }

    cur_batch = 1;

    // read data
    err = load_cifar_batch(cur_batch, data_set_root, &host_data, &host_labels, n_images, image_width, image_height,
                           n_channels, n_classes);

    if (err != 0) {
        std::fprintf(stderr, "Error reading Cifar data.\n");
        std::exit(1);
    }

    memory_t training_memory_type;

#if defined(MAGMADNN_HAVE_CUDA)
    training_memory_type = DEVICE;
    std::cout << "Training on GPUs" << std::endl;
#else
    training_memory_type = HOST;
    std::cout << "Training on CPUs" << std::endl;
#endif

    data = new Tensor<float>(host_data->get_shape(), {NONE, {}}, training_memory_type);
    labels = new Tensor<float>(host_labels->get_shape(), {NONE, {}}, training_memory_type);

    data->copy_from(*host_data);
    labels->copy_from(*host_labels);
    delete host_data;
    delete host_labels;

    // read labels
    label_names = get_class_names(data_set_root + "/batches.meta.txt");

    model::nn_params_t params;
    params.batch_size = 128;
    params.n_epochs = 30;
    params.learning_rate = 0.05;

    auto x_batch =
        op::var<float>("x_batch", {params.batch_size, 3, image_height, image_width}, {NONE, {}}, training_memory_type);

    auto input_layer = layer::input<float>(x_batch);
    auto conv2d1 = layer::conv2d<float>(input_layer->out(), {5, 5}, 32, {0, 0}, {1, 1}, {1, 1}, true, false);
    auto act1 = layer::activation<float>(conv2d1->out(), layer::RELU);
    auto pool1 = layer::pooling<float>(act1->out(), {2, 2}, {0, 0}, {2, 2}, {1, 1}, MAX_POOL);
    auto dropout1 = layer::dropout<float>(pool1->out(), 0.25);

    auto flatten = layer::flatten<float>(dropout1->out());

    auto fc1 = layer::fullyconnected<float>(flatten->out(), 128, true);
    auto act2 = layer::activation<float>(fc1->out(), layer::RELU);
    auto fc2 = layer::fullyconnected<float>(act2->out(), n_classes, false);
    auto act3 = layer::activation<float>(fc2->out(), layer::SOFTMAX);

    auto output = layer::output<float>(act3->out());

    std::vector<layer::Layer<float>*> layers = {input_layer, conv2d1, act1, pool1, dropout1, flatten,
                                                fc1,         act2,    fc2,  act3,  output};

    model::NeuralNetwork<float> model(layers, optimizer::CROSS_ENTROPY, optimizer::SGD, params);

    int32_t input;
    std::cout << "\n==== Cifar Interactive ====\n";
    do {
        std::cout << "\ncurrently using data batch: " << cur_batch << "\n";
        std::cout << "Options:\n";
        std::cout << "\t-1. EXIT\n";
        std::cout << "\t 0. Show Sample\n";
        std::cout << "\t 1. Train\n";
        std::cout << "\t 2. Predict\n";
        std::cout << "\t 3. Switch Batch\n";

        std::cout << "your option:  ";
        std::cin >> input;
        std::cout << "\n";

        if (std::cin.fail()) {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

            input = std::numeric_limits<int32_t>::max();
        }

        switch (input) {
            case -1:
                break;
            case 0:
                show_image(data, labels, label_names);
                break;
            case 1:
                train(model, data, labels);
                break;
            case 2:
                predict(model, data, labels, label_names);
                break;
            case 3:
                delete data;
                delete labels;
                switch_batch(&data, &labels, cur_batch, data_set_root, n_images, image_width, image_height, n_channels,
                             n_classes, true);
                break;
            default:
                std::cout << "Invalid Option. Try Again.\n";
                break;
        }
    } while (input != -1);

    delete data;
    delete labels;

    magmadnn_finalize();
}

void show_image(const Tensor<float>* data, const Tensor<float>* labels, const std::vector<std::string>& label_names) {
    int32_t index;
    std::cout << "Image Index:  ";
    std::cin >> index;
    std::cout << "\n";

    print_image(index, data, labels, label_names);
}

void train(model::NeuralNetwork<float>& model, Tensor<float>* data, Tensor<float>* labels) {
    model::metric_t metrics;

    std::cout << "\n";

    model.fit(data, labels, metrics, true);
}

void predict(model::NeuralNetwork<float>& model, Tensor<float>* images, Tensor<float>* labels,
             const std::vector<std::string>& label_names) {
    uint32_t image_index = 0, predicted_class;

    uint32_t n_channels = images->get_shape(1);
    uint32_t image_height = images->get_shape(2);
    uint32_t image_width = images->get_shape(3);

    std::cout << "Enter image index:  ";
    std::cin >> image_index;
    std::cout << "\n";

    Tensor<float> sample({n_channels * image_height * image_width}, {NONE, {}}, images->get_memory_type());

    sample.copy_from(*images, image_index * sample.get_size(), sample.get_size());

    Tensor<float>* probas = model.predict(&sample);
    predicted_class = model.predict_class(&sample);

    std::cout << "Confidence(s):\n";
    for (unsigned int i = 0; i < labels->get_shape(1); i++) std::cout << std::setfill('-') << std::setw(14) << "-";
    std::cout << "\n|";
    for (unsigned int i = 0; i < labels->get_shape(1); i++) {
        std::cout << std::setfill(' ') << std::setw(10) << label_names.at(i) << " "
                  << " | ";
    }
    std::cout << "\n";
    for (unsigned int i = 0; i < labels->get_shape(1); i++) std::cout << std::setfill('-') << std::setw(14) << "-";
    std::cout << "\n|";
    for (unsigned int i = 0; i < labels->get_shape(1); i++) {
        std::cout << std::fixed << std::setprecision(9) << probas->get(i) << " | ";
    }
    std::cout << "\n";
    for (unsigned int i = 0; i < labels->get_shape(1); i++) std::cout << std::setfill('-') << std::setw(14) << "-";
    std::cout << "\n";

    std::cout << "predicted class = " << label_names.at(predicted_class) << "\n";
    print_image(image_index, images, labels, label_names);
}

void switch_batch(Tensor<float>** data, Tensor<float>** labels, uint32_t& cur_batch, const std::string& cifar_root,
                  uint32_t& n_images, uint32_t& image_width, uint32_t& image_height, uint32_t& n_channels,
                  uint32_t& n_classes, bool normalize) {
    std::cout << "batch idx:  ";
    std::cin >> cur_batch;
    std::cout << "\n";

    load_cifar_batch(cur_batch, cifar_root, data, labels, n_images, image_width, image_height, n_channels, n_classes,
                     normalize);
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

std::vector<std::string> get_class_names(const std::string& file_name) {
    std::ifstream fin(file_name);
    std::string tmp;
    std::vector<std::string> vals;

    while (fin >> tmp) {
        vals.push_back(tmp);
    }
    return vals;
}

void print_image(uint32_t idx, const Tensor<float>* data, const Tensor<float>* labels,
                 const std::vector<std::string>& label_names) {
    uint8_t label = 0;
    uint32_t n_classes = labels->get_shape(1);
    // uint32_t n_channels = data->get_shape(1);
    uint32_t n_rows = data->get_shape(2), n_cols = data->get_shape(3);

    if (idx > data->get_shape(0)) {
        std::cout << "index too large.\n";
        return;
    }

    /* get the correct class */
    for (uint32_t i = 0; i < n_classes; i++) {
        if (std::fabs(labels->get(idx * n_classes + i) - 1.0f) <= 1E-8) {
            label = i;
            break;
        }
    }

    std::printf("Image[%u] is %s.\n", idx, label_names.at(label).c_str());

    for (uint32_t r = 0; r < n_rows; r++) {
        for (uint32_t c = 0; c < n_cols; c++) {
            float red_val = data->get({idx, static_cast<uint32_t>(0), r, c});
            float green_val = data->get({idx, static_cast<uint32_t>(1), r, c});
            float blue_val = data->get({idx, static_cast<uint32_t>(2), r, c});

            uint8_t red_int = (uint8_t)((red_val + 1.0f) * 128.0f);
            uint8_t green_int = (uint8_t)((green_val + 1.0f) * 128.0f);
            uint8_t blue_int = (uint8_t)((blue_val + 1.0f) * 128.0f);

            // std::printf("\x1B[48;5;255m\x1B[38;2;%d;%d;%dm%3u\x1B[0m", red_int, green_int, blue_int, red_int);
            std::printf("\x1B[48;2;%d;%d;%dm  \x1B[0m", red_int, green_int, blue_int);
        }
        std::printf("\n");
    }
}
