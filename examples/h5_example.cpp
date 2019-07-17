#include <math.h>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>
#include "H5Cpp.h"
#include "magmadnn.h"

using namespace magmadnn;
using namespace H5;

struct data {
    Tensor<float> *images;
    Tensor<float> *labels;
};

bool is_file_exist(H5std_string fileName);
data read_cbed_data(int &file_cntr, int num_labels);

int main() {
    magmadnn_init();

    int num_labels = 230;

    model::nn_params_t params;
    params.batch_size = 32;
    params.n_epochs = 5;
    params.learning_rate = 0.0001;

    auto x_batch = op::var<float>("x_batch", {params.batch_size, 3, 512, 512}, {NONE, {}}, DEVICE);

    auto input = layer::input<float>(x_batch);

    auto pool0 = layer::pooling<float>(input->out(), {4, 4}, {0, 0}, {4, 4}, MAX_POOL);

    auto conv2d1 = layer::conv2d<float>(pool0->out(), {5, 5}, 32, {0, 0}, {2, 2}, {1, 1}, true, false);
    auto act1 = layer::activation<float>(conv2d1->out(), layer::RELU);
    auto pool1 = layer::pooling<float>(act1->out(), {2, 2}, {0, 0}, {2, 2}, MAX_POOL);
    auto dropout1 = layer::dropout<float>(pool1->out(), 0.25);

    auto flatten = layer::flatten<float>(dropout1->out());

    auto fc1 = layer::fullyconnected<float>(flatten->out(), 512, true);
    auto act2 = layer::activation<float>(fc1->out(), layer::RELU);
    auto fc2 = layer::fullyconnected<float>(act2->out(), num_labels, false);
    auto act3 = layer::activation<float>(fc2->out(), layer::SOFTMAX);

    auto output = layer::output<float>(act3->out());

    std::vector<layer::Layer<float> *> layers = {input,   pool0, conv2d1, act1, pool1, dropout1,
                                                 flatten, fc1,   act2,    fc2,  act3,  output};

    model::NeuralNetwork<float> model(layers, optimizer::CROSS_ENTROPY, optimizer::SGD, params);

    model::metric_t metrics;

    int file_cntr = 0;
    for (int i = 0; i < 20; i++) {
        data d = read_cbed_data(file_cntr, num_labels);
        Tensor<float> *images = d.images;
        Tensor<float> *labels = d.labels;
        for (int i = 0; i < 10; i++) std::cout << images->get(i) << " ";
        std::cout << "\n";
        model.fit(images, labels, metrics, true);
    }

    magmadnn_finalize();
}

bool is_file_exist(H5std_string fileName) {
    std::ifstream infile(fileName);
    return infile.good();
}

data read_cbed_data(int &file_cntr, int num_labels) {
    const H5std_string FILE_NAME("/home/user1/train/batch_train_");
    int cntr = 0;

    Tensor<float> *images = new Tensor<float>({1400, 3, 512, 512}, {NONE, {}}, HOST);
    Tensor<float> *labels = new Tensor<float>({1400, num_labels}, {NONE, {}}, HOST);
    data grouped_data = {images, labels};

    for (int i = file_cntr; i < file_cntr + 30; i++, file_cntr++) {
        const H5std_string FILE_NAME_I(FILE_NAME + std::to_string(i) + ".h5");
        if (!is_file_exist(FILE_NAME_I)) {
            std::cout << FILE_NAME_I << " does not exist.\n";
            continue;
        }
        H5File file(FILE_NAME_I, H5F_ACC_RDONLY);
        Group g[file.getNumObjs()];
        DataSet d[file.getNumObjs()];
        Attribute a[file.getNumObjs()];

        std::cout << "file name: " << file.getFileName() << "\n";
        std::cout << "number of objects: " << file.getNumObjs() << "\n";

        for (unsigned j = 0; j < file.getNumObjs(); j++) {
            g[j] = file.openGroup(file.getObjnameByIdx(j));
            d[j] = g[j].openDataSet("cbed_stack");
            a[j] = g[j].openAttribute("space_group");

            float cbed[3][512][512];
            d[j].read(cbed, PredType::NATIVE_FLOAT);

            StrType stype = a[j].getStrType();
            std::string x = "";
            a[j].read(stype, x);

            for (int a = 0; a < 3; a++) {
                for (int b = 0; b < 512; b++) {
                    for (int c = 0; c < 512; c++) {
                        int num = cntr * 512 * 512 * 3 + a * 512 * 512 + b * 512 + c;
                        images->set(num, log(cbed[a][b][c]) / double(-50.0));
                    }
                }
            }

            int label;
            if (stoi(x) >= 1 && stoi(x) <= 2)
                label = 0;
            else if (stoi(x) >= 3 && stoi(x) <= 15)
                label = 1;
            else if (stoi(x) >= 16 && stoi(x) <= 74)
                label = 2;
            else if (stoi(x) >= 75 && stoi(x) <= 142)
                label = 3;
            else if (stoi(x) >= 143 && stoi(x) <= 167)
                label = 4;
            else if (stoi(x) >= 168 && stoi(x) <= 194)
                label = 5;
            else
                label = 6;
            for (int a = 0; a < num_labels; a++) {
                int num = cntr * num_labels + a;
                if (label == a) {
                    labels->set(num, 1.0f);
                } else {
                    labels->set(num, 0.0f);
                }
            }

            cntr++;
            if (cntr == 1400) return grouped_data;
        }

        file.close();

        std::printf("File %d closed.\n", i);
    }

    return grouped_data;
}