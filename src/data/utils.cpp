#include "tensor/tensor.h"

namespace magmadnn {
namespace data {
   
#define FREAD_CHECK(res, nmemb)           \
    if ((res) != (nmemb)) {               \
        fprintf(stderr, "fread fail.\n"); \
        return NULL;                      \
    }

inline void endian_swap(uint32_t& val) {
    /* taken from https://stackoverflow.com/questions/13001183/how-to-read-little-endian-integers-from-file-in-c */
    val = (val >> 24) | ((val << 8) & 0xff0000) | ((val >> 8) & 0xff00) | (val << 24);
}

magmadnn::Tensor<float>* read_mnist_images(const char* file_name, uint32_t& n_images, uint32_t& n_rows,
                                           uint32_t& n_cols) {
    FILE* file;
    unsigned char magic[4];
    magmadnn::Tensor<float>* data;
    uint8_t val;

    file = std::fopen(file_name, "r");

    if (file == NULL) {
        std::fprintf(stderr, "Could not open %s for reading.\n", file_name);
        return NULL;
    }

    FREAD_CHECK(fread(magic, sizeof(char), 4, file), 4);
    if (magic[2] != 0x08 || magic[3] != 0x03) {
        std::fprintf(stderr, "Bad file magic.\n");
        return NULL;
    }

    FREAD_CHECK(fread(&n_images, sizeof(uint32_t), 1, file), 1);
    endian_swap(n_images);

    FREAD_CHECK(fread(&n_rows, sizeof(uint32_t), 1, file), 1);
    endian_swap(n_rows);

    FREAD_CHECK(fread(&n_cols, sizeof(uint32_t), 1, file), 1);
    endian_swap(n_cols);

    printf("Preparing to read %u images with size %u x %u ...\n", n_images, n_rows, n_cols);

    char bytes[n_rows * n_cols];

    /* allocate tensor */
    data = new magmadnn::Tensor<float>({n_images, n_rows, n_cols}, {magmadnn::NONE, {}}, magmadnn::HOST);

    for (uint32_t i = 0; i < n_images; i++) {
        FREAD_CHECK(fread(bytes, sizeof(char), n_rows * n_cols, file), n_rows * n_cols);

        for (uint32_t r = 0; r < n_rows; r++) {
            for (uint32_t c = 0; c < n_cols; c++) {
                val = bytes[r * n_cols + c];

                data->set(i * n_rows * n_cols + r * n_cols + c, (val / 128.0f) - 1.0f);
            }
        }
    }
    printf("finished reading images.\n");

    fclose(file);

    return data;
}

magmadnn::Tensor<float>* read_mnist_labels(const char* file_name, uint32_t& n_labels, uint32_t n_classes) {
    FILE* file;
    unsigned char magic[4];
    magmadnn::Tensor<float>* labels;
    uint8_t val;

    file = std::fopen(file_name, "r");

    if (file == NULL) {
        std::fprintf(stderr, "Could not open %s for reading.\n", file_name);
        return NULL;
    }

    FREAD_CHECK(fread(magic, sizeof(char), 4, file), 4);

    if (magic[2] != 0x08 || magic[3] != 0x01) {
        std::fprintf(stderr, "Bad file magic.\n");
        return NULL;
    }

    FREAD_CHECK(fread(&n_labels, sizeof(uint32_t), 1, file), 1);
    endian_swap(n_labels);

    printf("Preparing to read %u labels with %u classes ...\n", n_labels, n_classes);

    /* allocate tensor */
    labels = new magmadnn::Tensor<float>({n_labels, n_classes}, {magmadnn::ZERO, {}}, magmadnn::HOST);

    printf("finished reading labels.\n");

    for (unsigned int i = 0; i < n_labels; i++) {
        FREAD_CHECK(fread(&val, sizeof(char), 1, file), 1);

        labels->set(i * n_classes + val, 1.0f);
    }

    fclose(file);

    return labels;
}

template <typename T>
void mnist_print_image(uint32_t image_idx, magmadnn::Tensor<T>* images, magmadnn::Tensor<T>* labels, uint32_t n_rows,
                       uint32_t n_cols) {
    uint8_t label = 0;
    uint32_t n_classes = labels->get_shape(1);

    /* assign the label */
    for (uint32_t i = 0; i < n_classes; i++) {
        if (std::fabs(labels->get(image_idx * n_classes + i) - 1.0f) <= 1E-8) {
            label = i;
            break;
        }
    }

    printf("Image[%u] is digit %u:\n", image_idx, label);

    for (unsigned int r = 0; r < n_rows; r++) {
        for (unsigned int c = 0; c < n_cols; c++) {
            printf("%3u ", (uint8_t)((images->get(image_idx * n_rows * n_cols + r * n_cols + c) + 1.0f) * 128.0f));
        }
        printf("\n");
    }
}

template void mnist_print_image<float>(uint32_t image_idx, magmadnn::Tensor<float>* images,
                                       magmadnn::Tensor<float>* labels, uint32_t n_rows, uint32_t n_cols);

template void mnist_print_image<double>(uint32_t image_idx, magmadnn::Tensor<double>* images,
                                        magmadnn::Tensor<double>* labels, uint32_t n_rows, uint32_t n_cols);

#undef FREAD_CHECK

////////////////////////////////////////////////////////////
// CIFAR https://www.cs.toronto.edu/~kriz/cifar.html

#define FREAD_CHECK(res, nmemb, ret)           \
    if ((res) != (nmemb)) {                    \
        std::fprintf(stderr, "fread fail.\n"); \
        return ret;                            \
    }

magmadnn::magmadnn_error_t read_cifar100(const std::string& file_name, magmadnn::Tensor<float>** data,
                                         magmadnn::Tensor<float>** labels, uint32_t n_images, uint32_t& image_width,
                                         uint32_t& image_height, uint32_t& n_channels, uint32_t& n_classes,
                                         uint32_t& n_super_classes /*, bool normalize = true*/) {
    // n_images = 50000; /* training set */
    // n_images = 10000; /* test set */
    image_width = 32;
    image_height = 32;
    n_channels = 3;
    n_classes = 100;
    n_super_classes = 20;

    *data = new magmadnn::Tensor<float>({n_images, n_channels, image_height, image_width}, {magmadnn::NONE, {}},
                                        magmadnn::HOST);
    *labels = new magmadnn::Tensor<float>({n_images, n_classes}, {magmadnn::ZERO, {}}, magmadnn::HOST);

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
        /* course label -- do nothing with it for now */

        FREAD_CHECK(fread(&val, sizeof(uint8_t), 1, file), 1, 1);
        (*labels)->set(i * n_classes + (val), 1.0f);
        /* fine label */

        FREAD_CHECK(fread(img, sizeof(uint8_t), n_channels * image_width * image_height, file),
                    n_channels * image_width * image_height, 1);

        for (uint32_t j = 0; j < n_channels * image_height * image_width; j++) {
            float normalized_val = (img[j] / 128.0f) - 1.0f;

            (*data)->set(i * n_channels * image_height * image_width + j, normalized_val);
        }
    }

    delete img;

    return (magmadnn::magmadnn_error_t) 0;
}

magmadnn::magmadnn_error_t read_cifar100_train(const std::string& file_name, magmadnn::Tensor<float>** data,
                                               magmadnn::Tensor<float>** labels, uint32_t& n_images,
                                               uint32_t& image_width, uint32_t& image_height, uint32_t& n_channels,
                                               uint32_t& n_classes,
                                               uint32_t& n_super_classes /*, bool normalize = true*/) {
    n_images = 50000;

    return read_cifar100(file_name, data, labels, n_images, image_width, image_height, n_channels, n_classes,
                         n_super_classes /*, bool normalize = true*/);
}

magmadnn::magmadnn_error_t read_cifar100_test(const std::string& file_name, magmadnn::Tensor<float>** data,
                                              magmadnn::Tensor<float>** labels, uint32_t& n_images,
                                              uint32_t& image_width, uint32_t& image_height, uint32_t& n_channels,
                                              uint32_t& n_classes,
                                              uint32_t& n_super_classes /*, bool normalize = true*/) {
    n_images = 10000;

    return read_cifar100(file_name, data, labels, n_images, image_width, image_height, n_channels, n_classes,
                         n_super_classes /*, bool normalize = true*/);
}

magmadnn::magmadnn_error_t load_cifar100(const std::string& cifar_root, magmadnn::Tensor<float>** data,
                                         magmadnn::Tensor<float>** labels, uint32_t& n_images, uint32_t& image_width,
                                         uint32_t& image_height, uint32_t& n_channels, uint32_t& n_classes,
                                         uint32_t& n_super_classes /*,
                                                                     bool normalize = true*/
) {
    return read_cifar100(cifar_root + "/train.bin", data, labels, n_images, image_width, image_height, n_channels,
                         n_classes, n_super_classes /*, normalize*/);
}

magmadnn::magmadnn_error_t read_cifar10(const std::string& file_name, magmadnn::Tensor<float>** data,
                                        magmadnn::Tensor<float>** labels, uint32_t& n_images, uint32_t& image_width,
                                        uint32_t& image_height, uint32_t& n_channels,
                                        uint32_t& n_classes /*, bool normalize*/) {
    n_images = 10000; /* magic numbers from cifar10 file format */
    image_width = 32;
    image_height = 32;
    n_channels = 3;
    n_classes = 10;

    *data = new magmadnn::Tensor<float>({n_images, n_channels, image_height, image_width}, {magmadnn::NONE, {}},
                                        magmadnn::HOST);
    *labels = new magmadnn::Tensor<float>({n_images, n_classes}, {magmadnn::ZERO, {}}, magmadnn::HOST);

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

    return (magmadnn::magmadnn_error_t) 0;
}

magmadnn::magmadnn_error_t load_cifar10_batch(
      uint32_t batch_idx, const std::string& cifar_root,
      magmadnn::Tensor<float>** data,
      magmadnn::Tensor<float>** labels,
      uint32_t& n_images, uint32_t& image_width,
      uint32_t& image_height, uint32_t& n_channels, uint32_t& n_classes/*,
      bool normalize*/) {
    return read_cifar10(cifar_root + "/data_batch_" + std::to_string(batch_idx) + ".bin", data, labels, n_images,
                        image_width, image_height, n_channels, n_classes /*, normalize*/);
}

#undef FREAD_CHECK

}  // namespace data
}  // namespace magmadnn
