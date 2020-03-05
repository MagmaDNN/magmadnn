#pragma once

#include "tensor/tensor.h"

namespace magmadnn {
namespace data {

// MNIST    

magmadnn::Tensor<float> *read_mnist_images(const char *file_name, uint32_t &n_images, uint32_t &n_rows, uint32_t &n_cols);

magmadnn::Tensor<float> *read_mnist_labels(const char *file_name, uint32_t &n_labels, uint32_t n_classes);

void print_image(uint32_t image_idx, magmadnn::Tensor<float> *images, magmadnn::Tensor<float> *labels, uint32_t n_rows, uint32_t n_cols);

// CIFAR

magmadnn::magmadnn_error_t read_cifar100(
      const std::string& file_name,
      magmadnn::Tensor<float>** data, magmadnn::Tensor<float>** labels,
      uint32_t n_images, uint32_t& image_width, uint32_t& image_height, uint32_t& n_channels,
      uint32_t& n_classes, uint32_t& n_super_classes/*, bool normalize = true*/);

magmadnn::magmadnn_error_t read_cifar100_train(
      const std::string& file_name, magmadnn::Tensor<float>** data, magmadnn::Tensor<float>** labels,
      uint32_t& n_images, uint32_t& image_width, uint32_t& image_height, uint32_t& n_channels,
      uint32_t& n_classes, uint32_t& n_super_classes/*, bool normalize = true*/);

magmadnn::magmadnn_error_t read_cifar100_test(
      const std::string& file_name, magmadnn::Tensor<float>** data, magmadnn::Tensor<float>** labels,
      uint32_t& n_images, uint32_t& image_width, uint32_t& image_height, uint32_t& n_channels,
      uint32_t& n_classes, uint32_t& n_super_classes/*, bool normalize = true*/);
   
magmadnn::magmadnn_error_t load_cifar100(
      const std::string& cifar_root,
      magmadnn::Tensor<float>** data,
      magmadnn::Tensor<float>** labels,
      uint32_t& n_images,
      uint32_t& image_width,
      uint32_t& image_height,
      uint32_t& n_channels,
      uint32_t& n_classes,
      uint32_t& n_super_classes/*,
                                 bool normalize = true*/
      );
 
magmadnn::magmadnn_error_t read_cifar10(
      const std::string& file_name,
      magmadnn::Tensor<float>** data, magmadnn::Tensor<float>** labels,
      uint32_t& n_images, uint32_t& image_width, uint32_t& image_height,
      uint32_t& n_channels,
      uint32_t& n_classes/*, bool normalize*/);  

magmadnn::magmadnn_error_t load_cifar10_batch(
      uint32_t batch_idx, const std::string& cifar_root,
      magmadnn::Tensor<float>** data,
      magmadnn::Tensor<float>** labels,
      uint32_t& n_images, uint32_t& image_width,
      uint32_t& image_height, uint32_t& n_channels, uint32_t& n_classes/*,
                                                                         bool normalize*/);
   
}} // End of namespace magmadnn::data
