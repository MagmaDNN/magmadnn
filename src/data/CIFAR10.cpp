#include "magmadnn/data/CIFAR10.h"
#include "magmadnn/data/utils.h"

// STD
#include <cassert>
#include <iostream>

namespace magmadnn {
namespace data {

template <typename T>
CIFAR10<T>::CIFAR10(std::string const& root, dataset_type type, uint32_t batch_idx)
{
   
   if ((batch_idx < 1) || (batch_idx > 5)) {
      throw ::magmadnn::Error(
            __FILE__, __LINE__,
            "No dataset associated with batch index: " + std::to_string(batch_idx));
   }
   
   std::string data_filename;

   if (type == dataset_type::Test) {
      data_filename = root + "/test_batch.bin";
   }
   else {
      data_filename = root + "/data_batch_" + std::to_string(batch_idx) + ".bin";
   }

   magmadnn::Tensor<T> *cifar10_images = nullptr;
   magmadnn::Tensor<T> *cifar10_labels = nullptr;
   
   magmadnn::magmadnn_error_t err = read_cifar10(
         data_filename, &cifar10_images, &cifar10_labels, 
         this->nimages_, this->ncols_, this->nrows_,
         this->nchanels_, this->nclasses_/*, bool normalize*/);

   if (err != static_cast<magmadnn::magmadnn_error_t>(0)) {
      throw ::magmadnn::Error(
            __FILE__, __LINE__,
            "Could not find CIFAR10 dataset in the following directory: " + root);
   }
   
   assert((cifar10_images != nullptr) && (cifar10_labels != nullptr));
   assert((this->nrows_ > 0) && (this->ncols_ > 0));

   this->nlabels(this->nimages_);
         
   this->images_.reset(cifar10_images);
   this->labels_.reset(cifar10_labels);

   // std::cout << "Number of images = " << this->nimages() << std::endl;
   // std::cout << "Image size = " << this->nrows() << "x" << this->ncols()<< std::endl;
   // std::cout << "Number of chanels = " << this->nchanels() << std::endl;
   
   // std::cout << "data_filename = " << data_filename << std::endl;
}

template class CIFAR10<float>;
   
}} // End of namespace magmadnn::data
