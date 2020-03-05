#include "magmadnn/data/CIFAR100.h"
#include "magmadnn/data/utils.h"

// STD
#include <cassert>
#include <iostream>

namespace magmadnn {
namespace data {

template <typename T>
CIFAR100<T>::CIFAR100(std::string const& root, dataset_type type)
{

   magmadnn::magmadnn_error_t err = 1;
      
   magmadnn::Tensor<T> *cifar100_images = nullptr;
   magmadnn::Tensor<T> *cifar100_labels = nullptr;

   std::string data_filename;

   if (type == dataset_type::Test) {
      // Test dataset filename
      data_filename = root + "/test.bin";

      err = read_cifar100_test(
            data_filename, &cifar100_images, &cifar100_labels,
            this->nimages_, this->ncols_, this->nrows_,
            this->nchanels_, this->nclasses_, this->nsuperclasses_);

   }
   else {
      // Training dataset filename
      data_filename = root + "/train.bin";

      err = read_cifar100_train(
            data_filename, &cifar100_images, &cifar100_labels,
            this->nimages_, this->ncols_, this->nrows_,
            this->nchanels_, this->nclasses_, this->nsuperclasses_);
   }

   // std::cout << "" << std::endl;
   if (err != static_cast<magmadnn::magmadnn_error_t>(0)) {
      throw ::magmadnn::Error(
            __FILE__, __LINE__,
            "Error when loading CIFAR100 dataset in the following directory: " + root);      
   }

   assert((cifar100_images != nullptr) && (cifar100_labels != nullptr));
   assert((this->nrows_ > 0) && (this->ncols_ > 0));

   this->nlabels(this->nimages_);
         
   this->images_.reset(cifar100_images);
   this->labels_.reset(cifar100_labels);

   // std::cout << "Number of images = " << this->nimages() << std::endl;
   // std::cout << "Image size = " << this->nrows() << "x" << this->ncols()<< std::endl;
   // std::cout << "Number of chanels = " << this->nchanels() << std::endl;
   
   // std::cout << "data_filename = " << data_filename << std::endl;
}

template class CIFAR100<float>;
   
}} // End of namespace magmadnn::data
