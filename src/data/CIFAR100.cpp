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
      
   std::string data_filename;

   if (type == dataset_type::Test) {
      data_filename = root + "/test.bin";
   }
   else {
      data_filename = root + "/train.bin";
   }

   magmadnn::Tensor<T> *cifar100_images = nullptr;
   magmadnn::Tensor<T> *cifar100_labels = nullptr;

   read_cifar100(
         data_filename, &cifar100_images, &cifar100_labels,
         this->nimages_, this->ncols_, this->nrows_,
         this->nchanels_, this->nclasses_, this->nsuperclasses_);

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
