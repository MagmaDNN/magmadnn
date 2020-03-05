#include "magmadnn/data/MNIST.h"

#include "magmadnn/types.h"
#include "tensor/tensor.h"
#include "magmadnn/data/utils.h"

// STD
#include <cassert>

namespace magmadnn {
namespace data {

template <typename T>
MNIST<T>::MNIST(std::string const& root, dataset_type type)
{

   std::string images_filename;
   std::string labels_filename;

   if (type == dataset_type::Test) {
      images_filename = root + "/t10k-images-idx3-ubyte";
      labels_filename = root + "/t10k-labels-idx1-ubyte";
   }
   else {
      images_filename = root + "/train-images-idx3-ubyte";
      labels_filename = root + "/train-labels-idx1-ubyte";
   }

   magmadnn::Tensor<T> *mnist_images = nullptr;
   magmadnn::Tensor<T> *mnist_labels = nullptr;

   this->nchanels(1);
   this->nclasses(10);

   mnist_images = read_mnist_images(images_filename.c_str(), this->nimages_, this->nrows_, this->ncols_);
   mnist_labels = read_mnist_labels(labels_filename.c_str(), this->nlabels_, this->nclasses_);

   assert((mnist_images != nullptr) && (mnist_labels != nullptr));
   assert((this->nrows_ > 0) && (this->ncols_ > 0));
   assert((this->nimages_ > 0) && (this->nlabels_ > 0) && (this->nclasses_ > 0));
   
   this->images_.reset(mnist_images);
   this->labels_.reset(mnist_labels);

}

template <typename T>
void MNIST<T>::print_image(uint32_t idx)
{
   mnist_print_image(
         idx, &this->images(), &this->labels(), this->nrows(), this->ncols()); 
}

// template class MNIST<int>;
template class MNIST<float>;
// template class MNIST<double>;
   
}} // End of namespace magmadnn::data
