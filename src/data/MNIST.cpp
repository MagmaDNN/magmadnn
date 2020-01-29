#include "magmadnn/data/MNIST.h"

#include "magmadnn/types.h"
#include "tensor/tensor.h"
#include "magmadnn/data/utils.h"

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

   magmadnn::Tensor<T> mnist_images = nullptr;
   magmadnn::Tensor<T> mnist_labels = nullptr;

   mnist_images = read_mnist_images("train-images-idx3-ubyte", this->nimages_, this->nrows_, this->ncols_);
   mnist_labels = read_mnist_labels("train-labels-idx1-ubyte", this->nlabels_, this->nclasses_);
   
   this->images_.reset(mnist_images);
   this->labels_.reset(mnist_labels);

}
   
}} // End of namespace magmadnn::data
