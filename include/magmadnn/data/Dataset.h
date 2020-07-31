#pragma once

#include <memory>

#include "magmadnn/types.h"
#include "tensor/tensor.h"

namespace magmadnn {
namespace data {

/* Type of dataset:
   Test: test set
   Train: training set
*/
enum dataset_type { Test, Train };

template <typename T>
class Dataset {
   public:
    explicit Dataset()
        : images_(nullptr),
          labels_(nullptr),
          nimages_(0),
          nlabels_(0),
          nchanels_(0),
          nrows_(0),
          ncols_(0),
          nclasses_(0) {}

    /* Size of the dataset
     */
    // magmadnn::size_type size() const {
    //    return this->size_;
    // };

    /* Return a tensor containing images from the MNIST dataset
     */
    magmadnn::Tensor<T> const& images() const { return *this->images_; }

    magmadnn::Tensor<T>& images() { return *this->images_; }

    uint32_t nimages() const { return this->nimages_; }

    uint32_t nlabels() const { return this->nlabels_; }

    void nlabels(uint32_t in_nlabels) { this->nlabels_ = in_nlabels; }

    /* Return a tensor containing lables from the MNIST dataset
     */
    magmadnn::Tensor<T> const& labels() const { return *this->labels_; }

    magmadnn::Tensor<T>& labels() { return *this->labels_; }

    /* Return the number of classes for this dataset
     */
    uint32_t nclasses() const { return this->nclasses_; }

    void nclasses(uint32_t in_nclasses) { this->nclasses_ = in_nclasses; }

    uint32_t nchanels() const { return this->nchanels_; }

    void nchanels(uint32_t in_nchanels) { this->nchanels_ = in_nchanels; }

    uint32_t nrows() const { return this->nrows_; }

    uint32_t ncols() const { return this->ncols_; }

   protected:
    // magmadnn::size_type size_;

    uint32_t nimages_;
    uint32_t nlabels_;

    uint32_t nchanels_;
    uint32_t nrows_;
    uint32_t ncols_;

    uint32_t nclasses_;

    std::unique_ptr<magmadnn::Tensor<T>> images_;  // Image samples
    std::unique_ptr<magmadnn::Tensor<T>> labels_;  // Corresponding labels
};

}  // namespace data
}  // namespace magmadnn
