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
   enum dataset_type {Test, Train};
   
   template <typename T>
   class Dataset {
   public:

      explicit Dataset() : images_(nullptr), labels_(nullptr) {}
         
      /* Size of the dataset 
       */
      // magmadnn::size_type size() const {
      //    return this->size_;
      // };

      /* Return a tensor containing images from the MNIST dataset
       */
      magmadnn::Tensor<T> const& images() const {
         return *this->images_;
      }

      int num_images() const {
         return this->nimages_;
      }

      int num_labels() const {
         return this->nlabels_;
      }
      
      /* Return a tensor containing lables from the MNIST dataset
       */
      magmadnn::Tensor<T> const& labels() const {
         return *this->labels_;
      }

      /* Return the number of classes for this dataset
       */
      int nclasses() const {
         return this->nclasses;
      }
      
   protected:
      // magmadnn::size_type size_;

      int nimages_;
      int nlabels_;

      int nrows_;
      int ncols_;
      
      int nclasses_;
      
      std::unique_ptr<magmadnn::Tensor<T>> images_; // Image samples
      std::unique_ptr<magmadnn::Tensor<T>> labels_; // Corresponding labels
   };  
   
}} // End of namespace magmadnn::data
