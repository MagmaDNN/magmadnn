#pragma once

#include "magmadnn/data/Dataset.h"
#include "magmadnn/types.h"
#include "tensor/tensor.h"

namespace magmadnn {
namespace data {

   template <typename T>
   class MNIST : Dataset<T> {
   public:

      explicit MNIST(std::string const& root, dataset_type type);
      
   };

}} // End of namespace magmadnn::data
      
