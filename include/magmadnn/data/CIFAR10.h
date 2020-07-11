#pragma once

#include "magmadnn/data/Dataset.h"
#include "magmadnn/exception.h"

namespace magmadnn {
namespace data {
   
   template <typename T>
   class CIFAR10 : public Dataset<T> {
   public:

      explicit CIFAR10(std::string const& root, dataset_type type)
         : CIFAR10(root, type, 1)
      {}
      
      explicit CIFAR10(std::string const& root, dataset_type type, uint32_t batch_idx);

   };
   
}} // End of namespace magmadnn::data
