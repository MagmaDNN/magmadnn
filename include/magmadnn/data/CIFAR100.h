#pragma once

#include "magmadnn/data/Dataset.h"
#include "magmadnn/exception.h"

namespace magmadnn {
namespace data {

template <typename T>
class CIFAR100 : public Dataset<T> {
public:
   
   explicit CIFAR100(std::string const& root, dataset_type type);

   /* Return the number of classes for this dataset
    */
   uint32_t nsuperclasses() const {
      return this->nsuperclasses_;
   }

   void nsuperclasses(uint32_t in_nsuperclasses) {
      this->nsuperclasses_ = in_nsuperclasses;
   }
   
private:

   uint32_t nsuperclasses_;
   
};
   
}} // End of namespace magmadnn::data
