#pragma once

#include "magmadnn/data/Dataset.h"
#include "magmadnn/types.h"
#include "tensor/tensor.h"

namespace magmadnn {
namespace data {

template <typename T>
class MNIST : public Dataset<T> {
   public:
    explicit MNIST(std::string const& root, dataset_type type);

    // Print image index `idx` from the dataset
    void print_image(uint32_t idx);
};

}  // namespace data
}  // namespace magmadnn
