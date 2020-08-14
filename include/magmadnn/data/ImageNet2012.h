#pragma once

#include "magmadnn/data/Dataset.h"
#include "magmadnn/exception.h"

#include <vector>
#include <fstream>

namespace magmadnn {
namespace data {

template <typename T>
class ImageNet2012 : public Dataset<T> {
public:

   explicit ImageNet2012(
         std::string const& root, dataset_type type,
         int const height, int const width,
         std::string const& imagenet_labels);

   std::vector<std::string> class_names;
};
   
}} // End of namespace magmadnn::data
