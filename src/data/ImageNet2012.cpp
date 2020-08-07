#include "magmadnn/data/ImageNet2012.h"
#include "magmadnn/data/image_io.h"
#include "tensor/tensor.h"

// STD
#include <cassert>
#include <iostream>
// #include <filesystem>
#include <dirent.h>

namespace magmadnn {
namespace data {

template <typename T>
ImageNet2012<T>::ImageNet2012(
      std::string const& root, dataset_type type,
      int const height, int const width,
      std::string const& imagenet_labels) {

   // std::cout << "[ImageNet2012]" << std::endl;

   if (!imagenet_labels.empty()) {

      // Get collection of class names
      std::ifstream labels_file(imagenet_labels);

      std::string class_name;   
      while (labels_file >> class_name) {

         // std::cout << "[ImageNet2012] label = " << label << std::endl;

         this->class_names.push_back(class_name);
      }

      // Set Number of classes
      this->nclasses_ = this->class_names.size();
      
      // Count the number of images in the dataset
      int num_images = 0;

      DIR *dir;
      struct dirent *ent;
      unsigned char isFile = 0x8;

      for (const auto& class_name: this->class_names) {

         // Directory path for current class
         std::string path = root + "/" + class_name;

         if ((dir = opendir (path.c_str())) != NULL) {
            while ((ent = readdir (dir)) != NULL) {
               if ( ent->d_type == isFile) {
                  // Make sure current iterate is a file.
                  //
                  //TODO: Make sure it is a valid JPEG file.
                  num_images++;
               }
            }
            closedir (dir);
         }
      }

      std::cout << "[ImageNet2012] number of images = " << num_images << std::endl;

      // Set number of images/labels
      this->nlabels_ = num_images;
      this->nimages_ = num_images;
      
      // Initialize tensor pointers
      this->labels_ = nullptr;
      this->images_ = nullptr;

      // Images should be in colors, no alpha channel
      this->nchanels_ = 3;      
      
      if (num_images > 0) {

         // Allocate tensors on host side
         magmadnn::Tensor<T> *imagenet2012_images = nullptr;
         magmadnn::Tensor<T> *imagenet2012_labels = nullptr;

         // Label tensor
         imagenet2012_labels = new magmadnn::Tensor<T>(
               {this->nlabels_, this->nclasses_}, {magmadnn::ZERO, {}}, magmadnn::HOST);
         // Image tensor
         imagenet2012_images = new magmadnn::Tensor<T>(
               {this->nimages_, this->nchanels_, height, width}, {magmadnn::NONE, {}}, magmadnn::HOST);

         this->images_.reset(imagenet2012_images);
         this->labels_.reset(imagenet2012_labels);
         
         // Go through the dataset and fill tensor
         // for (const auto& class_name: this->class_names) {
         for (int cidx = 0; cidx < this->class_names.size(); ++cidx) {

            // Get current class name
            auto const& class_name = this->class_names[cidx];
            // Directory path for current class            
            std::string path = root + "/" + class_name;

            std::cout << "[ImageNet2012] path = " << path << std::endl;
            
            if ((dir = opendir (path.c_str())) != NULL) {
               while ((ent = readdir (dir)) != NULL) {

                  if ( ent->d_type == isFile) {

                     // printf ("%s\n", ent->d_name);
                     std::string image_filename(ent->d_name); 
                     std::cout << "[ImageNet2012] image_filename = " << image_filename << std::endl;
                     std::string image_path = path + "/" + image_filename;
                     std::ifstream image_file(image_path, std::ios::binary | std::ios::ate);
                     size_t fsize = image_file.tellg();
                     image_file.seekg(0, std::ios::beg);
                     std::cout << "[ImageNet2012] fsize = " << fsize << std::endl;
                     std::shared_ptr<uint8_t> buff(new uint8_t[fsize], std::default_delete<uint8_t[]>());
                     image_file.read(reinterpret_cast<char*>(buff.get()), fsize);

                     int64_t width;
                     int64_t height;
            
                     if(get_jpeg_size(buff.get(), fsize, &width, &height)) {
                        std::cout << "[ImageNet2012] width = " << width
                                  << ", height = " << height << std::endl;
                     }
               
                  }
               }
               closedir (dir);
            }
         }
      }
   }
}

   
template class ImageNet2012<float>;

}} // End of namespace magmadnn::data
