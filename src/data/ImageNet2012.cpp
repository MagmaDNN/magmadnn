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
      std::string const& imagenet_labels/*, int const num_images_per_labels=2000*/) {

   // std::cout << "[ImageNet2012]" << std::endl;

   // TODO Add `is_color` as part of the ImageDataset structure
   const bool is_color = true;
      
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

      int64_t original_width;
      int64_t original_height;

      for (const auto& class_name: this->class_names) {

         // Directory path for current class
         std::string path = root + "/" + class_name;

         if ((dir = opendir (path.c_str())) != NULL) {
            while ((ent = readdir (dir)) != NULL) {
               if ( ent->d_type == isFile) {
                  // Make sure current iterate is a file.
                  //
                  //TODO: Make sure it is a valid JPEG file.

                  // std::string image_filename(ent->d_name); 
                  // std::string image_path = path + "/" + image_filename;

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

         this->nrows_ = height;
         this->ncols_ = width;

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

         std::size_t image_idx = 0;
         
         // Go through the dataset and fill tensor
         // for (const auto& class_name: this->class_names) {
         for (int cidx = 0; cidx < this->class_names.size(); ++cidx) {

            auto load_class_sa = std::chrono::high_resolution_clock::now();

            // Get current class name
            auto const& class_name = this->class_names[cidx];
            // Directory path for current class            
            std::string path = root + "/" + class_name;

            std::cout << "[ImageNet2012] path = " << path << std::endl;
            
            if ((dir = opendir (path.c_str())) != NULL) {
               while ((ent = readdir (dir)) != NULL) {

                  if ( ent->d_type == isFile) {

                     std::string image_filename(ent->d_name); 
                     std::string image_path = path + "/" + image_filename;

                     // printf ("%s\n", ent->d_name);
                     // std::cout << "[ImageNet2012] image_filename = " << image_filename << std::endl;
                     // std::ifstream image_file(image_path, std::ios::binary | std::ios::ate);
                     // size_t fsize = image_file.tellg();
                     // image_file.seekg(0, std::ios::beg);
                     // std::cout << "[ImageNet2012] fsize = " << fsize << std::endl;
                     // std::shared_ptr<uint8_t> buff(new uint8_t[fsize], std::default_delete<uint8_t[]>());
                     // image_file.read(reinterpret_cast<char*>(buff.get()), fsize);
            
                     // if(get_jpeg_size(buff.get(), fsize, &original_width, &original_height)) {
                     //    std::cout << "[ImageNet2012] width = " << width
                     //              << ", height = " << height << std::endl;                        
                     // }

                     add_image_to_tensor(
                           image_path, height, width, is_color,
                           this->images_.get(), image_idx);

                     // Set label corresponding to current image to 1.
                     this->labels_->set(image_idx * this->nclasses_ + cidx, T(1));
                     
                     ++image_idx;
                  }
               }
               closedir (dir);
            }
            auto load_class_en = std::chrono::high_resolution_clock::now();
            double t_load_class = std::chrono::duration<double>(
            load_class_en-load_class_sa).count();
            std::cout << "Load class time (s) = " << t_load_class << std::endl;

         }
      }
   }
}

   
template class ImageNet2012<float>;

}} // End of namespace magmadnn::data
