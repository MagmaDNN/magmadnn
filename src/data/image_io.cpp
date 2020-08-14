#include "magmadnn/data/image_io.h"
#include "tensor/tensor.h"

#include <stdexcept>
#include <string>

namespace magmadnn {
namespace data {

#if defined(MAGMADNN_HAVE_OPENCV)
cv::Mat cv_read_image(
      const std::string& filename,
      const int height, const int width, const bool is_color) {
   cv::Mat cv_img;
   // IMREAD_COLOR: Always convert image to the 3 channel BGR color image.
   // IMREAD_GRAYSCALE: Always convert image to the single channel
   // grayscale image.
   int cv_read_flag = (is_color ?
                       cv::IMREAD_COLOR :
                       cv::IMREAD_GRAYSCALE );
   cv::Mat cv_img_origin = cv::imread(filename, cv_read_flag);
   if (!cv_img_origin.data) {
      // throw std::runtime_error(
            // "Could NOT open image file: " + filename);
      return cv_img_origin;
   }
   if (height > 0 && width > 0) {
      cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
   } else {
      cv_img = cv_img_origin;
   }
   return cv_img;
}
#endif


// Read image whose path is set in `filename` and add the
// corresponding to the image pixels to the tensor `images_tensor` at
// `data_idx` index. The output image size can be specified with
// `height` and `width` parameters.
template<typename T>
void add_image_to_tensor(
      const std::string& filename,
      const int height, const int width, const bool is_color,
      magmadnn::Tensor<T>* images_tensor, unsigned int image_idx) {

#if defined(MAGMADNN_HAVE_OPENCV)

   // auto read_image_sa = std::chrono::high_resolution_clock::now();
   cv::Mat cv_img = cv_read_image(filename, height, width, is_color);
   // auto read_image_en = std::chrono::high_resolution_clock::now();
   // double t_read_image = std::chrono::duration<double>(
         // read_image_en-read_image_sa).count();
   // std::cout << "Read image time (s) = " << t_read_image << std::endl;
   
   if (cv_img.data) {
      auto nchannels = cv_img.channels();
      auto nrows = cv_img.rows;
      auto ncols = cv_img.cols;
      
      // auto set_tensor_sa = std::chrono::high_resolution_clock::now();
      for (unsigned int row = 0; row < nrows; ++row) {
         const uchar* ptr = cv_img.ptr<uchar>(row);
         int img_index = 0;
         for (unsigned int col = 0; col < ncols; ++col) {
            for (unsigned int ch = 0; ch < nchannels; ++ch) {
               unsigned int tensor_index = (col * nrows + row) * nchannels + ch;
               tensor_index = image_idx * nrows * ncols * nchannels + ch * ncols * ncols + row * ncols + col;     
               images_tensor->set(tensor_index, static_cast<T>(ptr[img_index]));
               // images_tensor->set({image_idx, ch, row, col}, static_cast<T>(ptr[img_index]));
               img_index++;
            }
         }
      }
      // auto set_tensor_en = std::chrono::high_resolution_clock::now();
      // double t_set_tensor = std::chrono::duration<double>(
         // set_tensor_en-set_tensor_sa).count();
      // std::cout << "Set tensor time (s) = " << t_set_tensor << std::endl;

   }

#else
   throw std::runtime_error("OpenCV must be enabled to read images");   
#endif

}

template void add_image_to_tensor<float>(
      const std::string& filename,
      const int height, const int width, const bool is_color,
      magmadnn::Tensor<float>* images_tensor, unsigned int image_idx);
   
// Read image whose path is set in `filename` and return a tensor
// corresponding to the image pixels. The output image size can be
// specified with `height` and `width` parameters.
template<typename T>
magmadnn::Tensor<T>* read_image(
      const std::string& filename,
      const int height, const int width, const bool is_color) {

   magmadnn::Tensor<T>* image_tensor = nullptr;
      
#if defined(MAGMADNN_HAVE_OPENCV)

   cv::Mat cv_img = cv_read_image(filename, height, width, is_color);

   if (cv_img.data) {
      auto nchannels = cv_img.channels();
      auto nrows = cv_img.rows;
      auto ncols = cv_img.cols;

      image_tensor = new magmadnn::Tensor<T>(
            {nchannels, nrows, ncols}, {magmadnn::NONE, {}}, magmadnn::HOST);

      for (int row = 0; row < nrows; ++row) {
         const uchar* ptr = cv_img.ptr<uchar>(row);
         int img_index = 0;
         for (int col = 0; col < ncols; ++col) {
            for (int ch = 0; ch < nchannels; ++ch) {
               // int tensor_index = (col * nrows + row) * nchannels + ch;
               // image_tensor.set(tensor_index, static_cast<T>(ptr[img_index]));
               image_tensor.set({ch, row, col}, static_cast<T>(ptr[img_index]));
               img_index++;
            }
         }
      }

   }
   
#else
   throw std::runtime_error("OpenCV must be enabled to read images");   
#endif

   return image_tensor;
}
   
// http://www.64lines.com/jpeg-width-height
// Gets the JPEG size from the array of data passed to the function,
// file reference: http://www.obrador.com/essentialjpeg/headerinfo.htm
bool get_jpeg_size(const uint8_t* data, uint32_t data_size, int64_t *width, int64_t *height) {
  // Check for valid JPEG image
  uint32_t i = 0;  // Keeps track of the position within the file
  if (data[i] == 0xFF && data[i+1] == 0xD8 && data[i+2] == 0xFF && data[i+3] == 0xE0) {
    i += 4;
    // Check for valid JPEG header (null terminated JFIF)
    if (data[i+2] == 'J' && data[i+3] == 'F' && data[i+4] == 'I'
        && data[i+5] == 'F' && data[i+6] == 0x00) {
      // Retrieve the block length of the first block since
      // the first block will not contain the size of file
      uint16_t block_length = data[i] * 256 + data[i+1];
      while (i < data_size) {
        i+=block_length;  // Increase the file index to get to the next block
        if (i >= data_size) return false;  // Check to protect against segmentation faults
        if (data[i] != 0xFF) return false;  // Check that we are truly at the start of another block
        uint8_t m = data[i+1];
        if (m == 0xC0 || (m >= 0xC1 && m <= 0xCF && m != 0xC4 && m != 0xC8 && m != 0xCC)) {
          // 0xFFC0 is the "Start of frame" marker which contains the file size
          // The structure of the 0xFFC0 block is quite simple
          // [0xFFC0][ushort length][uchar precision][ushort x][ushort y]
          *height = data[i+5]*256 + data[i+6];
          *width = data[i+7]*256 + data[i+8];
          return true;
        } else {
          i+=2;  // Skip the block marker
          block_length = data[i] * 256 + data[i+1];  // Go to the next block
        }
      }
      return false;  // If this point is reached then no size was found
    } else {
      return false;  // Not a valid JFIF string
    }
  } else {
    return false;  // Not a valid SOI header
  }
}
   
   
}} // End of namespace magmadnn::data
