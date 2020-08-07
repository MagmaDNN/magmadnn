#include "magmadnn/data/image_io.h"
#include <stdexcept>
#include <string>

namespace magmadnn {
namespace data {

#if defined(MAGMADNN_HAVE_OPENCV)
cv::Mat cv_read_and_resize_img(
      const std::string& filename,
      const int height, const int width, const bool is_color) {
   cv::Mat cv_img;
   int cv_read_flag = (is_color ? cv::IMREAD_COLOR :
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
