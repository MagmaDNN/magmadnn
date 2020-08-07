#pragma once

#if defined(MAGMADNN_CMAKE_BUILD)
#include "magmadnn/config.h"
#endif

#ifdef MAGMADNN_HAVE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // MAGMADNN_HAVE_OPENCV

namespace magmadnn {
namespace data {

#if defined(MAGMADNN_HAVE_OPENCV)
cv::Mat cv_read_and_resize_img(
      const std::string& filename,
      const int height, const int width, const bool is_color);
#endif
   
bool get_jpeg_size(const uint8_t* data, uint32_t data_size, int64_t *width, int64_t *height);
 
}} // End of namespace magmadnn::data
