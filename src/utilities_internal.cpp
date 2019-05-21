/**
 * @file utilities_internal.cpp
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-05-21
 * 
 * @copyright Copyright (c) 2019
 */
#include "utilities_internal.h"

namespace magmadnn {
namespace internal {

void debugf(const char *fmt, ...) {
    #if defined(DEBUG)
    va_list args;
    va_start(args, fmt);
    vfprintf(stdout, fmt, args);
    va_end(args);
    #endif
}

}   // namespace internal
}   // namespace magmadnn