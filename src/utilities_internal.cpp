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

int debugf(const char *fmt, ...) {
    #if defined(DEBUG)
    int bytes;
    va_list args;
    va_start(args, fmt);
    bytes = vfprintf(stdout, fmt, args);
    va_end(args);
    return bytes;
    #endif
    return 0;
}
}

}   // namespace internal
}   // namespace magmadnn