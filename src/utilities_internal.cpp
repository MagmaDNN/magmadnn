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

void print_vector(const std::vector<unsigned int>& vec, bool debug, char begin, char end, char delim) {
    int (*print)(const char*,...) = (debug) ? debugf : std::printf;

    print("%c ", begin);
    for (std::vector<unsigned int>::const_iterator it = vec.begin(); it != vec.end(); it++) {
        print("%lu", *it);
        if (it != vec.end()-1) print("%c ", delim);
    }
    print(" %c\n", end);
}

}   // namespace internal
}   // namespace magmadnn