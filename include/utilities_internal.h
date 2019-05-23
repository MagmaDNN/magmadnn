/**
 * @file utilities_internal.h
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-05-21
 * 
 * @copyright Copyright (c) 2019
 */
#pragma once

#include <cstdio>
#include <cstdarg>
#include <vector>

namespace magmadnn {
namespace internal {

/** Only prints #if DEBUG macro is defined.
 * @param fmt 
 * @param ... 
 */
int debugf(const char *fmt, ...);


void print_vector(const std::vector<unsigned int>& vec, bool debug=true, char begin='{', char end='}', char delim=',');

}   // namespace internal
}   // namespace magmadnn