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

namespace magmadnn {
namespace internal {

/** Only prints #if DEBUG macro is defined.
 * @param fmt 
 * @param ... 
 */
void debugf(const char *fmt, ...);

}   // namespace internal
}   // namespace magmadnn