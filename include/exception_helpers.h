#pragma once

namespace magmadnn {

/**
 * Marks a function as not yet implemented.
 *
 * Attempts to call this function will result in a runtime error of type
 * NotImplemented.
 */
#define MAGMADNN_NOT_IMPLEMENTED                                        \
   {                                                                    \
      throw ::magmadnn::NotImplemented(__FILE__, __LINE__, __func__);   \
   }                                                                    \
   static_assert(true,                                                  \
                 "This assert is used to counter the false positive extra " \
                 "semi-colon warnings")
   
} // magmadnn namespace
