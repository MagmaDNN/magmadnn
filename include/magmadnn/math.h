#pragma once

namespace magmadnn {

   // mathematical functions

   /**
    * Performs integer division with rounding up.
    *
    * @param num  numerator
    * @param den  denominator
    *
    * @return returns the ceiled quotient.
    */
   inline constexpr int ceildiv(int num, int den)
   {
      return (num + den - 1) / den;
   }

}  // namespace magmadnn
