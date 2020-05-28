#pragma once

#include "magmadnn.h"

namespace magmadnn {
namespace solver {

class StoppingCriterion {

   void num_iters(int in_num_iters) {
      this->num_iters_ = in_num_iters;
   }

   void elapsed_time(double in_elapsed_time) {
      this->elapsed_time_ = in_elapsed_time;
   }
   
private:
   int num_iters_;
   double elapsed_time_;
};
   
}} // End of magmadnn:solver namespace
