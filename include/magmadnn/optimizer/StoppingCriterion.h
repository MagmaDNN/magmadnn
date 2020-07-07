#pragma once

#include "magmadnn.h"

namespace magmadnn {
namespace solver {

class StoppingCriterion {

   StoppingCriterion()
      : elapsed_time_(0.0), num_iters_(0), 
   {}
   
   void num_iters(int in_num_iters) {
      this->num_iters_ = in_num_iters;
   }

   void elapsed_time(double in_elapsed_time) {
      this->elapsed_time_ = in_elapsed_time;
   }
   
private:
   double elapsed_time_;
   int num_iters_;
};
   
}} // End of magmadnn:solver namespace
