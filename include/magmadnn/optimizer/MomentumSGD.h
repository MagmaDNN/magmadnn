#pragma once

#include "magmadnn.h"

namespace magmadnn {
namespace solver {

template <
   typename T,
   bool async=false>
class MomentumSgdIter {
public:
   using MomentumTable = std::map<op::Operation<T> *, Tensor<T> *>;
public:
   
   MomentumSgdIter()
      : learning_rate_(static_cast<T>(1.0)), momentum_(0.9)
   {}

   MomentumSgdIter(T learning_rate, T momentum)
      : learning_rate_(learning_rate), momentum_(momentum)
   {}

   ~MomentumSgdIter() {
      momentum_table_.clear();
   }

   // Return learning rate
   T learning_rate() const { return this->learning_rate_; }

   T momentum() const { return this->momentum_; }
      
   void reset() {
      this->momentum_table_.clear();      
   }
   
   void step(
         std::vector<op::Operation<T> *> const& weights,
         op::GradTable<T> &grad_table,
         T scale = static_cast<T>(1.0)) {

      for (auto w = weights.begin(); w != weights.end(); ++w) {

         op::Operation<T> *var = *w;

         Tensor<T> *var_tensor = var->eval(false);
         Tensor<T> *grad = grad_table.get(var);
            
         if (!momentum_table_.count(var)) {
            // Init momentum to zero
            momentum_table_[var] = new Tensor<T>(grad->get_shape(), {ZERO, {}}, grad->get_memory_type());
         }

         Tensor<T> *prev_grad = momentum_table_[var];
            
         magmadnn::math::sgd_momentum(learning_rate_ * scale, momentum_, prev_grad, grad, var_tensor);
      }         
   }
      
#if defined(MAGMADNN_HAVE_CUDA)
   void step(
         cudaStream_t custream,
         std::vector<op::Operation<T> *> const& weights,
         op::GradTable<T> &grad_table,
         T scale = static_cast<T>(1.0)
         ) {

      for (auto w = weights.begin(); w != weights.end(); ++w) {

         op::Operation<T> *var = *w;

         Tensor<T> *var_tensor = var->eval(false);
         Tensor<T> *grad = grad_table.get(var);
            
         if (!momentum_table_.count(var)) {
            // Init momentum to zero
            momentum_table_[var] = new Tensor<T>(grad->get_shape(), {ZERO, {}}, grad->get_memory_type());
         }

         Tensor<T> *prev_grad = momentum_table_[var];
            
         // magmadnn::math::sgd_momentum(learning_rate_ * scale, momentum_, prev_grad, grad, var_tensor);

         if (var_tensor->get_memory_type() == HOST) {
            magmadnn::math::sgd_momentum_cpu(
                  learning_rate_ * scale, momentum_, prev_grad, grad,
                  var_tensor);

            // Check for update sparsity
            // T *grad_ptr = grad->get_ptr();
            // auto size = grad->get_size();
            // uint64_t nnz = 0;
            // for (uint64_t i = 0; i < size; ++i) {
            //    if (grad_ptr[i] > 1e-7)
            //       nnz++;
            // }
            
            // double density = static_cast<double>(nnz) / static_cast<double>(size);
            // std::cout << "density = " << density << std::endl;

         }
         else {
            magmadnn::math::sgd_momentum_device(
                  custream, learning_rate_ * scale, momentum_, prev_grad,
                  grad, var_tensor);
            if (!async) cudaStreamSynchronize(custream);               

            // Check for update sparsity
            // Tensor<T> tmp(grad->get_shape(), {NONE, {}}, HOST);
            // cudaStreamSynchronize(custream);
            // tmp.copy_from(*grad);
            // cudaStreamSynchronize(custream);

            // auto size = tmp.get_size();
            // uint64_t nnz = 0;
            // for (uint64_t i = 0; i < size; ++i) {
            //    if (tmp.get_ptr()[i] > 1e-7)
            //       nnz++;
            // }

            // double density = static_cast<double>(nnz) / static_cast<double>(size);
            // std::cout << "density = " << density << std::endl;

         }
      }      
   }
#endif

private:
   T learning_rate_;
   T momentum_;
   MomentumTable momentum_table_;
   
};
   
}} // End of magmadnn:solver namespace
