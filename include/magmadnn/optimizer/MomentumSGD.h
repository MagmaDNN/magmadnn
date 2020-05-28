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

template <typename T>
class MomentumSGD {
public:
   // Momentum SGD itereration (synchronous)
   using SgdIter = MomentumSgdIter<T, false>;

   MomentumSGD()
      : sgd_iter()
   {}

   MomentumSGD(T learning_rate, T momentum)
      : sgd_iter(learning_rate, momentum)
   {}

   // Determines a local minimum of the loss function for the given
   // neural network model
   void min(
         // op::Operation<T> lossfun, // Loss fucntion
         magmadnn::model::NeuralNetwork<T>& model,
         Tensor<T>& x, // Input data
         Tensor<T>& y,  // Labels on input data
         int batch_size,
         int nepoch
         ) {

      std::cout << "Serial algo" << std::endl;

      std::cout << "Batch size = " << batch_size << std::endl;
      std::cout << "Learning rate = " << sgd_iter.learning_rate() << std::endl;
      std::cout << "momentum = " << sgd_iter.momentum() << std::endl;
      std::cout << "Number of epochs = " << nepoch << std::endl;
         
      unsigned int n_samples = y.get_shape(0);
      op::Operation<T> *lossfun = model.lossfun(); 
      std::vector<op::Operation<T> *>& weights = model.weights();       
      // init the host tensors

      Tensor<T> *lossfun_tensor = lossfun->get_output_tensor();
         
      // This will store the result of the argmax on the output of the network
      Tensor<T> predicted({model.network_output_tensor()->get_shape(0)}, {ZERO,{}}, HOST);
      // This will store the result of the argmax on the ground_truth
      Tensor<T> actual({model.network_output_tensor()->get_shape(0)}, {ZERO, {}}, HOST);
      // Used to move network output onto CPU
      Tensor<T> host_network_output_tensor_ptr(model.network_output_tensor()->get_shape(), {NONE, {}}, HOST); 
      // Used to move ground_truth onto CPU
      Tensor<T> host_ground_truth_tensor_ptr(model.ground_truth_tensor()->get_shape(), {NONE,{}}, HOST);

      dataloader::LinearLoader<T> dataloader(&x, &y, batch_size);
      unsigned int sample_size_x = x.get_size() / x.get_shape(0);
      unsigned int sample_size_y = y.get_size() / y.get_shape(0);
      unsigned int batch_mem_space_x = batch_size * sample_size_x;
      unsigned int batch_mem_space_y = batch_size * sample_size_y;
         
      std::cout << "Number of batches = " << dataloader.get_num_batches() << std::endl;

      std::cout << "x, shape = " << x.get_shape(0) << ", " << x.get_shape(1) << ", " << x.get_shape(2) << std::endl;

      // Gradient table
      // op::GradTable<T> table;
      // table.clear(); // Init table
      // Cummulative loss value
      T cumulative_loss = 0.0;

      op::GradTable<T> grad_table;
      std::map<op::Operation<T> *, Tensor<T> *> momentum_table;

      // CUDA stream
      cudaStream_t custream;
         
      for (int e = 0; e < nepoch; ++e) {

         // Loss value for current epoch
         T loss = 0.0;
         // Gradient 2-norm
         // T normgrad = 0.0;
         // Number of correctly predicted values
         unsigned int n_correct = 0;
            
         for (int j = 0; j < dataloader.get_num_batches(); j++) {

            // load next batch into x and y
            // dataloader.next(
            //       model.get_network_input_tensor_ptr(),
            //       model.get_ground_truth_tensor_ptr());

            model.network_input_tensor()->copy_from(x, j * batch_mem_space_x, batch_mem_space_x);
            model.ground_truth_tensor()->copy_from(y, j * batch_mem_space_y, batch_mem_space_y);                   
               
            // forward pass
            lossfun->eval(true); // forces evaluation

            // model.get_optim()->minimize(lossfun, weights);

            grad_table.clear(); // Init table
            op::get_grad_table(weights, lossfun, grad_table); // Compute gradients

#if defined(MAGMADNN_HAVE_CUDA)
            // Synchronous step, no need for barrier
            sgd_iter.step(custream, weights, grad_table, 1.0);
#else
            sgd_iter.step(weights, grad_table, scale);
#endif                     

            // for (auto w = weights.begin(); w != weights.end(); ++w) {
                  
            //    op::Operation<T> *var = *w;
            //    Tensor<T> *var_tensor = var->eval(false);
            //    Tensor<T> *grad = grad_table.get(var); // Computed gradient
                  
            //    if (!momentum_table.count(var)) {
            //       // Init momentum to zero
            //       momentum_table[var] = new Tensor<T>(grad->get_shape(), {ZERO, {}}, grad->get_memory_type());
            //    }
            //    Tensor<T> *prev_grad_tensor = momentum_table[var];
     
            //    T *var_ptr = var_tensor->get_ptr();
            //    T *grad_ptr = grad->get_ptr();
            //    T *prev_grad_ptr = prev_grad_tensor->get_ptr(); 
                  
            //    for (unsigned int i = 0; i < grad->get_size(); i++) {
            //       // grad_ptr[i] /= static_cast<T>(batch_size);
            //       // normgrad += grad_ptr[i]*grad_ptr[i];
            //       prev_grad_ptr[i] = momentum_ * prev_grad_ptr[i] + (1 - momentum_) * grad_ptr[i];
            //       var_ptr[i] = var_ptr[i] - learning_rate_ * prev_grad_ptr[i];
            //       // var_ptr[i] = 0.0;
            //    }
            // }

            lossfun->eval(true); // forces evaluation

            /* get the argmax of the networks output (on CPU) */
            host_network_output_tensor_ptr.copy_from(*model.network_output_tensor());
            math::argmax(&host_network_output_tensor_ptr, 0, &predicted);

            /* get the argmax of the ground truth (on CPU) */
            host_ground_truth_tensor_ptr.copy_from(*model.ground_truth_tensor());
            math::argmax(&host_ground_truth_tensor_ptr, 0, &actual);

            /* update the accuracy and loss */
            for (unsigned int j = 0; j < batch_size; j++) {
               if (std::fabs(predicted.get(j) - actual.get(j)) <= 1E-8) {
                  n_correct++;
               }
            }

            lossfun_tensor->get_memory_manager()->sync();         
            loss += lossfun_tensor->get(0);

            // printf("Batch %d Loss = %f\n", j, lossfun_tensor->get(0));

         }

         loss /= dataloader.get_num_batches(); 
         // cumulative_loss += loss;
         // T avg_loss = cumulative_loss / (i + 1);

         printf("Epoch = %d\n", e+1);
         printf("Loss = %f\n", loss);
         printf("Accuracy = %f\n", static_cast<T>(n_correct) / static_cast<T>(n_samples));

         // normgrad = sqrt(normgrad);
         // printf("Norm grad = %e\n", normgrad);
            
         // Reset gradient memory before iterating to the next epoch
         sgd_iter.reset();
         
         dataloader.reset();
      }
   }
   
private:
   SgdIter sgd_iter;
};
   
}} // End of magmadnn:solver namespace
