#pragma once

#include "magmadnn.h"

#include "magmadnn.h"

namespace magmadnn {
namespace solver {

template <typename T>
class FMinSolver {
public:
      
   // Compute the training error i.e. loss value and accuracy
   void eval_error(
         magmadnn::model::NeuralNetwork<T>& model,
         Tensor<T>& x, // Input data
         Tensor<T>& y,  // Labels on input data
         int batch_size,
         T &loss, // Loss value
         T &accuracy
         ) {
         
      // This will store the result of the argmax on the output of the network
      Tensor<T> predicted({model.network_output_tensor()->get_shape(0)}, {ZERO,{}}, HOST);
      // This will store the result of the argmax on the ground_truth
      Tensor<T> actual({model.network_output_tensor()->get_shape(0)}, {ZERO, {}}, HOST);
      // Used to move network output onto CPU
      Tensor<T> host_network_output_tensor_ptr(model.network_output_tensor()->get_shape(), {NONE, {}}, HOST); 
      // Used to move ground_truth onto CPU
      Tensor<T> host_ground_truth_tensor_ptr(model.ground_truth_tensor()->get_shape(), {NONE,{}}, HOST);

      // Loss function
      op::Operation<T> *lossfun = model.lossfun(); 
      // Loss function tensor
      Tensor<T> *lossfun_tensor = lossfun->get_output_tensor();

      // Data loader
      dataloader::LinearLoader<T> dataloader(&x, &y, batch_size);
      unsigned int sample_size_x = x.get_size() / x.get_shape(0);
      unsigned int sample_size_y = y.get_size() / y.get_shape(0);
      unsigned int batch_mem_space_x = batch_size * sample_size_x;
      unsigned int batch_mem_space_y = batch_size * sample_size_y;

      // Number of correctly predicted samples
      unsigned int n_correct = 0;
      // Nimber of samples
      unsigned int n_samples = y.get_shape(0);
      // Init Loss
      loss = 0.0;

      // Number of batches
      auto nbatch = dataloader.get_num_batches();

      // Compute initial loss and accuracy
      for (int j = 0; j < nbatch; j++) {

         model.network_input_tensor()->copy_from(x, j * batch_mem_space_x, batch_mem_space_x);
         model.ground_truth_tensor()->copy_from(y, j * batch_mem_space_y, batch_mem_space_y);                   

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
      }

      loss /= static_cast<T>(nbatch);
      accuracy = static_cast<T>(n_correct) / static_cast<T>(n_samples);
   }

};

   
}} // End of magmadnn:solver namespace
