#pragma once

#include "magmadnn/optimizer/FMinSolver.h"
#include "magmadnn/optimizer/TrainStats.h"

#include <random>

#include <mpi.h>

namespace magmadnn {
namespace solver {

template <typename T>
class DistMomentumSGD : public magmadnn::solver::FMinSolver<T> {
public:
   // Momentum SGD itereration (synchronous)
   using SgdIter = MomentumSgdIter<T, false>;

   DistMomentumSGD()
      : sgd_iter(), rank_(-1), nnodes_(-1)
   {
      MPI_Comm_size(MPI_COMM_WORLD, &this->nnodes_);
      MPI_Comm_rank(MPI_COMM_WORLD, &this->rank_);
   }

   DistMomentumSGD(T learning_rate, T momentum)
      : sgd_iter(learning_rate, momentum), rank_(-1), nnodes_(-1)
   {
      MPI_Comm_size(MPI_COMM_WORLD, &this->nnodes_);
      MPI_Comm_rank(MPI_COMM_WORLD, &this->rank_);
   }

   void grad_reduce(
         std::vector<op::Operation<T> *> const& weights,
         op::GradTable<T> &grad_table) {
      
      for (auto w = weights.begin(); w != weights.end(); ++w) {

         op::Operation<T> *var = *w;

         Tensor<T> *grad = grad_table.get(var);

         // TODO use generic MPI types
         MPI_Allreduce(
               MPI_IN_PLACE, grad->get_ptr(), grad->get_size(), MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

      }
   }

   // Copy model parameters to every processes local model
   void model_bcast(
         magmadnn::model::NeuralNetwork<T>& model
         ) {

      std::vector<layer::Layer<T> *> layers = model.get_layers();

      for (layer::Layer<T> *layer: layers) {

         std::vector<::magmadnn::op::Operation<T> *> weights = layer->get_weights();

         for (magmadnn::op::Operation<T> *weight: weights) {

            Tensor<T> *output_tensor = weight->get_output_tensor();

            MPI_Bcast(
                  output_tensor->get_ptr(), output_tensor->get_size(), MPI_FLOAT, 0, MPI_COMM_WORLD);

         }
      }      
   }
   
   void min(
         // std::vector<magmadnn::model::NeuralNetwork<T>>& models,
         magmadnn::model::NeuralNetwork<T>& model,
         Tensor<T>& x, // Input data
         Tensor<T>& y,  // Labels on input data
         int batch_size,
         bool enable_train_error,
         int num_iters, double time_budget,
         std::vector<TrainStats<T>> &stats
         ) {

      std::string context = "DistMomentumSGD::min";

      std::cout << "Distributed memory algo" << std::endl;

      std::cout << "[" << context << "] " << "Batch size = " << batch_size << std::endl;
      std::cout << "[" << context << "] " << "Learning rate = " << sgd_iter.learning_rate() << std::endl;
      std::cout << "[" << context << "] " << "Momentum = " << sgd_iter.momentum() << std::endl;

      std::cout << "[" << context << "] " << "Number of iterations = " << num_iters << std::endl;
      std::cout << "[" << context << "] " << "Time budget = " << time_budget << std::endl;

      std::cout << "[" << context << "] " << "Number of processes = " << this->nnodes_ << std::endl;
      std::cout << "[" << context << "] " << "My rank = " << this->rank_ << std::endl;

      magmadnn::memory_t memory_type = model.memory_type();

      std::vector<op::Operation<T> *> &weights = model.weights();       
      op::Operation<T> *lossfun = model.lossfun(); 

      // Initialize data loader with training set (samples and labels)
      dataloader::LinearLoader<T> dataloader(&x, &y, batch_size);
      unsigned int sample_size_x = x.get_size() / x.get_shape(0);
      unsigned int sample_size_y = y.get_size() / y.get_shape(0);
      unsigned int batch_mem_space_x = batch_size * sample_size_x;
      unsigned int batch_mem_space_y = batch_size * sample_size_y;

      // Number of batches
      auto num_batches = dataloader.get_num_batches();

      auto seed = this->rank_;
      std::default_random_engine generator(seed);
      std::uniform_int_distribution<> distribution(0, num_batches-1);

      // GPU devid
      int devid = -1;
      
      if (memory_type == HOST) {
         std::cout << "[" << context << "] " << "CPU training" << std::endl;
      }
      else {
         int num_devices = -1;
         cudaError_t err;
         err = cudaGetDeviceCount( &num_devices );

         err = cudaGetDevice( &devid );
         
         std::cout << "[" << context << "] " << "GPU training (" << devid << ")" << std::endl;
         std::cout << "[" << context << "] " << "Total number of devices = " << num_devices << std::endl;
      }               
      std::cout << "[" << context << "] " << "Number of batches = " << num_batches << std::endl;

      // Calculate initial training accuracy

      // // Loss value
      // T loss = 0.0;
      // T accuracy = 0.0;
      // // Compute training error
      // this->eval_error(model, x, y, batch_size, loss, accuracy);

#if defined(MAGMADNN_HARNESS_HAVE_CUDA)      
      // cudaStream_t stream;
      // cudnnHandle_t cudnn_handle;
      // cublasHandle_t cublas_handle;

      // // CUDA stream
      // cudaStreamCreate(&stream);
      // // cuDNN
      // cudnnCreate(&cudnn_handle);
      // cudnnSetStream(cudnn_handle, stream);
      // // cuBLAS
      // cublasCreate(&cublas_handle);
      // cublasSetStream(cublas_handle, stream);

      magmadnn::CudaExecContext cuda_exec_ctx(devid);
      
      lossfun->set_async(true); // Asynchronous CUDA kernel submission
      lossfun->cuda_exec_context(cuda_exec_ctx);

      model.network_input_tensor()->cuda_exec_context(cuda_exec_ctx);
      model.ground_truth_tensor()->cuda_exec_context(cuda_exec_ctx);
      
      x.cuda_exec_context(cuda_exec_ctx);
      y.cuda_exec_context(cuda_exec_ctx);
#endif

      auto start = std::chrono::high_resolution_clock::now();
      auto now = std::chrono::high_resolution_clock::now();
      double elapsed_time = 0.0;

      // global gradient
      op::GradTable<T> grad_table;
      grad_table.clear();
      op::get_grad_table(weights, lossfun, grad_table);

      grad_table.zero();
#if defined(MAGMADNN_HARNESS_HAVE_CUDA)
      cuda_exec_ctx.synchronize();
#endif

      // Reset momentum
      sgd_iter.reset();

      int iters = 0;
      
      while (iters < num_iters) {

         //
         // Select a batch index randomly
         //
         int batch_idx = distribution(generator);

         // Load local model with randomly selected batch
         model.network_input_tensor()->copy_from(x, batch_idx * batch_mem_space_x, batch_mem_space_x);
         model.ground_truth_tensor()->copy_from(y, batch_idx * batch_mem_space_y, batch_mem_space_y);            

         //
         // Forward pass
         //
         lossfun->eval(true); // forces evaluation
#if defined(MAGMADNN_HARNESS_HAVE_CUDA)
         cuda_exec_ctx.synchronize();
#endif

         //
         // Compute local gradient using local model which is a copy
         // of the "global" one on rank 0
         //
         grad_table.clear();
         magmadnn::op::get_grad_table(weights, lossfun, grad_table);
#if defined(MAGMADNN_HARNESS_HAVE_CUDA)
         cuda_exec_ctx.synchronize();
#endif

         // Reduce gradient computed on the different processes
         this->grad_reduce(weights, grad_table);

         // Perform SGD step
         if (this->rank_ == 0) {
            
            auto nnodes = this->nnodes_;
            // Scaling factor for gradient step
            // T scale = 1.0;
            T scale = 1.0 / static_cast<T>(nnodes);

#if defined(MAGMADNN_HARNESS_HAVE_CUDA)
            // Synchronous step, no need for barrier
            sgd_iter.step(cuda_exec_ctx.stream(), weights, grad_table, scale);
            cuda_exec_ctx.synchronize();
#else
            sgd_iter.step(weights, grad_table, scale);
#endif                     
         }

         // Send model to participating nodes
         this->model_bcast(model);
            
         ++iters;
      }

#if defined(MAGMADNN_HARNESS_HAVE_CUDA)
      // Reset custream and cudnn
      lossfun->set_custream(nullptr);
      lossfun->set_cudnn_handle(::magmadnn::internal::MAGMADNN_SETTINGS->cudnn_handle);
      lossfun->set_cublas_handle(::magmadnn::internal::MAGMADNN_SETTINGS->cublas_handle);
#endif

   }

private:
   int nnodes_;
   int rank_;
   SgdIter sgd_iter;
};
   
}} // End of magmadnn:solver namespace
