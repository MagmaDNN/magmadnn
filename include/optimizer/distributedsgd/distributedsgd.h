/**
 * @file distributedsgd.h
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-08-15
 *
 * @copyright Copyright (c) 2019
 */
#pragma once

/* only include this class if MPI is available */
#if defined(MPI_VERSION) && defined(MPI_SUBVERSION)
#if !defined(_HAS_MPI_)
#define _HAS_MPI_
#endif

#if defined(MAGMADNN_HAVE_CUDA)
#include <cuda.h>
#endif

#endif

#include <map>
#include "compute/gradients.h"
#include "compute/gradtable.h"
#include "compute/operation.h"
#include "optimizer/optimizer.h"

#include "math/add.h"
#include "math/scalar_tensor_product.h"

namespace magmadnn {
namespace optimizer {

template <typename T>
class DistributedGradientDescent : public Optimizer<T> {
   public:
    DistributedGradientDescent(T learning_rate) : learning_rate(learning_rate) {
        this->_name = "DistributedGradientDescentOptimizer";

        nnodes = 1;

#if defined(_HAS_MPI_)
        MPI_Comm_size(MPI_COMM_WORLD, &nnodes);
        assert(nnodes >= 1);
        MPI_Comm_rank(MPI_COMM_WORLD, &this->rank);

#if defined(MAGMADNN_HAVE_CUDA)
        int num_devices;

        // query number of devices
        cudaError_t err;
        err = cudaGetDeviceCount(&num_devices);
        assert(err == 0 || err == cudaErrorNoDevice);
        // cudaErrchk(cudaSetDevice(this->rank%num_devices));
#endif
#endif
    }

    ~DistributedGradientDescent() {}

    virtual void minimize(op::Operation<T> *obj_func, const std::vector<op::Operation<T> *> &wrt) {
        typename std::vector<op::Operation<T> *>::const_iterator vit;

        this->_obj_func = obj_func;
        this->_obj_func->eval(false);

        this->table.clear();
        op::get_grad_table(wrt, this->_obj_func, this->table);

        for (vit = wrt.begin(); vit != wrt.end(); vit++) {
            Tensor<T> *grad = this->table.get(*vit);

#if defined(_HAS_MPI_)
            MPI_Allreduce(MPI_IN_PLACE, grad->get_ptr(), grad->get_size(), MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            math::scalar_tensor_product(static_cast<T>(1) / static_cast<T>(nnodes), grad, grad);
#endif

            this->update((*vit), grad);
        }
    }

    void set_learning_rate(T learning_rate) { this->learning_rate = learning_rate; }
    T get_learning_rate() { return this->learning_rate; }

   protected:
    virtual void update(op::Operation<T> *var, Tensor<T> *grad) {
        Tensor<T> *var_tensor = var->eval(false);

        math::add_in_place(-this->learning_rate, grad, static_cast<T>(1), var_tensor);
    }

#if defined(_HAS_MPI_)
    MPI_Op avg;
    int rank;
#endif
    T learning_rate;
    int nnodes;
    op::GradTable<T> table;
};

}  // namespace optimizer
}  // namespace magmadnn
