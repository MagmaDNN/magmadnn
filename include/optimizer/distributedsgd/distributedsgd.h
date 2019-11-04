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
#define _HAS_MPI_

#if defined(_HAS_CUDA_)
#include <cuda.h>
#endif

#endif

#include <map>
#include "compute/gradients.h"
#include "compute/gradtable.h"
#include "compute/operation.h"
#include "optimizer/optimizer.h"

namespace magmadnn {
namespace optimizer {

template <typename T, int N>
class DistributedGradientDescent : public Optimizer<T> {
   public:
    GradientDescent(T learning_rate) {
        this->_name = "DistributedGradientDescentOptimizer";

        static_assert(N >= 1);

#if defined(_HAS_MPI_)
        int n_nodes;
        MPI_Comm_size(MPI_COMM_WORLD, &n_nodes);
        assert(n_nodes == N);
        MPI_Comm_rank(MPI_COMM_WORLD, &this->rank);

#if defined(_HAS_CUDA_)
        cudaErrchk(cudaSetDevice(this->rank));
#endif

        MPI_Op_create([](void *a, void *b, int *len,
                         MPI_Datatype *dtype) { (*(T *) b) = ((*(T *) a) + (*(T *) b)) / ((float) N); },
                      true, &this->avg);
#endif
    }

    ~DistributedGradientDescent() {
#if defined(_HAS_MPI_)
        MPI_Op_free(&avg);
#endif
    }

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
            math::scalar_tensor_product(static_cast<T>(1) / static_cast<T>(N), grad, grad);
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
    op::GradTable<T> table;
};

}  // namespace optimizer
}  // namespace magmadnn

#if defined(_HAS_MPI_)
#undef _HAS_MPI_
#endif