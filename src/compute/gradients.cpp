/**
 * @file gradients.cpp
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-05-17
 * 
 * @copyright Copyright (c) 2019
 */
#include "compute/gradients.h"

namespace magmadnn {
namespace op {


template <typename T>
magmadnn_error_t get_grad_table(const std::vector<Operation<T> *>& vars, Operation<T> *graph, GradTable<T> &table) {
    magmadnn_error_t err;
    Tensor<T> *tmp;

    /* prune compute graph:
        construct a new graph G' that only contains nodes that are ancestors of graph and 
        descendents of nodes in vars. */
    /* TODO */

    /* init Loss in grad table to one */
    Tensor<T> *grad_loss = new Tensor<T> ({1}, {ONE, {}}, graph->get_memory_type());
    table.set(graph, grad_loss);

    /* compute the gradients for each variable */
    for (typename std::vector<Operation<T> *>::const_iterator vit = vars.begin(); vit != vars.end(); vit++) {
        if (*vit != NULL) {
            err = internal::build_grad(*vit, graph, table, &tmp);
        } else {
            return (magmadnn_error_t) 1;
        }

        if (err != 0) { return err; }
    }

    return (magmadnn_error_t) 0;
}
template magmadnn_error_t get_grad_table(const std::vector<Operation<int> *>& vars, Operation<int> *graph, GradTable<int> &table);
template magmadnn_error_t get_grad_table(const std::vector<Operation<float> *>& vars, Operation<float> *graph, GradTable<float> &table);
template magmadnn_error_t get_grad_table(const std::vector<Operation<double> *>& vars, Operation<double> *graph, GradTable<double> &table);

}   // namespace op

// build_grad should only be used internally
namespace internal {

template <typename T>
magmadnn_error_t build_grad(op::Operation<T> *var, op::Operation<T> *graph, op::GradTable<T> &table, Tensor<T> **grad) {
    Tensor<T> *tmp_grad, *bprop, *result;
    op::Operation<T> *consumer;
    std::vector<Tensor<T> *> bprops;
    magmadnn_error_t err;
    std::vector<op::Operation<T> *> consumers;

    /* error on null values */
    if (var == NULL || graph == NULL || grad == NULL) return (magmadnn_error_t) 1;

    /* get this entry in the grad table */
    tmp_grad = table.get(var);

    /* if not null then we have already calculated this gradient */
    if (tmp_grad != NULL) {
        *grad = tmp_grad;
        return (magmadnn_error_t) 0;
    }


    /* build gradients for each consumer to this operation in order to properly calculate ours */
    consumers = var->get_consumers();
    for (typename std::vector<op::Operation<T> *>::iterator vit = consumers.begin(); vit != consumers.end(); vit++) {

        consumer = (*vit);
        
        /* if this is null for some reason stop here */
        if (consumer == NULL) continue;

        /* build the gradient for consumer and keep track of it in bprops */
        err = build_grad(consumer, graph, table, &tmp_grad);
        if (err != 0) return err;

        bprop = consumer->grad(consumer, var, tmp_grad);
        bprops.push_back(bprop);

    }

    /* sum of each partial gradient is the total gradient */
    /* TODO : no need to sum if just one */
    if (bprops.size() == 0) {
        return (magmadnn_error_t) 2;
    } else if (bprops.size() == 1) {
        result = bprops.at(0);
    } else if (bprops.size() == 2) {
        /*
        result = op::add(bprops.at(0), bprops.at(1), true, false);
        */
        /* TODO : Add and sum tensors */
        result = NULL;
        fprintf(stderr, "Implement add in gradients\n");
    } else {
        /* currently sum cannot handle scalar values, so just tetrate adds for now */
        //result = op::sum(bprops);
        /* 
        result = bprops.at(0);
        for (unsigned int i = 1; i < bprops.size(); i++) {
            result = op::add(result, bprops.at(i));
        }*/
        result = NULL;
        fprintf(stderr, "Implement sum in gradients\n");
    }
    
    table.set(var, result);
    *grad = result;

    return (magmadnn_error_t) 0;
}
template magmadnn_error_t build_grad(op::Operation<int>* var, op::Operation<int> *graph, op::GradTable<int> &table, Tensor<int> **grad);
template magmadnn_error_t build_grad(op::Operation<float>* var, op::Operation<float> *graph, op::GradTable<float> &table, Tensor<float> **grad);
template magmadnn_error_t build_grad(op::Operation<double>* var, op::Operation<double> *graph, op::GradTable<double> &table, Tensor<double> **grad);


}   // namespace internal
}   // namespace magmadnn