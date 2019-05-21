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
    Operation<T> *tmp;

    /* prune compute graph:
        construct a new graph G' that only contains nodes that are ancestors of graph and 
        descendents of nodes in vars. */
    /* TODO */

    /* init Loss in grad table to one */
    Operation<T> *ones = op::var<T>("__grad_loss", graph->get_output_shape(), {IDENTITY, {}}, graph->get_memory_type());
    table.set(graph, ones);

    /* compute the gradients for each variable */
    for (typename std::vector<Operation<T> *>::const_iterator vit = vars.begin(); vit != vars.end(); vit++) {
        if (*vit != NULL) {
            internal::debugf("calling build_grad on %s.\n", (*vit)->to_string().c_str());
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
magmadnn_error_t build_grad(op::Operation<T> *var, op::Operation<T> *graph, op::GradTable<T> &table, op::Operation<T> **grad) {
    op::Operation<T> *tmp_grad, *result, *bprop, *consumer;
    std::vector<op::Operation<T> *> bprops;
    magmadnn_error_t err;
    std::vector<op::Operation<T> *> consumers;

    /* error on null values */
    if (var == NULL || graph == NULL || grad == NULL) return (magmadnn_error_t) 1;

    /* get this entry in the grad table */
    tmp_grad = table.get(var);

    /* if not null then we have already calculated this gradient */
    if (tmp_grad != NULL) {
        internal::debugf("grad for %s already present [%s].\n", var->to_string().c_str(), tmp_grad->to_string().c_str());
        *grad = tmp_grad;
        return (magmadnn_error_t) 0;
    }

    internal::debugf("%s needs grad (%lu consumer(s))\n", var->to_string().c_str(), var->get_consumers().size());

    /* build gradients for each consumer to this operation in order to properly calculate ours */
    consumers = var->get_consumers();
    for (typename std::vector<op::Operation<T> *>::iterator vit = consumers.begin(); vit != consumers.end(); vit++) {

        consumer = (*vit);
        
        if (consumer == NULL) continue;

        /* build the gradient for consumer and keep track of it in bprops */
        internal::debugf("calling build_grad on %s.\n", consumer->to_string().c_str());
        err = build_grad(consumer, graph, table, &tmp_grad);
        if (err != 0) return err;
        bprop = consumer->grad(consumer, var, tmp_grad);
        bprops.push_back(bprop);
    }

    /* sum of each partial gradient is the total gradient */
    /* TODO : no need to sum if just one */
    result = op::sum(bprops);
    table.set(var, result);
    *grad = result;

    return (magmadnn_error_t) 0;
}
template magmadnn_error_t build_grad(op::Operation<int>* var, op::Operation<int> *graph, op::GradTable<int> &table, op::Operation<int> **grad);
template magmadnn_error_t build_grad(op::Operation<float>* var, op::Operation<float> *graph, op::GradTable<float> &table, op::Operation<float> **grad);
template magmadnn_error_t build_grad(op::Operation<double>* var, op::Operation<double> *graph, op::GradTable<double> &table, op::Operation<double> **grad);


}   // namespace internal
}   // namespace magmadnn