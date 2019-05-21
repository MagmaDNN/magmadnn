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
        construct a new graph G' that only contains nodes that are ancestors of z and 
        descendents of nodes in vars. */
    /* TODO */

    /* init Loss in grad table to one */
    Operation<T> *ones = op::var<T>("__grad_loss", {1}, {ONE, {}}, graph->get_memory_type());
    table.set(graph, ones);

    /* compute the gradients for each variable */
    for (typename std::vector<Operation<T> *>::iterator vit = vars.begin(); vit != vars.end(); vit++) {
        err = internal::build_grad(*vit, graph, table, &tmp);

        if (err != 0) { delete tmp; return err; }
    }

    delete tmp;
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
    magmadnn_error_t err;

    /* get this entry in the grad table */
    tmp_grad = table.get(var);

    /* if not null then we have already calculated this gradient */
    if (tmp_grad != NULL) {
        *grad = tmp_grad;
        return (magmadnn_error_t) 0;
    }

    for (typename std::vector<Operation<T> *>::iterator vit = var->get_consumers().begin(); 
        vit != var->get_consumers().end(); vit++) {

        err = build_grad(*vit, graph, table, &tmp_grad);
        if (err != 0) return err;

        bprops.push_back((*vit)->grad(*vit, var, tmp_grad));
    }

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