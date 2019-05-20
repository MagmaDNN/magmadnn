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
magmadnn_error_t get_grad_table(std::vector<Variable<T> *>& vars, Operation<T> *graph, GradTable<T> &table) {
    magmadnn_error_t err;

    /* prune compute graph:
        construct a new graph G' that only contains nodes that are ancestors of z and 
        descendents of nodes in vars. */
    /* TODO */

    /* init z in grad table */

    /* compute the gradients for each variable */
    for (typename std::vector<Variable<T> *>::iterator vit = vars.begin(); vit != vars.end(); vit++) {
        err = internal::build_grad(*vit, graph, table, NULL);

        if (err != 0) { return err; }
    }

    return (magmadnn_error_t) 0;
}
template magmadnn_error_t get_grad_table(std::vector<Variable<int> *>& vars, Operation<int> *graph, GradTable<int> &table);
template magmadnn_error_t get_grad_table(std::vector<Variable<float> *>& vars, Operation<float> *graph, GradTable<float> &table);
template magmadnn_error_t get_grad_table(std::vector<Variable<double> *>& vars, Operation<double> *graph, GradTable<double> &table);

// build_grad should only be used internally
namespace internal {

template <typename T>
magmadnn_error_t build_grad(Variable<T> *var, Operation<T> *graph, GradTable<T> &table, Operation<T> **grad) {
    Operation<T> *tmp_grad, *result;
    std::vector<Operation<T> *> bprops;
    magmadnn_error_t err;

    /* get this entry in the grad table */
    tmp_grad = table.get(var);

    /* if not null then we have already calculated this gradient */
    if (tmp_grad != NULL) {
        *grad = tmp_grad;
        return (magmadnn_error_t) 0;
    }

    for (std::vector<Operation<T> *>::iterator vit = var->get_consumers().begin(); 
        vit != var->get_consumers().end(); vit++) {

        err = build_grad(*vit, graph, table, &tmp_grad);
        if (err != 0) return err;

        bprops.push_back(vit->grad());
    }

    /* result = sum(bprops) */
    table.set(var, result);
    *grad = result;

    return (magmadnn_error_t) 0;
}
template magmadnn_error_t build_grad(Variable<int>* var, Operation<int> *graph, GradTable<int> &table);
template magmadnn_error_t build_grad(Variable<float>* var, Operation<float> *graph, GradTable<float> &table);
template magmadnn_error_t build_grad(Variable<double>* var, Operation<double> *graph, GradTable<double> &table);


}   // namespace internal

}   // namespace op
}   // namespace magmadnn