/**
 * @file gradients.cpp
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-05-17
 *
 * @copyright Copyright (c) 2019
 */
#include "compute/gradients.h"

#include "utilities_internal.h"

namespace magmadnn {
namespace op {

magmadnn_error_t get_grad_table(const std::vector<Operation *> &vars, Operation *graph, GradTable &table) {
    magmadnn_error_t err;
    Tensor tmp;

    /* prune compute graph:
        construct a new graph G' that only contains nodes that are ancestors of
       graph and descendents of nodes in vars. */
    /* TODO */

    /* init Loss in grad table to one */
    Tensor grad_loss({1}, graph->dtype(), {ONE}, graph->get_memory_type());

    table.set(graph, grad_loss);

    /* compute the gradients for each variable */
    for (std::vector<Operation *>::const_iterator vit = vars.begin(); vit != vars.end(); vit++) {
        if (*vit != NULL) {
            err = internal::build_grad(*vit, graph, table, tmp);
        } else {
            return (magmadnn_error_t) 1;
        }

        if (err != 0) {
            return err;
        }
    }

    return (magmadnn_error_t) 0;
}

}  // namespace op

// build_grad should only be used internally
namespace internal {

magmadnn_error_t build_grad(op::Operation *var, op::Operation *graph, op::GradTable &table, Tensor &grad) {
    // Tensor *tmp_grad, *bprop, *result;
    op::Operation *consumer;
    std::vector<op::Operation *> consumers;
    std::vector<std::reference_wrapper<Tensor>> bprops;
    magmadnn_error_t err;

    /* error on null values */
    if (var == nullptr || graph == nullptr) return (magmadnn_error_t) 1;

    /* get this entry in the grad table */
    auto const &res = table.get(var);

    /* we've already calculated this gradient */
    if (res.first) {
        grad = res.second;
        return (magmadnn_error_t) 0;
    }

    Tensor &tmp_grad = res.second;

    /* build gradients for each consumer to this operation in order to properly
     * calculate ours */
    consumers = var->get_consumers();
    for (std::vector<op::Operation *>::iterator vit = consumers.begin(); vit != consumers.end(); vit++) {
        consumer = (*vit);

        /* if this is null for some reason stop here */
        if (consumer == nullptr) continue;

        /* build the gradient for consumer and keep track of it in bprops */
        err = build_grad(consumer, graph, table, tmp_grad);
        if (err != 0) return err;

        Tensor &bprop = consumer->grad(consumer, var, tmp_grad);
        bprops.push_back(bprop);
    }

    /* sum of each partial gradient is the total gradient */
    /* TODO : no need to sum if just one */
    if (bprops.size() == 0) {
        return (magmadnn_error_t) 2;
    } else if (bprops.size() == 1) {
        grad = bprops.at(0);
    } else if (bprops.size() == 2) {
        /* TODO : Add and sum tensors */
        // result = NULL;

        LOG(ERROR) << "Implement add in gradients\n";
    } else {
        /*
        result = bprops.at(0);
        for (unsigned int i = 1; i < bprops.size(); i++) {
            result = op::add(result, bprops.at(i));
        }*/
        // result = NULL;

        LOG(ERROR) << "Implement sum in gradients\n";
    }

    table.set(var, grad);

    return (magmadnn_error_t) 0;
}

}  // namespace internal
}  // namespace magmadnn