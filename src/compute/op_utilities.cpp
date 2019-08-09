/**
 * @file op_utilities.cpp
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-06-19
 *
 * @copyright Copyright (c) 2019
 */
#include "compute/op_utilities.h"

namespace magmadnn {
namespace op {
namespace utility {

magmadnn_error_t print_compute_graph(::magmadnn::op::Operation *_root, bool debug) {
    std::set<::magmadnn::op::Operation *> visited;
    std::deque<::magmadnn::op::Operation *> to_visit;
    ::magmadnn::op::Operation *cur;
    int (*print)(const char *, ...);
    typename std::vector<::magmadnn::op::Operation *>::const_iterator vit;
    std::vector<index_t>::const_iterator vui_it;

    print = (debug) ? ::magmadnn::internal::debugf : std::printf;

    to_visit.push_back(_root);
    visited.insert(_root);

    while (!to_visit.empty()) {
        cur = to_visit.front();
        to_visit.pop_front();

        print("Operation [%s]:\n", cur->to_string().c_str());

        print("\tShape: {");
        const std::vector<index_t> &out_shape = cur->get_output_shape();
        for (vui_it = out_shape.begin(); vui_it != out_shape.end(); vui_it++) {
            print(" %lu%s", (*vui_it), (vui_it == out_shape.end() - 1) ? " }" : ",");
        }

        print("\n\tConsumers:");
        std::vector<op::Operation *> const &consumers = cur->get_consumers();
        for (vit = consumers.begin(); vit != consumers.end(); vit++) {
            print(" [%s]", (*vit)->to_string().c_str());

            if (visited.find((*vit)) == visited.end()) {
                to_visit.push_back(*vit);
                visited.insert((*vit));
            }
        }

        print("\n\tInputs:");
        std::vector<op::Operation *> const &inputs = cur->get_inputs();
        for (vit = inputs.begin(); vit != inputs.end(); vit++) {
            print(" [%s]", (*vit)->to_string().c_str());

            if (visited.find((*vit)) == visited.end()) {
                to_visit.push_back(*vit);
                visited.insert((*vit));
            }
        }
        print("\n");
    }

    return (magmadnn_error_t) 0;
}

}  // namespace utility
}  // namespace op
}  // namespace magmadnn