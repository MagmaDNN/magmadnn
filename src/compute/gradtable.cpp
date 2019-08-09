/**
 * @file gradtable.cpp
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-05-17
 *
 * @copyright Copyright (c) 2019
 */
#include "compute/gradtable.h"

namespace magmadnn {
namespace op {

GradTable::GradTable() {
    // init
}

unsigned int GradTable::get_size() { return _table.size(); }

std::pair<bool, std::reference_wrapper<Tensor>> GradTable::get(Operation *var) { return *_table.find(var); }

void GradTable::set(Operation *var, Tensor &grad) {
    if (var == nullptr) return;

    /* add this gradient into the table */
    _table.insert(std::make_pair(var, std::ref(grad)));
}

void GradTable::clear() { this->_table.clear(); }

}  // namespace op
}  // namespace magmadnn