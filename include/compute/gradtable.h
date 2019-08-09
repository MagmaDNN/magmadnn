/**
 * @file gradtable.h
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-05-17
 *
 * @copyright Copyright (c) 2019
 */
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include "compute/operation.h"
#include "compute/variable.h"

namespace magmadnn {
namespace op {

/** GradTable class.
 * @tparam T Numeric
 */
class GradTable {
   public:
    /** Constructs a new grad table.
     */
    GradTable();

    /** returns the size of this grad table.
     * @return unsigned int the size of this grad table
     */
    unsigned int get_size();

    /** Takes a variable and returns the tree to compute its gradient.
     * @param var
     * @return Operation<T>*
     */
    std::pair<bool, std::reference_wrapper<Tensor>> get(Operation* var);

    /** Sets var's gradient to grad.
     * @param var
     * @param grad
     */
    void set(Operation* var, Tensor& grad);

    /** Removes all entries.
     */
    void clear();

   protected:
    std::map<Operation*, std::reference_wrapper<Tensor>> _table;  // the underlying table to store data
    typename std::map<Operation*, std::reference_wrapper<Tensor>>::iterator tmp_map_iterator;
};

}  // namespace op
}  // namespace magmadnn
