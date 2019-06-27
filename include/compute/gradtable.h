/**
 * @file gradtable.h
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-05-17
 * 
 * @copyright Copyright (c) 2019
 */
#pragma once

#include <map>
#include <string>
#include <cstdint>
#include "compute/operation.h"
#include "compute/variable.h"

namespace magmadnn {
namespace op {

/** GradTable class.
 * @tparam T Numeric
 */
template <typename T>
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
    Tensor<T>* get(Operation<T>* var);

    /** Sets var's gradient to grad.
     * @param var 
     * @param grad 
     */
    void set(Operation<T>* var, Tensor<T>* grad);

    /** Removes all entries.
     */
    void clear();

protected:
    std::map<Operation<T> *, Tensor<T>* > _table;   // the underlying table to store data
    typename std::map<Operation<T> *, Tensor<T> *>::iterator tmp_map_iterator;

};

}   // namespace op
}   // namespace magmadnn
