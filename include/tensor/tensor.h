/**
 * @file tensor.h
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-02-07
 *
 * @copyright Copyright (c) 2019
 */
#pragma once

#include <memory>
#include <vector>
#include "data_types.h"
#include "memory/memorymanager.h"
#include "tensor_internal.h"
#include "types.h"
#include "utilities_internal.h"

#if defined(_HAS_CUDA_)
#include "cudnn.h"
#endif

namespace magmadnn {

/* Default values for tensors.
   Initialize to CPU 0 if not indicated otherwise.
   And don't fill the tensor on creation unless specified.
*/
const memory_t TENSOR_DEFAULT_MEM_TYPE = HOST;
const device_t TENSOR_DEFAULT_DEVICE_ID = (device_t) 0;
const tensor_fill_t TENSOR_DEFAULT_FILL_TYPE = NONE;
const tensor_filler_t TENSOR_DEFAULT_FILLER = {TENSOR_DEFAULT_FILL_TYPE, {}};

class Tensor {
   public:
    Tensor();

    explicit Tensor(const std::vector<index_t>& shape, DataType dtype = FLOAT, tensor_filler_t filler = {NONE, {}},
                    memory_t mem_type = HOST, device_t device_id = 0);

    Tensor(const Tensor& t);

    Tensor(Tensor&& t);

    friend void swap(Tensor& left, Tensor& right);

    Tensor& operator=(Tensor t);

    /** Free tensor memory
     */
    ~Tensor();

    /** Copies data from src[begin_idx] to src[begin_idx+size] into this tensor.
     * @param src
     * @param begin_idx
     * @param size
     * @return magmadnn_error_t non-zero if error
     */
    magmadnn_error_t copy_from(const Tensor& src, index_t begin_idx, size_t size);

    /** Copies the tensor src into this tensor.
     * @param src
     * @return magmadnn_error_t non-zero if error.
     */
    magmadnn_error_t copy_from(const Tensor& src);

    /** Truncates the tensor src to match the dimensions in dims
     * @param src
     * @param dims should have same size as src.get_shape()
     * @return magmadnn_error_t non-zero if error
     */
    magmadnn_error_t copy_from(const Tensor& src, const std::vector<index_t>& dims);

    /** gets the value at the given index.
     * @param idx indices to retreive value from
     * @return the value at idx
     */
    template <typename T>
    T get(const std::vector<index_t>& idx) const;

    /** gets the value at the given index.
     * @param idx indices to retreive value from
     * @return the value at idx
     */
    template <typename T>
    T get(index_t flattened_idx) const;

    /** sets the value at the given index.
     * @param idx indices to set value at
     * @param val value to write into idx
     */
    template <typename T>
    void set(const std::vector<index_t>& idx, T val);

    /** sets the value at the given index.
     * @param idx indices to set value at
     * @param val value to write into idx
     */
    template <typename T>
    void set(index_t flattened_idx, T val);

    /** Returns the memory manager used by this tensor
     * @return MemoryManager<T>*
     */
    MemoryManager* get_memory_manager() { return this->memory_manager_ptr.get(); }
    const MemoryManager* get_memory_manager() const { return this->memory_manager_ptr.get(); }

    /** returns a <i>copy</i> of the shape of this tensor.
     * @deprecated since v1.2
     * @see shape()
     * @return std::vector<int>
     */
    const std::vector<index_t>& get_shape() const { return this->shape_; }

    /** returns the shape of this tensor
     * @return const std::vector<unsigned int>&
     */
    const std::vector<index_t>& shape() const { return this->shape_; }

    /** returns the axis size at idx of shape (i.e. shape[idx])
     * @deprecated since v1.2
     * @see shape(unsigned int)
     * @param idx
     * @return unsigned int
     */
    index_t get_shape(index_t idx) const { return this->shape_.at(idx); }

    /** returns the axis size at idx of shape
     * @param idx
     * @return unsigned int
     */
    index_t shape(index_t idx) const { return this->shape_.at(idx); }

    /** returns the number of elements in tensor
     * @deprecated since v1.2
     * @see size()
     * @return unsigned int total number of elements in tensor
     */
    size_t get_size() const { return this->size_; }

    /** returns the number of elements in tensor
     * @return unsigned int
     */
    size_t size() const { return this->size_; }

    DataType dtype() const { return this->dtype_; }

    /** returns the strides of this tensor
     * @return const std::vector<unsigned int>&
     */
    const std::vector<size_t>& strides() const { return this->strides_; }

    /** returns the pointer used by the memory manager.
     * @return T*
     */
    template <typename T>
    T* get_ptr() {
        return this->memory_manager_ptr->get_ptr<T>();
    }

    template <typename T>
    const T* get_ptr() const {
        return this->memory_manager_ptr->get_ptr<T>();
    }

    /** returns the memory type of this tensor
     * @return memory_t
     */
    memory_t get_memory_type() const { return this->mem_type_; }

    /** The device id used by this tensor.
     * @return device_t
     */
    device_t get_device_id() const { return this->device_id_; }

#if defined(_HAS_CUDA_)
    cudnnTensorDescriptor_t get_cudnn_tensor_descriptor() const { return desc_; }
#endif

    /** changes shape of tensor to match dims
     * @param dims should have the same size as this->size
     */
    void reshape(const std::vector<index_t>& dims);

    /** removes axes with length 1
     */
    void squeeze();

    /** adds a dimension with length 1 along axis dim
     * @param dim
     */
    void unsqueeze(index_t dim = 0);

   private:
    inline index_t get_flattened_index(const std::vector<index_t>& idx) const;
    index_t get_flattened_index_old(const std::vector<index_t>& idx) const;

/* device specific code */
#if defined(_HAS_CUDA_)
    void init_cudnn_descriptor();
    void free_cudnn_descriptor();

    cudnnTensorDescriptor_t desc_;
#endif

    std::shared_ptr<MemoryManager> memory_manager_ptr; /* allocated by init */

    std::vector<index_t> shape_;  /* tensor axes (shape) */
    DataType dtype_;              /* tensor data type */
    std::vector<size_t> strides_; /* axis strides in memory */
    size_t size_;                 /* total number of elements in tensor */
    memory_t mem_type_;           /* the type of memory to use for this tensor */
    device_t device_id_;          /* device number i.e. gpu0 or cpu1 */
};

}  // namespace magmadnn