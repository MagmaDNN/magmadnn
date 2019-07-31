/**
 * @file tensor.h
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-02-07
 *
 * @copyright Copyright (c) 2019
 */
#pragma once

#include <vector>
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
const tensor_filler_t<float> TENSOR_DEFAULT_FILLER = {TENSOR_DEFAULT_FILL_TYPE, {}};

template <typename T>
class Tensor {
   public:
    /** Initializes tensor with the given shape. Creates a new memory manager for this tensor. Defaults to the cpu.
     *	@param shape a vector of axis sizes
     */
    Tensor(std::vector<unsigned int> shape);

    /** Initializes tensor with the given shape. Creates a new memory manager for this tensor.
     * @param shape
     * @param mem_type
     */
    Tensor(std::vector<unsigned int> shape, memory_t mem_type);

    /** Initializes tensor with the given shape. Creates a new memory manager for this tensor. Uses the given device
     * type and device id.
     * @param shape a vector of axis sizes
     * @param device the type of device
     * @param device_id the id of the device to be used
     */
    Tensor(std::vector<unsigned int> shape, memory_t mem_type, device_t device_id);

    /** Initializes tensor with the given shape. Creates a new memory manager for this tensor. Sets every value of
     * tensor to _fill_.
     * @param shape a vector of axis sizes
     * @param fill value to set every element of tensor
     */
    Tensor(std::vector<unsigned int> shape, tensor_filler_t<T> filler);

    /**
     * @param shape
     * @param fill
     * @param mem_type
     */
    Tensor(std::vector<unsigned int> shape, tensor_filler_t<T> filler, memory_t mem_type);

    /** Initializes tensor with the given shape, fill, device, and device id. Creates a new memory manager.
     * @param shape a vector of axis sizes
     * @param fill value to set every element of tensor
     * @param device device type tensor will use
     * @param device_id id of device to use
     */
    Tensor(std::vector<unsigned int> shape, tensor_filler_t<T> filler, memory_t mem_type, device_t device_id);

    /** Free tensor memory
     */
    ~Tensor();

    /** Copies data from src[begin_idx] to src[begin_idx+size] into this[write_from] to this[write_from+size].
     * @param src
     * @param begin_idx
     * @param size
	 * @param write_from
     * @return magmadnn_error_t non-zero if error
     */
    magmadnn_error_t copy_from(const Tensor<T>& src, unsigned int begin_idx, unsigned int size, unsigned int write_from = 0);

    /** Copies the tensor src into this tensor.
     * @param src
     * @return magmadnn_error_t non-zero if error.
     */
    magmadnn_error_t copy_from(const Tensor<T>& src);

    /** Truncates the tensor src to match the dimensions in dims
     * @param src
     * @param dims should have same size as src.get_shape()
     * @return magmadnn_error_t non-zero if error
     */
    magmadnn_error_t copy_from(const Tensor<T>& src, const std::vector<unsigned int>& dims);

    /** gets the value at the given index.
     * @param idx indices to retreive value from
     * @return the value at idx
     */
    T get(const std::vector<int>& idx) const;

    /** gets the value at the given index.
     * @param idx indices to retrieve value from
     * @return T the value at idx
     */
    T get(const std::vector<unsigned int>& idx) const;

    /** gets the value at the given index.
     * @param idx indices to retrieve value from
     * @return T the value at idx
     */
    T get(const std::initializer_list<int>& idx) const;

    /** gets the value at the given index.
     * @param idx indices to retrieve value from
     * @return T the value at idx
     */
    T get(const std::initializer_list<unsigned int>& idx) const;

    /** gets the value at the given index.
     * @param idx indices to retreive value from
     * @return the value at idx
     */
    T get(unsigned int flattened_idx) const;

    /** Gets the value at the given index.
     * @param idx
     * @return const T
     */
    const T operator[](unsigned int idx) const;

    const T operator[](const std::initializer_list<unsigned int>& idx);

    /** sets the value at the given index.
     * @param idx indices to set value at
     * @param val value to write into idx
     */
    void set(const std::vector<int>& idx, T val);

    /** sets the value at the given index.
     * @param idx indices to set value at
     * @param val value to write into idx
     */
    void set(const std::vector<unsigned int>& idx, T val);

    /** sets the value at the given index.
     * @param idx indices to set value at
     * @param val value to write into idx
     */
    void set(const std::initializer_list<int>& idx, T val);

    /** sets the value at the given index.
     * @param idx indices to set value at
     * @param val value to write into idx
     */
    void set(const std::initializer_list<unsigned int>& idx, T val);

    /** sets the value at the given index.
     * @param idx indices to set value at
     * @param val value to write into idx
     */
    void set(unsigned int flattened_idx, T val);

    /** Returns the memory manager used by this tensor
     * @return MemoryManager<T>*
     */
    MemoryManager<T>* get_memory_manager() const { return this->mem_manager; }

    /** returns a <i>copy</i> of the shape of this tensor.
     * @return std::vector<int>
     */
    std::vector<unsigned int> get_shape() const { return this->shape; }

    /** returns the axis size at idx of shape (i.e. shape[idx])
     * @param idx
     * @return unsigned int
     */
    unsigned int get_shape(unsigned int idx) const;

    /** returns the number of elements in tensor
     * @return unsigned int total number of elements in tensor
     */
    unsigned int get_size() const { return this->size; }

    /** returns the pointer used by the memory manager.
     * @return T*
     */
    T* get_ptr() { return this->mem_manager->get_ptr(); }

    /** returns the memory type of this tensor
     * @return memory_t
     */
    memory_t get_memory_type() const { return this->mem_type; }

    /** The device id used by this tensor.
     * @return device_t
     */
    device_t get_device_id() const { return this->device_id; }

#if defined(_HAS_CUDA_)
    cudnnTensorDescriptor_t get_cudnn_tensor_descriptor() { return desc; }
#endif

    /** changes shape of tensor to match dims
     * @param dims should have the same size as this->size
     */
    void reshape(const std::vector<unsigned int>& dims);

    /** removes axes with length 1
     */
    void squeeze();

    /** adds a dimension with length 1 along axis dim
     * @param dim
     */
    void unsqueeze(unsigned int dim = 0);

   private:
    void init(std::vector<unsigned int>& shape, tensor_filler_t<T> filler, memory_t mem_type, device_t device_id);
    unsigned int get_flattened_index(const std::vector<unsigned int>& idx) const;
    unsigned int get_flattened_index_old(const std::vector<unsigned int>& idx) const;

/* device specific code */
#if defined(_HAS_CUDA_)
    void init_cudnn_descriptor();
    void free_cudnn_descriptor();

    cudnnTensorDescriptor_t desc;
#endif

    MemoryManager<T>* mem_manager; /* allocated by init */

    std::vector<unsigned int> shape;   /* tensor axes (shape) */
    std::vector<unsigned int> strides; /* axis strides in memory */
    unsigned int size;                 /* total number of elements in tensor */
    memory_t mem_type;                 /* the type of memory to use for this tensor */
    device_t device_id;                /* device number i.e. gpu0 or cpu1 */
};

/** Tensor typedefs. Shorthand for Tensors of different types.
 */
typedef Tensor<int> tensori_t;
typedef Tensor<float> tensorf_t;
typedef Tensor<double> tensord_t;

}  // namespace magmadnn
