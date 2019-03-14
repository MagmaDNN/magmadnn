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
#include "types.h"
#include "memory/memorymanager.h"
#include "tensor_internal.h"

namespace skepsi {

/* Default values for tensors.
   Initialize to CPU 0 if not indicated otherwise.
   And don't fill the tensor on creation unless specified.
*/
const memory_t TENSOR_DEFAULT_MEM_TYPE = HOST;
const device_t TENSOR_DEFAULT_DEVICE_ID = (device_t) 0;
const tensor_fill_t TENSOR_DEFAULT_FILL_TYPE = NONE;
const tensor_filler_t<float> TENSOR_DEFAULT_FILLER = { TENSOR_DEFAULT_FILL_TYPE, {} };


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

	/** Initializes tensor with the given shape. Creates a new memory manager for this tensor. Uses the given device type and device id.
	 * @param shape a vector of axis sizes
	 * @param device the type of device
	 * @param device_id the id of the device to be used
	 */
	Tensor(std::vector<unsigned int> shape, memory_t mem_type, device_t device_id);

	/** Initializes tensor with the given shape. Creates a new memory manager for this tensor. Sets every value of tensor to _fill_.
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


	/** Copies data from src[begin_idx] to src[begin_idx+size] into this tensor.
	 * @param src 
	 * @param begin_idx 
	 * @param size 
	 * @return skepsi_error_t non-zero if error
	 */
	skepsi_error_t copy_from(const Tensor<T>& src, unsigned int begin_idx, unsigned int size);

	/** Copies the tensor src into this tensor.
	 * @param src 
	 * @return skepsi_error_t non-zero if error.
	 */
	skepsi_error_t copy_from(const Tensor<T>& src);


	/** gets the value at the given index.
	 * @param idx indices to retreive value from
	 * @return the value at idx
	 */
	T get(const std::vector<int>& idx);

	/** gets the value at the given index.
	 * @param idx indices to retreive value from
	 * @return the value at idx
	 */
	T get(unsigned int flattened_idx);

	/** sets the value at the given index.
	 * @param idx indices to set value at
	 * @param val value to write into idx
	 */
	void set(const std::vector<int>& idx, T val);

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

private:
	void init(std::vector<unsigned int>& shape, tensor_filler_t<T> filler, memory_t mem_type, device_t device_id);
	unsigned int get_flattened_index(const std::vector<int>& idx);

	MemoryManager<T> *mem_manager;	/* allocated by init */
	
	std::vector<unsigned int> shape;	/* tensor axes (shape) */
	unsigned int size;		/* total number of elements in tensor */
	memory_t mem_type;		/* the type of memory to use for this tensor */
	device_t device_id;		/* device number i.e. gpu0 or cpu1 */

};

} // namespace skepsi
