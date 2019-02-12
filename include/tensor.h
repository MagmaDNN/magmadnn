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
#include "memorymanager.h"
#include "tensor_utilities.h"

namespace skepsi {

/* Default values for tensors.
   Initialize to CPU 0 if not indicated otherwise.
   And don't fill the tensor on creation unless specified.
*/
const memory_t TENSOR_DEFAULT_MEM_TYPE = HOST;
const device_t TENSOR_DEFAULT_DEVICE_ID = (device_t) 0;
const tensor_fill_t TENSOR_DEFAULT_FILL_TYPE = NONE;
const tensor_filler_t TENSOR_DEFAULT_FILLER = { TENSOR_DEFAULT_FILL_TYPE, {} };


template <typename T>
class tensor {
public:

	/** Initializes tensor with the given shape. Creates a new memory manager for this tensor. Defaults to the cpu.
	 *	@param shape a vector of axis sizes
	 */
	tensor(std::vector<unsigned int> shape);

	/** Initializes tensor with the given shape. Creates a new memory manager for this tensor.
	 * @param shape
	 * @param mem_type
	 */
	tensor(std::vector<unsigned int> shape, memory_t mem_type);

	/** Initializes tensor with the given shape. Creates a new memory manager for this tensor. Uses the given device type and device id.
	 * @param shape a vector of axis sizes
	 * @param device the type of device
	 * @param device_id the id of the device to be used
	 */
	tensor(std::vector<unsigned int> shape, memory_t mem_type, device_t device_id);

	/** Initializes tensor with the given shape. Creates a new memory manager for this tensor. Sets every value of tensor to _fill_.
	 * @param shape a vector of axis sizes
	 * @param fill value to set every element of tensor
	 */
	tensor(std::vector<unsigned int> shape, tensor_filler_t filler);

	/**
	 * @param shape 
	 * @param fill 
	 * @param mem_type 
	 */
	tensor(std::vector<unsigned int> shape, tensor_filler_t filler, memory_t mem_type);
	
	/** Initializes tensor with the given shape, fill, device, and device id. Creates a new memory manager.
	 * @param shape a vector of axis sizes
	 * @param fill value to set every element of tensor
	 * @param device device type tensor will use
	 * @param device_id id of device to use
	 */
	tensor(std::vector<unsigned int> shape, tensor_filler_t filler, memory_t mem_type, device_t device_id);

	/** Free tensor memory
	 */
	~tensor();


	/** gets the value at the given index.
	 * @param idx indices to retreive value from
	 * @return the value at idx
	 */
	T get(const std::vector<int>& idx);

	/** sets the value at the given index.
	 * @param idx indices to set value at
	 * @param val value to write into idx
	 */
	void set(const std::vector<int>& idx, T val);	
	

	/** Returns the memory manager used by this tensor
	 * @return memorymanager<T>* 
	 */
	memorymanager<T>* get_memory_manager() { return this->mem_manager; }

	/** returns a <i>copy</i> of the shape of this tensor.
	 * @return std::vector<int> 
	 */
	std::vector<unsigned int> get_shape() { return this->shape; }

	/** returns the number of elements in tensor
	 * @return unsigned int total number of elements in tensor
	 */
	unsigned int get_size() { return this->size; }

	/** The device id used by this tensor.
	 * @return device_t 
	 */
	device_t get_device_id() { return this->device_id; }

private:
	void init(std::vector<unsigned int>& shape, tensor_filler_t filler, memory_t mem_type, device_t device_id);
	unsigned int get_flattened_index(const std::vector<int>& idx);

	memorymanager<T> *mem_manager;	/* allocated by init */
	
	std::vector<unsigned int> shape;	/* tensor axes (shape) */
	unsigned int size;		/* total number of elements in tensor */
	memory_t mem_type;		/* the type of memory to use for this tensor */
	device_t device_id;		/* device number i.e. gpu0 or cpu1 */

};

} // namespace skepsi