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

namespace skepsi {

template <typename T>
class tensor {
public:

	/** Initializes tensor with the given shape. Creates a new memory manager for this tensor. Defaults to the cpu.
	 *	@param shape a vector of axis sizes
	 */
	tensor(std::vector<int> shape);

	/** Initializes tensor with the given shape. Creates a new memory manager for this tensor. Uses the given device type and device id.
	 * @param shape a vector of axis sizes
	 * @param device the type of device
	 * @param device_id the id of the device to be used
	 */
	tensor(std::vector<int> shape, device_t device_id);

	/** Initializes tensor with the given shape. Creates a new memory manager for this tensor. Sets every value of tensor to _fill_.
	 * @param shape a vector of axis sizes
	 * @param fill value to set every element of tensor
	 */
	tensor(std::vector<int> shape, T fill);
	
	/** Initializes tensor with the given shape, fill, device, and device id. Creates a new memory manager.
	 * @param shape a vector of axis sizes
	 * @param fill value to set every element of tensor
	 * @param device device type tensor will use
	 * @param device_id id of device to use
	 */
	tensor(std::vector<int> shape, T fill, device_t device_id);

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
	
	memorymanager<T>* get_memory_manager() { return this->mem_manager; }
	std::vector<int> get_shape() { return this->shape; }
	device_t get_device_id() { return this->device_id; }

private:
	void init(std::vector<int>& shape, T fill, device_t device_id);
	int get_flattened_index(const std::vector<int>& idx);

	memorymanager<T> *mem_manager;	/* allocated by init */
	
	std::vector<int> shape;	/* tensor axes (shape) */
	device_t device_id;		/* device number i.e. gpu0 or cpu1 */

};

} // namespace skepsi