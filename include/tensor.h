#pragma once

#include <vector>
#include "types.h"

template <typename T>
class tensor {
public:

	tensor(std::vector<int> shape);
	tensor(std::vector<int> shape, device_t device, device_id_t device_id);
	tensor(std::vector<int> shape, T fill);
	tensor(std::vector<int> shape, T fill, device_t device, device_id_t device_id);
	~tensor();

	T get(const std::vector<int>& idx);
	T set(const std::vector<int>& idx, T val);	
	
	memorymanager<T> get_memory_manager() { return this->mem_manager; }
	std::vector<int> get_shape() { return this->shape; }
	device_t get_device() { return this->device; }

private:
	init(std::vector<int>& shape, T fill, device_t device, device_id_t device_id);
	int get_flattened_index(const std::vector<int>& idx);
	void set_device();

	memorymanager<T> *mem_manager;
	
	std::vector<int> shape;	/* tensor axes (shape) */
	device_t device;		/* device enum - CUDA or HOST */
	device_id_t device_id;	/* device number i.e. gpu0 or cpu1 */

};
