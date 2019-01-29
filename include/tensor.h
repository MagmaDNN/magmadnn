#pragma once

#include <vector>
#include "types.h"

template <typename T>
class tensor {
public:

	tensor(std::vector<int> shape);
	tensor(std::vector<int> shape, device_t device);
	tensor(std::vector<int> shape, T fill);
	tensor(std::vector<int> shape, T fill, device_t device);

	

	
	

private:

	std::vector<int> shape;
	device_t device;

};
