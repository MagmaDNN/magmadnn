#include "tensor.h"


template <typename T>


tensor(std::vector<int> shape){ }

tensor(std::vector<int> shape, device_t device, device_id_t device_id){ }

tensor(std::vector<int> shape, T fill){ }


tensor(std::vector<int> shape, T fill, device_t device, device_id_t device_id){ }

~tensor(){ }


T get(const std::vector<int>& idx){ }


void set(const std::vector<int>& idx, T val){ }


init(std::vector<int>& shape, T fill, device_t device, device_id_t device_id){ }
int get_flattened_index(const std::vector<int>& idx){ }

