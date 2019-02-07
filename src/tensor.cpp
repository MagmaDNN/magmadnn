#include "tensor.h"



template <typename T>
tensor<T>::tensor(std::vector<int> shape) { }

template <typename T>
tensor<T>::tensor(std::vector<int> shape, device_t device_id) { }

template <typename T>
tensor<T>::tensor(std::vector<int> shape, T fill) { }


template <typename T>
tensor<T>::tensor(std::vector<int> shape, T fill, device_t device_id) { }

template <typename T>
tensor<T>::~tensor() { }


template <typename T>
T tensor<T>::get(const std::vector<int>& idx) { }


template <typename T>
void tensor<T>::set(const std::vector<int>& idx, T val) { }


template <typename T>
void tensor<T>::init(std::vector<int>& shape, T fill, device_t device_id) { }

template <typename T>
int tensor<T>::get_flattened_index(const std::vector<int>& idx) { }

