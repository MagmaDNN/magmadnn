/**
 * @file tensor.cpp
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-02-07
 *
 * @copyright Copyright (c) 2019
 */
#include "tensor/tensor.h"

namespace magmadnn {

template <typename T>
Tensor<T>::Tensor(std::vector<unsigned int> shape) {
    init(shape, {TENSOR_DEFAULT_FILL_TYPE, {}}, TENSOR_DEFAULT_MEM_TYPE, TENSOR_DEFAULT_DEVICE_ID);
}

template <typename T>
Tensor<T>::Tensor(std::vector<unsigned int> shape, memory_t mem_type) {
    init(shape, {TENSOR_DEFAULT_FILL_TYPE, {}}, mem_type, TENSOR_DEFAULT_DEVICE_ID);
}

template <typename T>
Tensor<T>::Tensor(std::vector<unsigned int> shape, memory_t mem_type, device_t device_id) {
    init(shape, {TENSOR_DEFAULT_FILL_TYPE, {}}, mem_type, device_id);
}

template <typename T>
Tensor<T>::Tensor(std::vector<unsigned int> shape, tensor_filler_t<T> filler) {
    init(shape, filler, TENSOR_DEFAULT_MEM_TYPE, TENSOR_DEFAULT_DEVICE_ID);
}

template <typename T>
Tensor<T>::Tensor(std::vector<unsigned int> shape, tensor_filler_t<T> filler, memory_t mem_type) {
    init(shape, filler, mem_type, TENSOR_DEFAULT_DEVICE_ID);
}

template <typename T>
Tensor<T>::Tensor(std::vector<unsigned int> shape, tensor_filler_t<T> filler, memory_t mem_type, device_t device_id) {
    init(shape, filler, mem_type, device_id);
}

template <typename T>
Tensor<T>::~Tensor() {
    delete mem_manager;

#if defined(_HAS_CUDA_)
    this->free_cudnn_descriptor();
#endif
}

template <typename T>
void Tensor<T>::init(std::vector<unsigned int>& shape, tensor_filler_t<T> filler, memory_t mem_type,
                     device_t device_id) {
    // tensor must have at least 1 axis
    assert(shape.size() != 0);

    // initialize class variables
    this->shape = shape;
    this->mem_type = mem_type;
    this->device_id = device_id;

    // calculate stride values
    this->strides.resize(shape.size());
    unsigned int tmp_stride = 1;
    for (int i = ((int) shape.size()) - 1; i >= 0; i--) {
        strides[i] = tmp_stride;
        tmp_stride *= shape[i];
    }

    // calculate the total number of elements
    this->size = 1;
    for (unsigned int i = 0; i < shape.size(); i++) {
        this->size *= shape[i];
    }

    // create memory manager
    this->mem_manager = new MemoryManager<T>(size, mem_type, device_id);

    internal::fill_memory(*mem_manager, filler);

#if defined(_HAS_CUDA_)
    /* create a cudnn descriptor */
    this->init_cudnn_descriptor();
#endif
}

template <typename T>
magmadnn_error_t Tensor<T>::copy_from(const Tensor<T>& src, unsigned int begin_idx, unsigned int size, unsigned int write_from) {
    assert(this->size >= size + write_from);
    assert(src.size >= (begin_idx + size));

    return this->mem_manager->copy_from(*src.get_memory_manager(), begin_idx, size, write_from);
}

template <typename T>
magmadnn_error_t Tensor<T>::copy_from(const Tensor<T>& src) {
    return copy_from(src, 0, src.get_size(), 0);
}

template <typename T>
magmadnn_error_t Tensor<T>::copy_from(const Tensor<T>& src, const std::vector<unsigned int>& dims) {
    for (unsigned int i = 0; i < dims.size(); i++) {
        assert(dims[i] != 0);
        assert(dims[i] <= src.get_shape()[i]);
    }
    magmadnn_error_t m = 0;
    std::vector<unsigned int> target_shape(dims.size(), 0);
    int curr_pos = target_shape.size() - 1;
    while (curr_pos >= 0) {
        curr_pos = target_shape.size() - 1;
        this->set(target_shape, src.get(target_shape));
        target_shape[curr_pos]++;
        while (target_shape[curr_pos] == dims[curr_pos]) {
            target_shape[curr_pos] = 0;
            curr_pos--;
            if (curr_pos < 0) break;
            target_shape[curr_pos]++;
        }
    }
    return m;
}

template <typename T>
T Tensor<T>::get(const std::vector<int>& idx) const {
    std::vector<unsigned int> ui_vec(idx.begin(), idx.end());
    return mem_manager->get(get_flattened_index(ui_vec));
}

template <typename T>
T Tensor<T>::get(const std::vector<unsigned int>& idx) const {
    return mem_manager->get(get_flattened_index(idx));
}

template <typename T>
T Tensor<T>::get(const std::initializer_list<int>& idx) const {
    std::vector<unsigned int> ui_vec(idx.begin(), idx.end());
    return mem_manager->get(get_flattened_index(ui_vec));
}

template <typename T>
T Tensor<T>::get(const std::initializer_list<unsigned int>& idx) const {
    return mem_manager->get(get_flattened_index(idx));
}

template <typename T>
T Tensor<T>::get(unsigned int flattened_idx) const {
    return mem_manager->get(flattened_idx);
}

template <typename T>
const T Tensor<T>::operator[](unsigned int idx) const {
    return mem_manager->get(idx);
}

template <typename T>
const T Tensor<T>::operator[](const std::initializer_list<unsigned int>& idx) {
    return mem_manager->get(get_flattened_index(idx));
}

template <typename T>
void Tensor<T>::set(const std::vector<int>& idx, T val) {
    std::vector<unsigned int> ui_vec(idx.begin(), idx.end());
    mem_manager->set(get_flattened_index(ui_vec), val);
}

template <typename T>
void Tensor<T>::set(const std::vector<unsigned int>& idx, T val) {
    mem_manager->set(get_flattened_index(idx), val);
}

template <typename T>
void Tensor<T>::set(const std::initializer_list<int>& idx, T val) {
    std::vector<unsigned int> ui_vec(idx.begin(), idx.end());
    mem_manager->set(get_flattened_index(ui_vec), val);
}

template <typename T>
void Tensor<T>::set(const std::initializer_list<unsigned int>& idx, T val) {
    mem_manager->set(get_flattened_index(idx), val);
}

template <typename T>
void Tensor<T>::set(unsigned int flattened_idx, T val) {
    mem_manager->set(flattened_idx, val);
}

template <typename T>
unsigned int Tensor<T>::get_shape(unsigned int idx) const {
    assert(idx < this->shape.size());
    return this->shape[idx];
}

template <typename T>
unsigned int Tensor<T>::get_flattened_index(const std::vector<unsigned int>& idx) const {
    unsigned int flattened_idx = 0;

    for (unsigned int i = 0; i < idx.size(); i++) {
        flattened_idx += idx[i] * strides[i];
    }

    return flattened_idx;
}

/** @deprecated */
template <typename T>
unsigned int Tensor<T>::get_flattened_index_old(const std::vector<unsigned int>& idx) const {
    unsigned int jump_size = 1;  // the total amount to jump to get to next axis
    unsigned int flattened_idx = 0;

    for (int i = ((int) idx.size()) - 1; i >= 0; i--) {
        flattened_idx += idx[i] * jump_size;
        jump_size *= shape[i];
    }
    return flattened_idx;
}

template <typename T>
void Tensor<T>::reshape(const std::vector<unsigned int>& dims) {
    long long dims_size = 1;
    for (unsigned int i = 0; i < dims.size(); i++) dims_size *= dims[i];
    assert(size == dims_size);
    shape = dims;

    /* update strides */
    strides.resize(dims.size());
    unsigned int tmp_stride = 1;
    for (int i = ((int) shape.size()) - 1; i >= 0; i--) {
        strides[i] = tmp_stride;
        tmp_stride *= shape[i];
    }

#if defined(_HAS_CUDA_)
    /* update cudnn descriptor if on GPU */
    int n = 1, c = 1, h = 1, w = 1;
    if (shape.size() == 4) {
        n = shape[0];
        c = shape[1];
        h = shape[2];
        w = shape[3];
    } else if (shape.size() == 3) {
        n = shape[0];
        c = shape[1];
        h = shape[2];
    } else if (shape.size() == 2) {
        n = shape[0];
        c = shape[1];
    } else if (shape.size() == 1) {
        n = shape[0];
    } else {
        fprintf(stderr, "Cannot create tensor descriptor for tensor of this shape\n");
    }
    cudnnErrchk(cudnnSetTensor4dDescriptor(this->desc, CUDNN_TENSOR_NCHW,
                                           ::magmadnn::internal::get_cudnn_data_type(static_cast<T>(0)), n, c, h, w));
#endif
}

template <typename T>
void Tensor<T>::squeeze() {
    std::vector<unsigned int> new_shape;
    for (unsigned int i = 0; i < shape.size(); i++) {
        if (shape[i] > 1) new_shape.push_back(shape[i]);
    }
    if (new_shape.size() == 0) new_shape.push_back(1);
    shape = new_shape;
}

template <typename T>
void Tensor<T>::unsqueeze(unsigned int dim) {
    assert(dim <= shape.size());
    shape.insert(shape.begin() + dim, 1, 1);
}

#if defined(_HAS_CUDA_)
template <typename T>
void Tensor<T>::init_cudnn_descriptor() {
    int n = 1, c = 1, h = 1, w = 1;

    cudnnCreateTensorDescriptor(&desc);

    if (shape.size() == 4) {
        n = shape[0];
        c = shape[1];
        h = shape[2];
        w = shape[3];
    } else if (shape.size() == 3) {
        n = shape[0];
        c = shape[1];
        h = shape[2];
    } else if (shape.size() == 2) {
        n = shape[0];
        c = shape[1];
    } else if (shape.size() == 1) {
        n = shape[0];
    } else {
        fprintf(stderr, "Cannot create tensor descriptor for tensor of this shape\n");
    }

    cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, internal::get_cudnn_data_type((T) 0), n, c, h, w);
}

template <typename T>
void Tensor<T>::free_cudnn_descriptor() {
    cudnnDestroyTensorDescriptor(this->desc);
}
#endif

/* COMPILE FOR INT, FLOAT, AND DOUBLE */
template class Tensor<int>;
template class Tensor<float>;
template class Tensor<double>;

}  // namespace magmadnn
