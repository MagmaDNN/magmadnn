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
Tensor<T>::Tensor() {
    init({0}, {TENSOR_DEFAULT_FILL_TYPE, {}}, TENSOR_DEFAULT_MEM_TYPE, TENSOR_DEFAULT_DEVICE_ID);
}

template <typename T>
Tensor<T>::Tensor(const std::vector<unsigned int>& shape) {
    init(shape, {TENSOR_DEFAULT_FILL_TYPE, {}}, TENSOR_DEFAULT_MEM_TYPE, TENSOR_DEFAULT_DEVICE_ID);
}

template <typename T>
Tensor<T>::Tensor(const std::vector<unsigned int>& shape, memory_t mem_type) {
    init(shape, {TENSOR_DEFAULT_FILL_TYPE, {}}, mem_type, TENSOR_DEFAULT_DEVICE_ID);
}

template <typename T>
Tensor<T>::Tensor(const std::vector<unsigned int>& shape, memory_t mem_type, device_t device_id) {
    init(shape, {TENSOR_DEFAULT_FILL_TYPE, {}}, mem_type, device_id);
}

template <typename T>
Tensor<T>::Tensor(const std::vector<unsigned int>& shape, tensor_filler_t<T> filler) {
    init(shape, filler, TENSOR_DEFAULT_MEM_TYPE, TENSOR_DEFAULT_DEVICE_ID);
}

template <typename T>
Tensor<T>::Tensor(const std::vector<unsigned int>& shape, tensor_filler_t<T> filler, memory_t mem_type) {
    init(shape, filler, mem_type, TENSOR_DEFAULT_DEVICE_ID);
}

template <typename T>
Tensor<T>::Tensor(const std::vector<unsigned int>& shape, tensor_filler_t<T> filler, memory_t mem_type,
                  device_t device_id) {
    init(shape, filler, mem_type, device_id);
}

template <typename T>
Tensor<T>::~Tensor() {
#if defined(_HAS_CUDA_)
    this->free_cudnn_descriptor();
#endif
}

template <typename T>
void Tensor<T>::init(const std::vector<unsigned int>& shape, tensor_filler_t<T> filler, memory_t mem_type,
                     device_t device_id) {
    // tensor must have at least 1 axis
    assert(shape.size() != 0);

    // initialize class variables
    this->_shape = shape;
    this->mem_type = mem_type;
    this->device_id = device_id;

    // calculate stride values
    this->_strides.resize(shape.size());
    unsigned int tmp_stride = 1;
    for (int i = ((int) shape.size()) - 1; i >= 0; i--) {
        _strides[i] = tmp_stride;
        tmp_stride *= shape[i];
    }

    // calculate the total number of elements
    this->_size = 1;
    for (unsigned int i = 0; i < shape.size(); i++) {
        this->_size *= shape[i];
    }

    // create memory manager
    // this->mem_manager = new MemoryManager<T>(size, mem_type, device_id);
    this->memory_manager_ptr = std::make_shared<MemoryManager<T>>(this->_size, mem_type, device_id);

    internal::fill_memory(*memory_manager_ptr, filler);

#if defined(_HAS_CUDA_)
    /* create a cudnn descriptor */
    this->init_cudnn_descriptor();
#endif
}

template <typename T>
Tensor<T>::Tensor(const Tensor<T>& t) {
    this->_shape = t._shape;
    this->_strides = t._strides;
    this->_size = t._size;

    this->mem_type = t.mem_type;
    this->device_id = t.device_id;

    /* shared pointer -- reference count the memory manager */
    /* TODO -- copy from memory */
    this->memory_manager_ptr = t.memory_manager_ptr;

/* TODO -- replace cudnn_descriptor with reference counting */
#if defined(_HAS_CUDA_)
    this->init_cudnn_descriptor();
#endif
}

template <typename T>
Tensor<T>::Tensor(Tensor<T>&& t) {
    swap(*this, t);

    /* TODO -- replace cudnn_descriptor with reference counting */
#if defined(_HAS_CUDA_)
    this->init_cudnn_descriptor();
#endif
}

template <typename T>
void swap(Tensor<T>& left, Tensor<T>& right) {
    using std::swap;

    swap(left._shape, right._shape);
    swap(left._strides, right._strides);
    swap(left._size, right._size);
    swap(left.mem_type, right.mem_type);
    swap(left.device_id, right.device_id);

    swap(left.memory_manager_ptr, right.memory_manager_ptr);
}

template <typename T>
Tensor<T>& Tensor<T>::operator=(Tensor<T> t) {
    /* by passing by value we allow the compiler to elide the copying of an rvalue */

    swap(*this, t);

    /* TODO -- replace cudnn_descriptor with reference counting */
#if defined(_HAS_CUDA_)
    this->init_cudnn_descriptor();
#endif
    return *this;
}

template <typename T>
magmadnn_error_t Tensor<T>::copy_from(const Tensor<T>& src, unsigned int begin_idx, unsigned int size) {
    assert(this->_size >= size);
    assert(src._size >= (begin_idx + size));

    return this->memory_manager_ptr->copy_from(*src.get_memory_manager(), begin_idx, size);
}

template <typename T>
magmadnn_error_t Tensor<T>::copy_from(const Tensor<T>& src) {
    return copy_from(src, 0, src.size());
}

template <typename T>
magmadnn_error_t Tensor<T>::copy_from(const Tensor<T>& src, const std::vector<unsigned int>& dims) {
    for (unsigned int i = 0; i < dims.size(); i++) {
        assert(dims[i] != 0);
        assert(dims[i] <= src.shape(i));
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
    return memory_manager_ptr->get(get_flattened_index(ui_vec));
}

template <typename T>
T Tensor<T>::get(const std::vector<unsigned int>& idx) const {
    return memory_manager_ptr->get(get_flattened_index(idx));
}

template <typename T>
T Tensor<T>::get(const std::initializer_list<int>& idx) const {
    std::vector<unsigned int> ui_vec(idx.begin(), idx.end());
    return memory_manager_ptr->get(get_flattened_index(ui_vec));
}

template <typename T>
T Tensor<T>::get(const std::initializer_list<unsigned int>& idx) const {
    return memory_manager_ptr->get(get_flattened_index(idx));
}

template <typename T>
T Tensor<T>::get(unsigned int flattened_idx) const {
    return memory_manager_ptr->get(flattened_idx);
}

template <typename T>
T Tensor<T>::operator[](unsigned int idx) const {
    return memory_manager_ptr->get(idx);
}

template <typename T>
T Tensor<T>::operator[](const std::initializer_list<unsigned int>& idx) const {
    return memory_manager_ptr->get(get_flattened_index(idx));
}

template <typename T>
void Tensor<T>::set(const std::vector<int>& idx, T val) {
    std::vector<unsigned int> ui_vec(idx.begin(), idx.end());
    memory_manager_ptr->set(get_flattened_index(ui_vec), val);
}

template <typename T>
void Tensor<T>::set(const std::vector<unsigned int>& idx, T val) {
    memory_manager_ptr->set(get_flattened_index(idx), val);
}

template <typename T>
void Tensor<T>::set(const std::initializer_list<int>& idx, T val) {
    std::vector<unsigned int> ui_vec(idx.begin(), idx.end());
    memory_manager_ptr->set(get_flattened_index(ui_vec), val);
}

template <typename T>
void Tensor<T>::set(const std::initializer_list<unsigned int>& idx, T val) {
    memory_manager_ptr->set(get_flattened_index(idx), val);
}

template <typename T>
void Tensor<T>::set(unsigned int flattened_idx, T val) {
    memory_manager_ptr->set(flattened_idx, val);
}

template <typename T>
unsigned int Tensor<T>::get_shape(unsigned int idx) const {
    /* .at automatically checks for out of bounds */
    return this->_shape.at(idx);
}

template <typename T>
unsigned int Tensor<T>::shape(unsigned int idx) const {
    return this->_shape.at(idx);
}

template <typename T>
inline unsigned int Tensor<T>::get_flattened_index(const std::vector<unsigned int>& idx) const {
    unsigned int flattened_idx = 0;

    for (unsigned int i = 0; i < idx.size(); i++) {
        flattened_idx += idx[i] * _strides[i];
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
        jump_size *= _shape[i];
    }
    return flattened_idx;
}

template <typename T>
void Tensor<T>::reshape(const std::vector<unsigned int>& dims) {
    long long dims_size = 1;
    for (unsigned int i = 0; i < dims.size(); i++) dims_size *= dims[i];
    assert(_size == dims_size);
    _shape = dims;

    /* update strides */
    _strides.resize(dims.size());
    unsigned int tmp_stride = 1;
    for (int i = ((int) _shape.size()) - 1; i >= 0; i--) {
        _strides[i] = tmp_stride;
        tmp_stride *= _shape[i];
    }

#if defined(_HAS_CUDA_)
    /* update cudnn descriptor if on GPU */
    int n = 1, c = 1, h = 1, w = 1;
    if (_shape.size() == 4) {
        n = _shape[0];
        c = _shape[1];
        h = _shape[2];
        w = _shape[3];
    } else if (_shape.size() == 3) {
        n = _shape[0];
        c = _shape[1];
        h = _shape[2];
    } else if (_shape.size() == 2) {
        n = _shape[0];
        c = _shape[1];
    } else if (_shape.size() == 1) {
        n = _shape[0];
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
    for (unsigned int i = 0; i < _shape.size(); i++) {
        if (_shape[i] > 1) new_shape.push_back(_shape[i]);
    }
    if (new_shape.size() == 0) new_shape.push_back(1);
    _shape = new_shape;
}

template <typename T>
void Tensor<T>::unsqueeze(unsigned int dim) {
    assert(dim <= _shape.size());
    _shape.insert(_shape.begin() + dim, 1, 1);
}

#if defined(_HAS_CUDA_)
template <typename T>
void Tensor<T>::init_cudnn_descriptor() {
    int n = 1, c = 1, h = 1, w = 1;

    cudnnCreateTensorDescriptor(&desc);

    if (_shape.size() == 4) {
        n = _shape[0];
        c = _shape[1];
        h = _shape[2];
        w = _shape[3];
    } else if (_shape.size() == 3) {
        n = _shape[0];
        c = _shape[1];
        h = _shape[2];
    } else if (_shape.size() == 2) {
        n = _shape[0];
        c = _shape[1];
    } else if (_shape.size() == 1) {
        n = _shape[0];
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
