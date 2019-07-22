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

Tensor::Tensor() : Tensor({0}) {}

Tensor::Tensor(const std::vector<index_t>& shape, DataType dtype, tensor_filler_t filler, memory_t mem_type,
               device_t device_id)
    : shape_(shape), dtype_(dtype), mem_type_(mem_type), device_id_(device_id) {
    MAGMADNN_ASSERT(shape.size() != 0, "Invalid tensor shape.\n");

    this->strides_.resize(shape_.size());
    size_t tmp_stride = 1;
    for (int i = ((int) shape_.size()) - 1; i >= 0; i--) {
        strides_[i] = tmp_stride;
        tmp_stride *= shape_[i];
    }

    // calculate size
    this->size_ = 1;
    for (const auto& axis : shape_) {
        this->size_ *= axis;
    }

    this->memory_manager_ptr = std::make_shared<MemoryManager>(size_, dtype_, mem_type_, device_id_);

    internal::fill_memory(*memory_manager_ptr, filler);

#if defined(_HAS_CUDA_)
    this->init_cudnn_descriptor();
#endif
}

Tensor::~Tensor() {
#if defined(_HAS_CUDA_)
    this->free_cudnn_descriptor();
#endif
}

Tensor::Tensor(const Tensor& t) {
    this->shape_ = t.shape_;
    this->strides_ = t.strides_;
    this->size_ = t.size_;
    this->dtype_ = t.dtype_;

    this->mem_type_ = t.mem_type_;
    this->device_id_ = t.device_id_;

    /* shared pointer -- reference count the memory manager */
    /* TODO -- copy from memory */
    this->memory_manager_ptr = t.memory_manager_ptr;

/* TODO -- replace cudnn_descriptor with reference counting */
#if defined(_HAS_CUDA_)
    this->init_cudnn_descriptor();
#endif
}

Tensor::Tensor(Tensor&& t) {
    swap(*this, t);

    /* TODO -- replace cudnn_descriptor with reference counting */
#if defined(_HAS_CUDA_)
    this->init_cudnn_descriptor();
#endif
}

void swap(Tensor& left, Tensor& right) {
    using std::swap;

    swap(left.shape_, right.shape_);
    swap(left.strides_, right.strides_);
    swap(left.size_, right.size_);
    swap(left.dtype_, right.dtype_);
    swap(left.mem_type_, right.mem_type_);
    swap(left.device_id_, right.device_id_);

    swap(left.memory_manager_ptr, right.memory_manager_ptr);
}

Tensor& Tensor::operator=(Tensor t) {
    /* by passing by value we allow the compiler to elide the copying of an rvalue */
    swap(*this, t);

    /* TODO -- replace cudnn_descriptor with reference counting */
#if defined(_HAS_CUDA_)
    this->init_cudnn_descriptor();
#endif
    return *this;
}

magmadnn_error_t Tensor::copy_from(const Tensor& src, index_t begin_idx, size_t size) {
    MAGMADNN_ASSERT(this->size_ >= size, "Copy tensor too large.");
    MAGMADNN_ASSERT(src.size_ >= (begin_idx + size), "Invalid indices.");

    return this->memory_manager_ptr->copy_from(*src.get_memory_manager(), begin_idx, size);
}

magmadnn_error_t Tensor::copy_from(const Tensor& src) { return copy_from(src, 0, src.size()); }

magmadnn_error_t Tensor::copy_from(const Tensor& src, const std::vector<index_t>& dims) {
    for (unsigned int i = 0; i < dims.size(); i++) {
        assert(dims[i] != 0);
        assert(dims[i] <= src.shape(i));
    }
    magmadnn_error_t m = 0;
    std::vector<index_t> target_shape(dims.size(), 0);
    int curr_pos = target_shape.size() - 1;
    while (curr_pos >= 0) {
        curr_pos = target_shape.size() - 1;

        FOR_ALL_DTYPES(dtype_, Dtype, { this->set<Dtype>(target_shape, src.get<Dtype>(target_shape)); });

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
T Tensor::get(const std::vector<index_t>& idx) const {
    return memory_manager_ptr->get<T>(get_flattened_index(idx));
}
#define COMPILE_TENSOR_GET(type) template type Tensor::get(const std::vector<index_t>&) const;
CALL_FOR_ALL_TYPES(COMPILE_TENSOR_GET)
#undef COMPILE_TENSOR_GET

template <typename T>
T Tensor::get(index_t flattened_idx) const {
    return memory_manager_ptr->get<T>(flattened_idx);
}
#define COMPILE_TENSOR_GET(type) template type Tensor::get(index_t) const;
CALL_FOR_ALL_TYPES(COMPILE_TENSOR_GET)
#undef COMPILE_TENSOR_GET

template <typename T>
void Tensor::set(const std::vector<index_t>& idx, T val) {
    memory_manager_ptr->set<T>(get_flattened_index(idx), val);
}
#define COMPILE_TENSOR_SET(type) template void Tensor::set(const std::vector<index_t>&, type);
CALL_FOR_ALL_TYPES(COMPILE_TENSOR_SET)
#undef COMPILE_TENSOR_SET

template <typename T>
void Tensor::set(index_t flattened_idx, T val) {
    memory_manager_ptr->set<T>(flattened_idx, val);
}
#define COMPILE_TENSOR_SET(type) template void Tensor::set(index_t, type);
CALL_FOR_ALL_TYPES(COMPILE_TENSOR_SET)
#undef COMPILE_TENSOR_SET

inline index_t Tensor::get_flattened_index(const std::vector<index_t>& idx) const {
    index_t flattened_idx = 0;

    for (index_t i = 0; i < idx.size(); i++) {
        flattened_idx += idx[i] * strides_[i];
    }

    return flattened_idx;
}

/** @deprecated */
index_t Tensor::get_flattened_index_old(const std::vector<index_t>& idx) const {
    index_t jump_size = 1;  // the total amount to jump to get to next axis
    index_t flattened_idx = 0;

    for (int i = ((int) idx.size()) - 1; i >= 0; i--) {
        flattened_idx += idx[i] * jump_size;
        jump_size *= shape_[i];
    }
    return flattened_idx;
}

void Tensor::reshape(const std::vector<index_t>& dims) {
    size_t dims_size = 1;
    for (unsigned int i = 0; i < dims.size(); i++) dims_size *= dims[i];
    assert(size_ == dims_size);
    shape_ = dims;

    /* update strides */
    strides_.resize(dims.size());
    index_t tmp_stride = 1;
    for (int i = ((int) shape_.size()) - 1; i >= 0; i--) {
        strides_[i] = tmp_stride;
        tmp_stride *= shape_[i];
    }

#if defined(_HAS_CUDA_)
    /* update cudnn descriptor if on GPU */
    int n = 1, c = 1, h = 1, w = 1;
    if (shape_.size() == 4) {
        n = shape_[0];
        c = shape_[1];
        h = shape_[2];
        w = shape_[3];
    } else if (shape_.size() == 3) {
        n = shape_[0];
        c = shape_[1];
        h = shape_[2];
    } else if (shape_.size() == 2) {
        n = shape_[0];
        c = shape_[1];
    } else if (shape_.size() == 1) {
        n = shape_[0];
    } else {
        fprintf(stderr, "Cannot create tensor descriptor for tensor of this shape\n");
    }
    cudnnErrchk(cudnnSetTensor4dDescriptor(this->desc_, CUDNN_TENSOR_NCHW,
                                           ::magmadnn::internal::get_cudnn_data_type(dtype_), n, c, h, w));
#endif
}

void Tensor::squeeze() {
    std::vector<index_t> new_shape;
    for (index_t i = 0; i < shape_.size(); i++) {
        if (shape_[i] > 1) new_shape.push_back(shape_[i]);
    }
    if (new_shape.size() == 0) new_shape.push_back(1);
    shape_ = new_shape;
}

void Tensor::unsqueeze(index_t dim) {
    assert(dim <= shape_.size());
    shape_.insert(shape_.begin() + dim, 1, 1);
}

#if defined(_HAS_CUDA_)
void Tensor::init_cudnn_descriptor() {
    int n = 1, c = 1, h = 1, w = 1;

    cudnnCreateTensorDescriptor(&desc_);

    if (shape_.size() == 4) {
        n = shape_[0];
        c = shape_[1];
        h = shape_[2];
        w = shape_[3];
    } else if (shape_.size() == 3) {
        n = shape_[0];
        c = shape_[1];
        h = shape_[2];
    } else if (shape_.size() == 2) {
        n = shape_[0];
        c = shape_[1];
    } else if (shape_.size() == 1) {
        n = shape_[0];
    } else {
        fprintf(stderr, "Cannot create tensor descriptor for tensor of this shape\n");
    }

    cudnnSetTensor4dDescriptor(desc_, CUDNN_TENSOR_NCHW, internal::get_cudnn_data_type(this->dtype_), n, c, h, w);
}

void Tensor::free_cudnn_descriptor() { cudnnDestroyTensorDescriptor(this->desc_); }
#endif

}  // namespace magmadnn
