/**
 * @file memorymanager.cpp
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-02-07
 *
 * @copyright Copyright (c) 2019
 */
#include "memory/memorymanager.h"

namespace magmadnn {

MemoryManager::MemoryManager(size_t size, DataType dtype, memory_t mem_type, device_t device_id)
    : dtype_(dtype), mem_type_(mem_type), size_(size) {
    set_device(device_id);

    // initialize based on the chosen memory type
    switch (mem_type) {
        case HOST:
            init_host();
            break;
#if defined(_HAS_CUDA_)
        case DEVICE:
            init_device();
            break;
        case MANAGED:
            init_managed();
            break;
        case CUDA_MANAGED:
            init_cuda_managed();
            break;
#endif
        default:
            fprintf(stderr, "Invalid memory type.\n");
    }
}

void MemoryManager::init_host() { host_ptr_ = (void*) std::malloc(size_ * getDataTypeSize(dtype_)); }

#if defined(_HAS_CUDA_)
void MemoryManager::init_device() { cudaErrchk(cudaMalloc((void**) &device_ptr_, size_ * getDataTypeSize(dtype_))); }

void MemoryManager::init_managed() {
    host_ptr_ = (void*) std::malloc(size_ * getDataTypeSize(dtype_));
    cudaErrchk(cudaMalloc((void**) &device_ptr_, size_ * getDataTypeSize(dtype_)));
}

void MemoryManager::init_cuda_managed() {
    cudaErrchk(cudaMallocManaged((void**) &cuda_managed_ptr_, size_ * getDataTypeSize(dtype_)));
}
#endif

/* TODO -- revisit memory manager copy, move, and assignment */
/*
template <typename T>
MemoryManager<T>::MemoryManager(const MemoryManager& that)
    : mem_type(that.mem_type), device_id(that.device_id), size(that.size) {
    //  TODO -- size issue here

    this->copy_from(that);
}
*/

/*
template <typename T>
MemoryManager<T>::MemoryManager(MemoryManager<T>&& that) {
    this->mem_type = that.mem_type;
    this->device_id = that.device_id;
    this->size = that.size;

    //  TODO -- size issue here

    this->copy_from(that);
}

template <typename T>
MemoryManager<T>& MemoryManager<T>::operator=(MemoryManager<T> that) {
    this->mem_type = that.mem_type;
    this->device_id = that.device_id;
    this->size = that.size;

    //  TODO -- size issue here

    this->copy_from(that);

    return *this;
}
*/

MemoryManager::~MemoryManager() {
    switch (mem_type_) {
        case HOST:
            std::free(host_ptr_);
            break;
#if defined(_HAS_CUDA_)
        case DEVICE:
            cudaErrchk(cudaFree(device_ptr_));
            break;
        case MANAGED:
            std::free(host_ptr_);
            break;
            cudaErrchk(cudaFree(device_ptr_));
            break;
        case CUDA_MANAGED:
            cudaErrchk(cudaFree(cuda_managed_ptr_));
            break;
#endif
    }
}

magmadnn_error_t MemoryManager::copy_from(const MemoryManager& src, index_t begin_idx, size_t copy_size) {
    assert(this->size_ >= copy_size);
    assert(src.size_ >= (begin_idx + copy_size));
    assert(src.dtype_ == dtype_);

    if (copy_size == 0) return (magmadnn_error_t) 0;

    FOR_ALL_DTYPES(dtype_, Dtype, {
        switch (src.mem_type_) {
            case HOST:
                return copy_from_host(src.get_host_ptr<Dtype>(), begin_idx, copy_size);
#if defined(_HAS_CUDA_)
            case DEVICE:
                return copy_from_device(src.get_device_ptr<Dtype>(), begin_idx, copy_size);
            case MANAGED:
                return copy_from_managed(src.get_host_ptr<Dtype>(), src.get_device_ptr<Dtype>(), begin_idx, copy_size);
            case CUDA_MANAGED:
                return copy_from_cudamanaged(src.get_cuda_managed_ptr<Dtype>(), begin_idx, copy_size);
#endif
            default:
                return (magmadnn_error_t) 1;
        }
    });

    return (magmadnn_error_t) 1;
}

magmadnn_error_t MemoryManager::copy_from(const MemoryManager& src) { return copy_from(src, 0, size_); }

magmadnn_error_t MemoryManager::copy_from(const MemoryManager& src, size_t copy_size) {
    return copy_from(src, 0, copy_size);
}

template <typename T>
magmadnn_error_t MemoryManager::copy_from_host(const T* src, index_t begin_idx, size_t copy_size) {
    switch (mem_type_) {
        case HOST:
            // host --> host
            std::copy(src + begin_idx, (src + begin_idx) + copy_size, reinterpret_cast<T*>(host_ptr_));
            return (magmadnn_error_t) 0;
#if defined(_HAS_CUDA_)
        case DEVICE:
            // host --> device
            cudaErrchk(cudaMemcpy(device_ptr_, src + begin_idx, copy_size * sizeof(T), cudaMemcpyHostToDevice));
            return (magmadnn_error_t) 0;
        case MANAGED:
            // host --> managed
            std::copy(src + begin_idx, (src + begin_idx) + copy_size, reinterpret_cast<T*>(host_ptr_));
            sync(false);
            return (magmadnn_error_t) 0;
        case CUDA_MANAGED:
            // host --> cmanaged
            std::copy(src + begin_idx, (src + begin_idx) + copy_size, reinterpret_cast<T*>(cuda_managed_ptr_));
            sync(false);
            return (magmadnn_error_t) 0;
#endif
    }

    return (magmadnn_error_t) 1;
}

#if defined(_HAS_CUDA_)
template <typename T>
magmadnn_error_t MemoryManager::copy_from_device(const T* src, index_t begin_idx, size_t copy_size) {
    magmadnn_error_t err = (magmadnn_error_t) 0;

    switch (mem_type_) {
        case HOST:
            // device --> host
            cudaErrchk(cudaMemcpy(host_ptr_, src + begin_idx, copy_size * sizeof(T), cudaMemcpyDeviceToHost));
            break;
        case DEVICE:
            // device --> device
            cudaErrchk(cudaMemcpy(device_ptr_, src + begin_idx, copy_size * sizeof(T), cudaMemcpyDeviceToDevice));
            break;
        case MANAGED:
            // device --> managed
            cudaErrchk(cudaMemcpy(device_ptr_, src + begin_idx, copy_size * sizeof(T), cudaMemcpyDeviceToDevice));
            sync(true);
            return err;
        case CUDA_MANAGED:
            // device --> cmanaged
            cudaErrchk(cudaMemcpy(cuda_managed_ptr_, src + begin_idx, copy_size * sizeof(T), cudaMemcpyDeviceToDevice));
            sync(true);
            return err;
    }

    return err;
}

template <typename T>
magmadnn_error_t MemoryManager::copy_from_managed(const T* host_src, const T* device_src, index_t begin_idx,
                                                  size_t copy_size) {
    switch (mem_type_) {
        case HOST:
            // managed --> host
            std::copy(host_src + begin_idx, (host_src + begin_idx) + copy_size, reinterpret_cast<T*>(host_ptr_));
            return (magmadnn_error_t) 0;
        case DEVICE:
            // managed --> device
            cudaErrchk(
                cudaMemcpy(device_ptr_, device_src + begin_idx, copy_size * sizeof(T), cudaMemcpyDeviceToDevice));
            return (magmadnn_error_t) 0;
        case MANAGED:
            // managed --> managed
            std::copy(host_src + begin_idx, (host_src + begin_idx) + copy_size, reinterpret_cast<T*>(host_ptr_));
            sync(false);
            return (magmadnn_error_t) 0;
        case CUDA_MANAGED:
            // managed --> cmanaged
            std::copy(host_src + begin_idx, (host_src + begin_idx) + copy_size,
                      reinterpret_cast<T*>(cuda_managed_ptr_));
            sync(false);
            return (magmadnn_error_t) 0;
    }

    return (magmadnn_error_t) 1;
}

template <typename T>
magmadnn_error_t MemoryManager::copy_from_cudamanaged(const T* src, index_t begin_idx, size_t copy_size) {
    switch (mem_type_) {
        case HOST:
            // cmanaged --> host
            std::copy(src + begin_idx, (src + begin_idx) + copy_size, reinterpret_cast<T*>(host_ptr_));
            return (magmadnn_error_t) 0;
        case DEVICE:
            // cmanaged --> device
            cudaErrchk(cudaMemcpy(device_ptr_, src + begin_idx, copy_size * sizeof(T), cudaMemcpyDeviceToDevice));
            return (magmadnn_error_t) 0;
        case MANAGED:
            // cmanaged --> managed
            std::copy(src + begin_idx, (src + begin_idx) + copy_size, reinterpret_cast<T*>(host_ptr_));
            sync(false);
            return (magmadnn_error_t) 0;
        case CUDA_MANAGED:
            std::copy(src + begin_idx, (src + begin_idx) + copy_size, reinterpret_cast<T*>(cuda_managed_ptr_));
            sync(false);
            return (magmadnn_error_t) 0;
    }

    return (magmadnn_error_t) 1;
}
#endif

magmadnn_error_t MemoryManager::sync(bool gpu_was_modified) const {
#if defined(_HAS_CUDA_)
    cudaError_t err = (cudaError_t) 0;

    if (mem_type_ == CUDA_MANAGED) {
        cudaErrchk(cudaDeviceSynchronize());
    } else if (mem_type_ == MANAGED) {
        if (gpu_was_modified) {
            cudaErrchk(cudaMemcpy(host_ptr_, device_ptr_, size_ * getDataTypeSize(dtype_), cudaMemcpyDeviceToHost));
        } else {
            cudaErrchk(cudaMemcpy(device_ptr_, host_ptr_, size_ * getDataTypeSize(dtype_), cudaMemcpyHostToDevice));
        }
    }
    return (magmadnn_error_t) err;
#else

    // nothing to sync, no error
    return (magmadnn_error_t) 0;

#endif
}

template <typename T>
T MemoryManager::get(index_t idx) const {
    assert(idx < size_);

    switch (mem_type_) {
        case HOST:
            return reinterpret_cast<T*>(host_ptr_)[idx];
#if defined(_HAS_CUDA_)
        case DEVICE:
            // cudaErrchk(cudaSetDevice(device_id));
            return internal::get_device_array_element(reinterpret_cast<T*>(device_ptr_), idx);
        case MANAGED:
            return reinterpret_cast<T*>(host_ptr_)[idx];
        case CUDA_MANAGED:
            return reinterpret_cast<T*>(cuda_managed_ptr_)[idx];
#endif
    }
    return (T) 0;
}
#define COMPILE_MEMORYMANAGER_GET(type) template type MemoryManager::get<type>(index_t idx) const;
CALL_FOR_ALL_TYPES(COMPILE_MEMORYMANAGER_GET)
#undef COMPILE_MEMORYMANAGER_GET

template <typename T>
void MemoryManager::set(index_t idx, T val) {
    assert(idx < size_);

    // note: don't sync on managed type memories
    switch (mem_type_) {
        case HOST:
            reinterpret_cast<T*>(host_ptr_)[idx] = val;
            break;
#if defined(_HAS_CUDA_)
        case DEVICE:
            // cudaErrchk(cudaSetDevice(device_id));
            internal::set_device_array_element(reinterpret_cast<T*>(device_ptr_), idx, val);
            break;
        case MANAGED:
            reinterpret_cast<T*>(host_ptr_)[idx] = val;
            // cudaErrchk(cudaSetDevice(device_id));
            internal::set_device_array_element(reinterpret_cast<T*>(device_ptr_), idx, val);
            break;
        case CUDA_MANAGED:
            reinterpret_cast<T*>(cuda_managed_ptr_)[idx] = val;
            break;
#endif
    }
}
#define COMPILE_MEMORYMANAGER_SET(type) template void MemoryManager::set<type>(index_t idx, type val);
CALL_FOR_ALL_TYPES(COMPILE_MEMORYMANAGER_SET)
#undef COMPILE_MEMORYMANAGER_SET

magmadnn_error_t MemoryManager::set_device(device_t device_id) {
#if defined(_HAS_CUDA_)
    int n_devices = 0;
    cudaErrchk(cudaGetDeviceCount(&n_devices));
    if ((int) device_id >= n_devices) {
        fprintf(stderr, "invalid device id\n");
        return (magmadnn_error_t) 1;
    }
#endif

    this->device_id_ = device_id;
    return (magmadnn_error_t) 0;
}

template <typename T>
T* MemoryManager::get_host_ptr() {
    return reinterpret_cast<T*>(host_ptr_);
}
#define COMPILE_GETPTR(type) template type* MemoryManager::get_host_ptr();
CALL_FOR_ALL_TYPES(COMPILE_GETPTR)
#undef COMPILE_GETPTR

template <typename T>
const T* MemoryManager::get_host_ptr() const {
    return reinterpret_cast<const T*>(host_ptr_);
}
#define COMPILE_GETPTR(type) template const type* MemoryManager::get_host_ptr() const;
CALL_FOR_ALL_TYPES(COMPILE_GETPTR)
#undef COMPILE_GETPTR

#if defined(_HAS_CUDA_)
template <typename T>
T* MemoryManager::get_device_ptr() {
    return reinterpret_cast<T*>(device_ptr_);
}
#define COMPILE_GETPTR(type) template type* MemoryManager::get_device_ptr();
CALL_FOR_ALL_TYPES(COMPILE_GETPTR)
#undef COMPILE_GETPTR

template <typename T>
const T* MemoryManager::get_device_ptr() const {
    return reinterpret_cast<const T*>(device_ptr_);
}
#define COMPILE_GETPTR(type) template const type* MemoryManager::get_device_ptr() const;
CALL_FOR_ALL_TYPES(COMPILE_GETPTR)
#undef COMPILE_GETPTR

template <typename T>
T* MemoryManager::get_cuda_managed_ptr() {
    return reinterpret_cast<T*>(cuda_managed_ptr_);
}
#define COMPILE_GETPTR(type) template type* MemoryManager::get_cuda_managed_ptr();
CALL_FOR_ALL_TYPES(COMPILE_GETPTR)
#undef COMPILE_GETPTR

template <typename T>
const T* MemoryManager::get_cuda_managed_ptr() const {
    return reinterpret_cast<const T*>(cuda_managed_ptr_);
}
#define COMPILE_GETPTR(type) template const type* MemoryManager::get_cuda_managed_ptr() const;
CALL_FOR_ALL_TYPES(COMPILE_GETPTR)
#undef COMPILE_GETPTR
#endif

template <typename T>
T* MemoryManager::get_ptr() {
    switch (mem_type_) {
        case HOST:
            return get_host_ptr<T>();
#if defined(_HAS_CUDA_)
        case DEVICE:
            return get_device_ptr<T>();
        case MANAGED:
            // returns device by default for managed
            return get_device_ptr<T>();
        case CUDA_MANAGED:
            return get_cuda_managed_ptr<T>();
#endif
        default:
            return nullptr;
    }
}
#define COMPILE_GETPTR(type) template type* MemoryManager::get_ptr();
CALL_FOR_ALL_TYPES(COMPILE_GETPTR)
#undef COMPILE_GETPTR

template <typename T>
const T* MemoryManager::get_ptr() const {
    switch (mem_type_) {
        case HOST:
            return get_host_ptr<T>();
#if defined(_HAS_CUDA_)
        case DEVICE:
            return get_device_ptr<T>();
        case MANAGED:
            // returns device by default for managed
            return get_device_ptr<T>();
        case CUDA_MANAGED:
            return get_cuda_managed_ptr<T>();
#endif
        default:
            return nullptr;
    }
}
#define COMPILE_GETPTR(type) template const type* MemoryManager::get_ptr() const;
CALL_FOR_ALL_TYPES(COMPILE_GETPTR)
#undef COMPILE_GETPTR

}  // namespace magmadnn
