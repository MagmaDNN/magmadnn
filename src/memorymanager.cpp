/**
 * @file memorymanager.cpp
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-02-07
 * 
 * @copyright Copyright (c) 2019
 */
#include "memorymanager.h"

namespace skepsi {

template <typename T>
memorymanager<T>::memorymanager(unsigned int size, memory_t mem_type, device_t device_id) : 
    mem_type(mem_type), device_id(device_id), size(size) {

        // initialize based on the chosen memory type
        switch (mem_type) {
            case HOST:
                init_host(); break;
            #ifdef _HAS_CUDA_
            case DEVICE:
                init_device(); break;
            case MANAGED:
                init_managed(); break;
            case CUDA_MANAGED:
                init_cuda_managed(); break;
            #endif
            default:
                fprintf(stderr, "Invalid memory type.\n");
        }
}

template <typename T>
void memorymanager<T>::init_host() {
    host_ptr = (T *) std::malloc(size * sizeof(T));
}

#ifdef _CUDA_HOST_
template <typename T>
void memorymanager<T>::init_device() {
    cudaMalloc(&device_ptr, size * sizeof(T));
}

template <typename T>
void memorymanager<T>::init_managed() {
    host_ptr = (T *) std::malloc(size * sizeof(T));
    cudaMalloc(&device_ptr, size * sizeof(T));
}

template <typename T>
void memorymanager<T>::init_cuda_managed() {
    cudaMallocManaged(&cuda_managed_ptr, size * sizeof(T));
}
#endif

template <typename T>
memorymanager<T>::~memorymanager<T>() {
    switch (mem_type) {
        case HOST:
            std::free(host_ptr); break;
        #ifdef _HAS_CUDA_
        case DEVICE:
            cudaFree(device_ptr); break;
        case MANAGED:
            std::free(host_ptr); break;
            cudaFree(device_ptr); break;
        case CUDA_MANAGED:
            cudaFree(device_ptr); break;
        #endif
    }
}

template <typename T>
skepsi_error_t memorymanager<T>::copy_from(const memorymanager<T>& src, unsigned int begin_idx, unsigned int copy_size) {
    assert( this->size == src.size );
    
    if (copy_size == 0) return (skepsi_error_t) 0;

    if (src.mem_type == HOST) {
        return copy_from_host(src.host_ptr, begin_idx, copy_size);
    }
    #ifdef _HAS_CUDA_
    else if (src.mem_type == DEVICE) {
        return copy_from_device(src.device_ptr, begin_idx, copy_size);
    } else if (src.mem_type == MANAGED) {
        return copy_from_managed(src.host_ptr, src.device_ptr, begin_idx, copy_size);
    } else if (src.mem_type == CUDAMANAGED) {
        return copy_from_cudamanaged(src.managed_ptr, begin_idx, copy_size);
    }
    #endif
    
    return (skepsi_error_t) 1;
}

template <typename T>
skepsi_error_t memorymanager<T>::copy_from(const memorymanager<T>& src) {
    return copy_from(src, 0, size);
}

template <typename T>
skepsi_error_t memorymanager<T>::copy_from(const memorymanager<T>& src, unsigned int copy_size) {
    return copy_from(src, 0, copy_size);
}

template <typename T>
skepsi_error_t memorymanager<T>::copy_from_host(T *src, unsigned int begin_idx, unsigned int copy_size) {

    switch (mem_type) {
        case HOST:
            // host --> host
			std::copy(src + begin_idx, (src+begin_idx) + copy_size, host_ptr);
            return (skepsi_error_t) 0;
        #ifdef _HAS_CUDA_
        case DEVICE:
            // host --> device
            return (skepsi_error_t) cudaMemcpy(device_ptr, src+begin_idx, copy_size*sizeof(T), cudaMemcpyHostToDevice);
        case MANAGED:
            // host --> managed
			std::copy(src+begin_idx, (src+begin_idx) + copy_size, host_ptr);
            sync(false);
            return (skepsi_error_t) 0;
        case CUDA_MANAGED:
            // host --> cmanaged
            std::copy(src+begin_idx, (src+begin_idx) + copy_size, managed_ptr);
            sync(false);
            return (skepsi_error_t) 0;
        #endif
    }

    return (skepsi_error_t) 1;
}

#ifdef _HAS_CUDA_
template <typename T>
skepsi_error_t memorymanager<T>::copy_from_device(T *src, unsigned int begin_idx, unsigned int copy_size) {

    switch (mem_type) {
        case HOST:
            // device --> host
            return cudaMemcpy(host_ptr, src+begin_idx, copy_size*sizeof(T), cudaMemcpyDeviceToHost);
        case DEVICE:
            // device --> device
            return cudaMemcpy(device_ptr, src+begin_idx, copy_size*sizeof(T), cudaMemcpyDeviceToDevice);
        case MANAGED:
            // device --> managed
            skepsi_error_t err = cudaMemcpy(device_ptr, src+begin_idx, copy_size*sizeof(T), cudaMemcpyDeviceToDevice);
            sync(true);
            return err;
        case CUDA_MANAGED:
            // device --> cmanaged
            skepsi_error_t err = cudaMemcpy(managed_ptr, src+begin_idx, copy_size*sizeof(T), cudaMemcpyDeviceToDevice);
            sync(true);
            return err;
    }

    return (skepsi_error_t) 1;
}

template <typename T>
skepsi_error_t memorymanager<T>::copy_from_managed(T *host_src, T *device_src, unsigned int begin_idx, unsigned int copy_size) {

    switch (mem_type) {
        case HOST:
            // managed --> host
			std::copy(host_src+begin_idx, (host_src+begin_idx) + copy_size, host_ptr);
        case DEVICE:
            // managed --> device
            return cudaMemcpy(device_ptr, device_src+begin_idx, copy_size*sizeof(T), cudaMemcpyDeviceToDevice);
        case MANAGED:
            // managed --> managed
            std::copy(host_src+begin_idx, (host_src+begin_idx) + copy_size, host_ptr);
            sync(false);
            return (skepsi_error_t) 0;
        case CUDA_MANAGED:
            // managed --> cmanaged
            std::copy(host_src+begin_idx, (host_src+begin_idx) + copy_size, managed_ptr);
            sync(false);
            return (skepsi_error_t) 0;
    }

    return (skepsi_error_t) 1;
}

template <typename T>
skepsi_error_t memorymanager<T>::copy_from_cudamanaged(T *src, unsigned int begin_idx, unsigned int copy_size) {

    switch (mem_type) {
        case HOST:
            // cmanaged --> host
			std::copy(src+begin_idx, (src+begin_idx) + copy_size, host_ptr);
            return (skepsi_error_t) 0;
        case DEVICE:
            // cmanaged --> device
            return cudaMemcpy(device_ptr, src+begin_idx, copy_size*sizeof(T), cudaMemcpyDeviceToDevice);
        case MANAGED:
            // cmanaged --> managed
            std::copy(src+begin_idx, (src+begin_idx) + copy_size, host_ptr);
            sync(false);
            return (skepsi_error_t) 0;
        case CUDA_MANAGED:
            std::copy(src+begin_idx, (src+begin_idx) + copy_size, managed_ptr);
            sync(false);
            return (skepsi_error_t) 0;
    }

    return (skepsi_error_t) 1;
}
#endif

template <typename T>
skepsi_error_t memorymanager<T>::sync(bool gpu_was_modified) {
    #ifdef _HAS_CUDA_
        cudaError_t err = (cudaError_t) 0;

        if (mem_type == CUDA_MANAGED) {
            err = cudaDeviceSynchronize();
        } else if (mem_type == MANAGED) {
            if (gpu_was_modified) {
                err = cudaMemcpy(host_ptr, device_ptr, size*sizeof(T), cudaMemcpyDeviceToHost);
            } else {
                err = cudaMemcpy(device_ptr, host_ptr, size*sizeof(T), cudaMemcpyHostToDevice);
            }
        }
        return (skepsi_error_t) err;
    #else

        // nothing to sync, no error
        return (skepsi_error_t) 0;

    #endif
}

template <typename T>
T memorymanager<T>::get(unsigned int idx) {
    assert( idx < size );

    switch (mem_type) {
        case HOST:
            return host_ptr[idx];
        #ifdef _HAS_CUDA_
        case DEVICE:
            // TODO
            return (T) 0;
        case MANAGED:
            return host_ptr[idx];
        case CUDA_MANAGED:
            return managed_ptr[idx];
        #endif
    }
    return (T) 0;
}

template <typename T>
void memorymanager<T>::set(unsigned int idx, T val) {
    assert( idx < size );

    // note: don't sync on managed type memories
    switch (mem_type) {
        case HOST:
            host_ptr[idx] = val; break;
        #ifdef _HAS_CUDA_
        case DEVICE:
            // TODO
            break;
        case MANAGED:
            host_ptr[idx] = val;
            // TODO: set device pointer
            break;
        case CUDA_MANAGED:
            managed_ptr[idx] = val; break;
        #endif
    }
}

template <typename T>
T* memorymanager<T>::get_host_ptr() {
    return host_ptr;
}

#ifdef _HAS_CUDA_
template <typename T>
T* memorymanager<T>::get_device_ptr() {
    return device_ptr;
}

template <typename T>
T* memorymanager<T>::get_cuda_managed_ptr() {
    return cuda_managed_ptr;
}
#endif

template <typename T>
T* memorymanager<T>::get_ptr() {
    switch (mem_type) {
        case HOST:
            return get_host_ptr();
        #ifdef _HAS_CUDA_
        case DEVICE:
            return get_device_ptr();
        case MANAGED:
            fprintf(stderr, "memorymanager::get_ptr() is not defined for MANAGED type memory\n");
            return (T*) NULL;
        case CUDA_MANAGED:
            return get_cuda_managed_ptr();
        #endif
		default:
			return (T*) NULL;
    }
}




/* COMPILE FOR INT, FLOAT, AND DOUBLE */
template class memorymanager<int>;
template class memorymanager<float>;
template class memorymanager<double>;

} // namespace skepsi
