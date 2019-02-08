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

    /* assert malloc was successful */
    assert( host_ptr != NULL );
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
            // TODO
            break;
        case MANAGED:
            std::free(host_ptr); break;
        case CUDA_MANAGED:
            // TODO
            break;
        #endif
        default:
            break;
    }
}

template <typename T>
skepsi_error_t memorymanager<T>::copy_from(const memorymanager<T>& src) {
    // stay within same memory type and size for now
    assert( this->mem_type == src.mem_type );
    assert( this->size == src.size );

    switch (mem_type) {
        case HOST:
			std::copy(src.host_ptr, src.host_ptr + src.size, this->host_ptr); break;
        #ifdef _HAS_CUDA_
        case DEVICE:
            // TODO
            fprintf(stderr, "not implemented\n"); break;
        case MANAGED:
            // assumes src is synced
			std::copy(src.host_ptr, src.host_ptr + src.size, this->host_ptr);
            // TODO alert cpu change
            this->sync();
            break;
        case CUDA_MANAGED:
            // TODO
            fprintf(stderr, "not yet implemented\n"); break;
        #endif
    }

    return 0;
}

template <typename T>
skepsi_error_t memorymanager<T>::copy_from_host(T *src) {

    switch (mem_type) {
        case HOST:
			std::copy(src, src + this->size, this->host_ptr);
        #ifdef _HAS_CUDA_
        case DEVICE:
            // TODO
            fprintf(stderr, "not yet implemented\n"); break;
        case MANAGED:
            // TODO
			std::copy(src, src + this->size, this->host_ptr);
            // TODO alert cpu modified
            this->sync(); break;
        case CUDA_MANAGED:
            // TODO
            fprintf(stderr, "not yet implemented\n"); break;
        #endif
    }

    return 0;
}

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
            // TODO possibly sync here
            return host_ptr[idx];
        case CUDA_MANAGED:
            // TODO
            return (T) 0;
        #endif
        default:
            return (T) 0;
    }
}

template <typename T>
void memorymanager<T>::set(unsigned int idx, T val) {
    assert( idx < size );

    switch (mem_type) {
        case HOST:
            host_ptr[idx] = val; break;
        #ifdef _HAS_CUDA_
        case DEVICE:
            // TODO
            break;
        case MANAGED:
            // TODO possibly sync here
            host_ptr[idx] = val; break;
        case CUDA_MANAGED:
            // TODO
            break;
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
