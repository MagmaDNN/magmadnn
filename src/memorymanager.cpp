/**
 * @file memorymanager.cpp
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-02-07
 * 
 * @copyright Copyright (c) 2019
 */
#include "memorymanager.h"



template <typename T>
memorymanager<T>::memorymanager(unsigned int size, memory_t mem_type, device_t device_id) : 
    mem_type(mem_type), device_id(device_id), size(size) {

        // initialize based on the chosen memory type
        switch (mem_type) {
            case DEVICE:
                init_device(); break;
            case HOST:
                init_host(); break;
            case MANAGED:
                init_managed(); break;
            case CUDA_MANAGED:
                init_cuda_managed(); break;
            default:
                fprintf(stderr, "Invalid memory type.\n");
        }
}

template <typename T>
void memorymanager<T>::init_device() {

    fprintf(stderr, "Memory type not yet implemented.\n");
}

template <typename T>
void memorymanager<T>::init_host() {
    host_ptr = (T *) std::malloc(size * sizeof(T));

    /* assert malloc was successful */
    assert( host_ptr != NULL );
}

template <typename T>
void memorymanager<T>::init_managed() {
    fprintf(stderr, "Memory type not yet implemented.\n");
}

template <typename T>
void memorymanager<T>::init_cuda_managed() {
    fprintf(stderr, "Memory type not yet implemented.\n");
}

template <typename T>
memorymanager<T>::~memorymanager<T>() {
    switch (mem_type) {
        case DEVICE:
            // TODO
            break;
        case HOST:
            std::free(host_ptr); break;
        case MANAGED:
            std::free(host_ptr); break;
        case CUDA_MANAGED:
            // TODO
            break;
        default:
            break;
    }
}

template <typename T>
error_t memorymanager<T>::copy_from(const memorymanager<T>& src) {
    // stay within same memory type and size for now
    assert( this->mem_type == src.mem_type );
    assert( this->size == src.size );

    switch (mem_type) {
        case DEVICE:
            // TODO
            fprintf(stderr, "not implemented\n");
            break;
        case HOST:
			std::copy(src.host_ptr, src.host_ptr + src.size, this->host_ptr);
            break;
        case MANAGED:
            // assumes src is synced
			std::copy(src.host_ptr, src.host_ptr + src.size, this->host_ptr);
            // TODO alert cpu change
            this->sync();
            break;
        case CUDA_MANAGED:
            // TODO
            fprintf(stderr, "not yet implemented\n");
            break;
    }

    return 0;
}

template <typename T>
error_t memorymanager<T>::copy_from_host(T *src) {

    switch (mem_type) {
        case DEVICE:
            // TODO
            fprintf(stderr, "not yet implemented\n"); break;
        case HOST:
			std::copy(src, src + this->size, this->host_ptr);
        case MANAGED:
            // TODO
			std::copy(src, src + this->size, this->host_ptr);
            // TODO alert cpu modified
            this->sync(); break;
        case CUDA_MANAGED:
            // TODO
            fprintf(stderr, "not yet implemented\n"); break;
    }

    return 0;
}

template <typename T>
error_t memorymanager<T>::sync() {
    // TODO
    return (error_t) 0;
}

template <typename T>
T memorymanager<T>::get(unsigned int idx) {
    assert( idx < size );

    switch (mem_type) {
        case DEVICE:
            // TODO
            return (T) 0;
        case HOST:
            return host_ptr[idx];
        case MANAGED:
            // TODO possibly sync here
            return host_ptr[idx];
        case CUDA_MANAGED:
            // TODO
            return (T) 0;
        default:
            return (T) 0;
    }
}

template <typename T>
void memorymanager<T>::set(unsigned int idx, T val) {
    assert( idx < size );

    switch (mem_type) {
        case DEVICE:
            // TODO
            break;
        case HOST:
            host_ptr[idx] = val; break;
        case MANAGED:
            // TODO possibly sync here
            host_ptr[idx] = val; break;
        case CUDA_MANAGED:
            // TODO
            break;
    }
}

template <typename T>
T* memorymanager<T>::get_device_ptr() {
    return device_ptr;
}

template <typename T>
T* memorymanager<T>::get_host_ptr() {
    return host_ptr;
}

template <typename T>
T* memorymanager<T>::get_cuda_managed_ptr() {
    return cuda_managed_ptr;
}

template <typename T>
T* memorymanager<T>::get_ptr() {
    switch (mem_type) {
        case DEVICE:
            return get_device_ptr();
        case HOST:
            return get_host_ptr();
        case MANAGED:
            fprintf(stderr, "memorymanager::get_ptr() is not defined for MANAGED type memory\n");
            return (T*) NULL;
        case CUDA_MANAGED:
            return get_cuda_managed_ptr();
		default:
			return (T*) NULL;
    }
}




/* COMPILE FOR INT, FLOAT, AND DOUBLE */
template class memorymanager<int>;
template class memorymanager<float>;
template class memorymanager<double>;
