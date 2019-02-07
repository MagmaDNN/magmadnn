#include "memorymanager.h"

template <typename T>
memorymanager<T>::memorymanager(unsigned int size, memory_t mem_type, device_t device_id) : 
    size(size), mem_type(mem_type), device_id(device_id) {

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
    host_ptr = (T *) malloc(size * sizeof(T));

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
            break;
        case HOST:
            free(host_ptr); break;
        case MANAGED:
            free(host_ptr); break;
        case CUDA_MANAGED:
            break;
        default:
            break;
    }
}

template <typename T>
error_t memorymanager<T>::copy_from(memorymanager<T> src) {

    return 0;
}

template <typename T>
error_t memorymanager<T>::copy_from_host(T *src) {
    return 0;
}

template <typename T>
T memorymanager<T>::get(unsigned int idx) {
    assert( idx < size );

    switch (mem_type) {
        case DEVICE:
            // TODO
            return NULL;
        case HOST:
            return host_ptr[idx];
        case MANAGED:
            // TODO possibly sync here
            return host_ptr[idx];
        case CUDA_MANAGED:
            // TODO
            return NULL;
        default:
            return NULL;
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