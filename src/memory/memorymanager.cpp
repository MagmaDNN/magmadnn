/**
 * @file memorymanager.cpp
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-02-07
 * 
 * @copyright Copyright (c) 2019
 */
#include "memory/memorymanager.h"

namespace skepsi {

template <typename T>
memorymanager<T>::memorymanager(unsigned int size, memory_t mem_type, device_t device_id) : 
    mem_type(mem_type), size(size) {

		set_device(device_id);

        // initialize based on the chosen memory type
        switch (mem_type) {
            case HOST:
                init_host(); break;
            #if defined(_HAS_CUDA_)
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

#if defined(_HAS_CUDA_)
template <typename T>
void memorymanager<T>::init_device() {
    cudaMalloc((void**) &device_ptr, size * sizeof(T));
}

template <typename T>
void memorymanager<T>::init_managed() {
    host_ptr = (T *) std::malloc(size * sizeof(T));
    cudaMalloc((void**) &device_ptr, size * sizeof(T));
}

template <typename T>
void memorymanager<T>::init_cuda_managed() {
    cudaMallocManaged((void**) &cuda_managed_ptr, size * sizeof(T));
}
#endif

template <typename T>
memorymanager<T>::~memorymanager<T>() {
    switch (mem_type) {
        case HOST:
            std::free(host_ptr); break;
        #if defined(_HAS_CUDA_)
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
    #if defined(_HAS_CUDA_)
    else if (src.mem_type == DEVICE) {
        return copy_from_device(src.device_ptr, begin_idx, copy_size);
    } else if (src.mem_type == MANAGED) {
        return copy_from_managed(src.host_ptr, src.device_ptr, begin_idx, copy_size);
    } else if (src.mem_type == CUDA_MANAGED) {
        return copy_from_cudamanaged(src.cuda_managed_ptr, begin_idx, copy_size);
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
        #if defined(_HAS_CUDA_)
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
            std::copy(src+begin_idx, (src+begin_idx) + copy_size, cuda_managed_ptr);
            sync(false);
            return (skepsi_error_t) 0;
        #endif
    }

    return (skepsi_error_t) 1;
}

#if defined(_HAS_CUDA_)
template <typename T>
skepsi_error_t memorymanager<T>::copy_from_device(T *src, unsigned int begin_idx, unsigned int copy_size) {

	skepsi_error_t err = (skepsi_error_t) 0;

    switch (mem_type) {
        case HOST:
            // device --> host
            err = cudaMemcpy(host_ptr, src+begin_idx, copy_size*sizeof(T), cudaMemcpyDeviceToHost);
        case DEVICE:
            // device --> device
            err = cudaMemcpy(device_ptr, src+begin_idx, copy_size*sizeof(T), cudaMemcpyDeviceToDevice);
        case MANAGED:
            // device --> managed
            err = cudaMemcpy(device_ptr, src+begin_idx, copy_size*sizeof(T), cudaMemcpyDeviceToDevice);
            sync(true);
            return err;
        case CUDA_MANAGED:
            // device --> cmanaged
            err = cudaMemcpy(cuda_managed_ptr, src+begin_idx, copy_size*sizeof(T), cudaMemcpyDeviceToDevice);
            sync(true);
            return err;
    }

    return err;
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
            std::copy(host_src+begin_idx, (host_src+begin_idx) + copy_size, cuda_managed_ptr);
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
            std::copy(src+begin_idx, (src+begin_idx) + copy_size, cuda_managed_ptr);
            sync(false);
            return (skepsi_error_t) 0;
    }

    return (skepsi_error_t) 1;
}
#endif

template <typename T>
skepsi_error_t memorymanager<T>::sync(bool gpu_was_modified) {
    #if defined(_HAS_CUDA_)
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
T memorymanager<T>::get(unsigned int idx) const {
    assert( idx < size );

    switch (mem_type) {
        case HOST:
            return host_ptr[idx];
        #if defined(_HAS_CUDA_)
        case DEVICE:
			cudaSetDevice(device_id);
            return internal::get_device_array_element(device_ptr, idx);
        case MANAGED:
            return host_ptr[idx];
        case CUDA_MANAGED:
            return cuda_managed_ptr[idx];
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
        #if defined(_HAS_CUDA_)
        case DEVICE:
			cudaSetDevice(device_id);
            internal::set_device_array_element(device_ptr, idx, val); break;
        case MANAGED:
            host_ptr[idx] = val;
			cudaSetDevice(device_id);
            internal::set_device_array_element(device_ptr, idx, val);
            break;
        case CUDA_MANAGED:
            cuda_managed_ptr[idx] = val; break;
        #endif
    }
}

template <typename T>
skepsi_error_t memorymanager<T>::set_device(device_t device_id) {

    #if defined(_HAS_CUDA_)
	int n_devices = 0;
	cudaGetDeviceCount(&n_devices);
	if ((int)device_id >= n_devices) {
		fprintf(stderr, "invalid device id\n");
		return (skepsi_error_t) 1;
	}
    #endif

	this->device_id = device_id;
	return (skepsi_error_t) 0;
}

template <typename T>
T* memorymanager<T>::get_host_ptr() {
    return host_ptr;
}

#if defined(_HAS_CUDA_)
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
        #if defined(_HAS_CUDA_)
        case DEVICE:
            return get_device_ptr();
        case MANAGED:
            // returns device by default for managed
            return get_device_ptr();
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
