#pragma once

#include <vector>
#include <stdio.h>
#include <assert.h>
#include "types.h"

template <typename T>
class memorymanager {

    /** MemoryManager class to keep track of a memory address across devices.
     *  @param size the size of the memory to allocate/manage
     *  @param mem_type what memory type will this data belong to
     *  @param device_id what device will the data reside on (preferred if mem_type is CUDA_MANAGED) 
     */
    memorymanager(unsigned int size, memory_t mem_type, device_t device_id);

    /** Destroys the memory manager object and releases all its data.
     */
    ~memorymanager();

    error_t copy_from(memorymanager<T> src);
    error_t copy_from_host(T *src);

    error_t sync();

    error_t set_device(device_t device_id);

    T get(unsigned int idx);
    void set(unsigned int idx, T val);

    T* get_device_ptr();
    T* get_host_ptr();
    T* get_cuda_managed_ptr();

	private:
        void init_device();
        void init_host();
        void init_managed();
        void init_cuda_managed();

		memory_t mem_type;
        device_t device_id;
        
        T* host_ptr;
        T* device_ptr;
        T* cuda_managed_ptr;
        unsigned int size;

};

