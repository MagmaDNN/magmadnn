/**
 * @file memorymanager.h
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-02-07
 * 
 * @copyright Copyright (c) 2019
 */
#pragma once

#include <vector>
#include <algorithm>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "types.h"

// include cuda files if on GPU
#ifdef __HAS_CUDA__
#include <cuda.h>
#endif

template <typename T>
class memorymanager {
public:
    /** MemoryManager class to keep track of a memory address across devices.
     *  @param size the size of the memory to allocate/manage
     *  @param mem_type what memory type will this data belong to
     *  @param device_id what device will the data reside on (preferred if mem_type is CUDA_MANAGED) 
     */
    memorymanager(unsigned int size, memory_t mem_type, device_t device_id);

    /** Destroys the memory manager object and releases all its data.
     */
    ~memorymanager();

    /** Copies the data from src memory manager into the pointer here. Asserts that
     *  src and this have the same size.
     *  @param src the memorymanager to copy data from
     *  @return the error code (0 - no error, 1 - src ptr not allocated)
     */
    skepsi_error_t copy_from(const memorymanager<T>& src);

    /** copies memory from a host ptr into this memorymanager. will throw an error if it
     *  reaches the end of src allocated mem before this is filled.
     *  @param src the array to copy into this.
     *  @return the error code (0 - good, 1 - not enough memory)
     */
    skepsi_error_t copy_from_host(T *src);

    /** If MANAGED or CUDA_MANAGED this ensures that data is the same on all devices. It 
     * will wait for any gpu kernels to finish before copying data. If HOST or DEVICE memory
     * this does nothing.
     * @return an error code
     */
    skepsi_error_t sync();

    /** Changes the device this memory manager points to. Note that the memory type
     *  is still the same, but the device_id will be different.
     *  @return an error code (0 - ok)
     */
    skepsi_error_t set_device(device_t device_id);

    /** Returns the value at idx. Error if idx is out of range.
     *  @param idx index to retrieve
     *  @return the value at index idx.
     */
    T get(unsigned int idx);

    /** Sets the value at idx to val. Error if idx is out of range.
     *  @param idx index to set
     *  @param val value to set at idx
     */
    void set(unsigned int idx, T val);


    /** returns a CUDA pointer
     *  @return a pointer to the memory on a cuda device.
     */
    T* get_device_ptr();

    /** returns a CPU pointer to the data.
     *  @return cpu pointer
     */
    T* get_host_ptr();
    
    /** returns the managed CUDA memory.
     *  @return pointer to data memory
     */
    T* get_cuda_managed_ptr();
    
    /** Returns a pointer to whatever memory type this is using. Is not
     *  defined for MANAGED memory type (returns NULL).
     *  @return the data ptr
     */
    T* get_ptr();

    /** Returns the size of this memorymanager
     * @return unsigned int  the size of this memory manager
     */
    unsigned int get_size() { return size; }

private:

    /** init with DEVICE parameters */
    void init_device();

    /** init with HOST parameters */
    void init_host();

    /** init with MANAGED parameters */
    void init_managed();

    /** init with CUDA_MANAGED parameters */
    void init_cuda_managed();

	memory_t mem_type;
    device_t device_id;
        
    unsigned int size;
    T* host_ptr;
    T* device_ptr;
    T* cuda_managed_ptr;
};

