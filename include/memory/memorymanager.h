/**
 * @file memorymanager.h
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-02-07
 *
 * @copyright Copyright (c) 2019
 */
#pragma once

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <vector>
#include "data_types.h"
#include "types.h"
#include "utilities_internal.h"

// include cuda files if on GPU
#if defined(_HAS_CUDA_)
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "memory_internal_device.h"
#endif

namespace magmadnn {

class MemoryManager {
   public:
    /** MemoryManager class to keep track of a memory address across devices.
     *  @param size the size of the memory to allocate/manage
     *  @param mem_type what memory type will this data belong to
     *  @param device_id what device will the data reside on (preferred if mem_type is CUDA_MANAGED)
     */
    MemoryManager(size_t size, DataType dtype, memory_t mem_type, device_t device_id);

    /** Copy Constructor
     * @param that
     */
    MemoryManager(const MemoryManager& that) = delete;

    /** Move Constructor -- this allows the copying of rvalues
     * @param that some rvalue MemoryManager
     */
    MemoryManager(MemoryManager&& that) = delete;

    /** assignment operator.
     * @param that
     * @return MemoryManager&
     */
    MemoryManager& operator=(MemoryManager that) = delete;

    /** Destroys the memory manager object and releases all its data.
     */
    ~MemoryManager();

    /** Copies the data from src memory manager into the pointer here. Asserts that
     *  src and this have the same size.
     *  @param src the memorymanager to copy data from
     *  @return the error code (0 - no error, 1 - src ptr not allocated)
     */
    magmadnn_error_t copy_from(const MemoryManager& src, index_t begin_idx, size_t size);

    /** Copies the data from src memory manager into the pointer here. Asserts that
     *  src and this have the same size.
     *  @param src the memorymanager to copy data from
     *  @return the error code (0 - no error, 1 - src ptr not allocated)
     */
    magmadnn_error_t copy_from(const MemoryManager& src, size_t size);

    /** Copies the data from src memory manager into the pointer here. Asserts that
     *  src and this have the same size.
     *  @param src the memorymanager to copy data from
     *  @return the error code (0 - no error, 1 - src ptr not allocated)
     */
    magmadnn_error_t copy_from(const MemoryManager& src);

    /** copies memory from a host ptr into this memorymanager. will throw an error if it
     *  reaches the end of src allocated mem before this is filled.
     *  @param src the array to copy into this.
     *  @return the error code (0 - good, 1 - not enough memory)
     */
    template <typename T>
    magmadnn_error_t copy_from_host(const T* src, index_t begin_idx, size_t size);

#if defined(_HAS_CUDA_)
    /** copies memory from a device ptr into this memorymanager. will throw an error if it
     *  reaches the end of src allocated mem before this is filled.
     *  @param src the array to copy into this.
     *  @return the error code (0 - good, 1 - not enough memory)
     */
    template <typename T>
    magmadnn_error_t copy_from_device(const T* src, index_t begin_idx, size_t size);

    /** copies memory from a managed ptr into this memorymanager. will throw an error if it
     *  reaches the end of src allocated mem before this is filled.
     *  @param src the array to copy into this.
     *  @return the error code (0 - good, 1 - not enough memory)
     */
    template <typename T>
    magmadnn_error_t copy_from_managed(const T* host_src, const T* device_src, index_t begin_idx, size_t size);

    /** copies memory from a cuda managed ptr into this memorymanager. will throw an error if it
     *  reaches the end of src allocated mem before this is filled.
     *  @param src the array to copy into this.
     *  @return the error code (0 - good, 1 - not enough memory)
     */
    template <typename T>
    magmadnn_error_t copy_from_cudamanaged(const T* src, index_t begin_idx, size_t size);
#endif

    /** If MANAGED or CUDA_MANAGED this ensures that data is the same on all devices. It
     * will wait for any gpu kernels to finish before copying data. If HOST or DEVICE memory
     * this does nothing.
     * @param gpu_was_modified If true then data will be copied from gpu to cpu, else if false vice-versa.
     * By default true.
     * @return an error code
     */
    magmadnn_error_t sync(bool gpu_was_modified = true) const;

    /** Changes the device this memory manager points to. Note that the memory type
     *  is still the same, but the device_id will be different.
     *  @return an error code (0 - ok)
     */
    magmadnn_error_t set_device(device_t device_id);

    /** Returns the value at idx. Error if idx is out of range.
     *  @param idx index to retrieve
     *  @return the value at index idx.
     */
    template <typename T>
    T get(index_t idx) const;

    /** Sets the value at idx to val. Error if idx is out of range.
     *  @param idx index to set
     *  @param val value to set at idx
     */
    template <typename T>
    void set(index_t idx, T val);

    /** returns a CPU pointer to the data.
     *  @return cpu pointer
     */
    template <typename T>
    T* get_host_ptr();

    /** Get a constant pointer to the host memory
     * @return const T*
     */
    template <typename T>
    const T* get_host_ptr() const;

#if defined(_HAS_CUDA_)
    /** returns a CUDA pointer
     *  @return a pointer to the memory on a cuda device.
     */
    template <typename T>
    T* get_device_ptr();

    /** get a constant pointer to device memory
     * @return const T*
     */
    template <typename T>
    const T* get_device_ptr() const;

    /** returns the managed CUDA memory.
     *  @return pointer to data memory
     */
    template <typename T>
    T* get_cuda_managed_ptr();

    /** get a constant pointer to cuda_managed memory
     * @return const T*
     */
    template <typename T>
    const T* get_cuda_managed_ptr() const;
#endif

    /** Returns a pointer to whatever memory type this is using. For MANAGED
     *  memory type it returns the device pointer.
     *  @return the data ptr
     */
    template <typename T>
    T* get_ptr();

    /** Get a constant pointer to this MemoryManager's memory.
     * @return const T*
     */
    template <typename T>
    const T* get_ptr() const;

    /** Returns the size of this memorymanager
     * @return unsigned int  the size of this memory manager
     */
    size_t get_size() const { return size_; }

    size_t size() const { return size_; }

    DataType dtype() const { return dtype_; }

    /** Returns the memory type of this memory manager.
     * @return memory_t
     */
    memory_t get_memory_type() const { return mem_type_; }

   private:
    /** init with HOST parameters */
    void init_host();

#if defined(_HAS_CUDA_)
    /** init with DEVICE parameters */
    void init_device();

    /** init with MANAGED parameters */
    void init_managed();

    /** init with CUDA_MANAGED parameters */
    void init_cuda_managed();
#endif

    DataType dtype_;
    memory_t mem_type_;
    device_t device_id_;

    size_t size_;
    void* host_ptr_;

#if defined(_HAS_CUDA_)
    void* device_ptr_;
    void* cuda_managed_ptr_;
#endif
};

}  // namespace magmadnn
