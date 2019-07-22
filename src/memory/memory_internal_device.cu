/**
 * @file memory_internal_device.cu
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-02-11
 * 
 * @copyright Copyright (c) 2019
 */
#include "memory/memory_internal_device.h"


namespace magmadnn {
namespace internal {


/** Sets result to the value of arr[idx]. 
	@param arr a device array
	@param idx index of arr to retrieve
	@param result set to arr[idx]. Must be a device allocated variable with size=sizeof(T). 
*/
template <typename T>
__global__ void kernel_get_device_array_element(T *arr, index_t idx, T *result) {
	*result = arr[idx];
}

template <typename T>
T get_device_array_element(T *arr, index_t idx) {
	T host_value;
	T *device_value;
	cudaMalloc(&device_value, sizeof(T));

	kernel_get_device_array_element <<<1, 1>>> (arr, idx, device_value);
	
	cudaMemcpy(&host_value, device_value, sizeof(T), cudaMemcpyDeviceToHost);

	return host_value;
}
#define COMPILE_GETDEVICEARRAYELEMENT(type) template type get_device_array_element(type *arr, index_t idx);
CALL_FOR_ALL_TYPES(COMPILE_GETDEVICEARRAYELEMENT)
#undef COMPILE_GETDEVICEARRAYELEMENT


/** Sets an element on a device.
	@param arr device array
	@param idx index to set
	@param val value to set arr[idx]
*/
template <typename T>
__global__ void kernel_set_device_array_element(T *arr, index_t idx, T val) {
	arr[idx] = val;
}

template <typename T>
void set_device_array_element(T *arr, index_t idx, T val) {
	kernel_set_device_array_element <<<1, 1>>> (arr, idx, val);
}
#define COMPILE_SETDEVICEARRAYELEMENT(type) template void set_device_array_element(type *arr, index_t idx, type val);
CALL_FOR_ALL_TYPES(COMPILE_SETDEVICEARRAYELEMENT)
#undef COMPILE_SETDEVICEARRAYELEMENT

} // namespace internal
} // namespace magmadnn
