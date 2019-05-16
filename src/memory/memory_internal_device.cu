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
__global__ void kernel_get_device_array_element(T *arr, unsigned int idx, T *result) {
	*result = arr[idx];
}

template <typename T>
T get_device_array_element(T *arr, unsigned int idx) {
	T host_value;
	T *device_value;
	cudaMalloc(&device_value, sizeof(T));

	kernel_get_device_array_element <<<1, 1>>> (arr, idx, device_value);
	
	cudaMemcpy(&host_value, device_value, sizeof(T), cudaMemcpyDeviceToHost);

	return host_value;
}
template int get_device_array_element(int *arr, unsigned int idx);
template float get_device_array_element(float *arr, unsigned int idx);
template double get_device_array_element(double *arr, unsigned int idx);


/** Sets an element on a device.
	@param arr device array
	@param idx index to set
	@param val value to set arr[idx]
*/
template <typename T>
__global__ void kernel_set_device_array_element(T *arr, unsigned int idx, T val) {
	arr[idx] = val;
}

template <typename T>
void set_device_array_element(T *arr, unsigned int idx, T val) {
	kernel_set_device_array_element <<<1, 1>>> (arr, idx, val);
}
template void set_device_array_element(int *arr, unsigned int idx, int val);
template void set_device_array_element(float *arr, unsigned int idx, float val);
template void set_device_array_element(double *arr, unsigned int idx, double val);

} // namespace internal
} // namespace magmadnn
