#include "memory_utilities.h"



namespace skepsi {



/** Sets result to the value of arr[idx]. 
	@param arr a device array
	@param idx index of arr to retrieve
	@param result set to arr[idx]. Must be a device allocated variable with size=sizeof(T). 
*/
template <typename T>
__global__ void kernel_get_device_array_element(T *arr, unsigned int idx, T *result) {
	*result = arr[idx];
}


/**	gets the device array element at idx.
	@param arr the device array
	@param idx the index to retrieve the array from
*/
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



}; // namespace skepsi
