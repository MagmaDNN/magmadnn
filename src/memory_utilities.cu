#include "memory_utilities.h"



namespace skepsi {



template <typename T>
__global__ void kernel_get_device_array_element(T *arr, unsigned int idx, T *result) {
	*result = arr[idx];
}


/**	sets the type of the
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
