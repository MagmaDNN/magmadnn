/**
 * @file fill_internal_host.cpp
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-02-12
 * 
 * @copyright Copyright (c) 2019
 */

#include "tensor/fill_internal.h"

namespace skepsi {
namespace internal {

template <typename T>
void fill_uniform(memorymanager<T> &m, const std::vector<T>& params) {

    assert( params.size() >= 2 );

    T start_val, end_val;
    start_val = params[0];
    end_val = params[1];

    std::default_random_engine random_generator;
    std::uniform_real_distribution<T> uniform_distribution(start_val, end_val);

    switch (m.get_memory_type()) {
        case HOST:
            for (unsigned int i = 0; i < m.get_size(); i++)
                m.get_host_ptr()[i] = uniform_distribution(random_generator);
            break;

        #if defined(_HAS_CUDA_)
        case DEVICE:
            // TODO replace with kernel call
            for (unsigned int i = 0; i < m.get_size(); i++)
                m.set(i, uniform_distribution(random_generator));
            break;
        case MANAGED:
            for (unsigned int i = 0; i < m.get_size(); i++)
                m.get_host_ptr()[i] = uniform_distribution(random_generator);
            m.sync(false);
            break;
        case CUDA_MANAGED:
            for (unsigned int i = 0; i < m.get_size(); i++)
                m.get_host_ptr()[i] = uniform_distribution(random_generator);
            m.sync(false);
            break;
        #endif
    }
}
template void fill_uniform(memorymanager<int>&, const std::vector<int>&);
template void fill_uniform(memorymanager<float>&, const std::vector<float>&);
template void fill_uniform(memorymanager<double>&, const std::vector<double>&);


template <typename T>
void fill_glorot(memorymanager<T> &m, const std::vector<T>& params) {

    assert( params.size() >= 2 );

    T mean, std_dev;
    mean = params[0];
    std_dev = params[1];

    std::default_random_engine random_generator;
    std::normal_distribution<T> normal_dist(mean, std_dev);

    switch (m.get_memory_type()) {
        case HOST:
            for (unsigned int i = 0; i < m.get_size(); i++)
                m.get_host_ptr()[i] = normal_dist(random_generator);
            break;
            
        #if defined(_HAS_CUDA_)
        case DEVICE:
            // TODO replace with kernel call
            for (unsigned int i = 0; i < m.get_size(); i++)
                m.set(i, normal_dist(random_generator));
            break;
        case MANAGED:
            for (unsigned int i = 0; i < m.get_size(); i++)
                m.get_host_ptr()[i] = normal_dist(random_generator);
            m.sync(false);
            break;
        case CUDA_MANAGED:
            for (unsigned int i = 0; i < m.get_size(); i++)
                m.get_host_ptr()[i] = normal_dist(random_generator);
            m.sync(false);
            break;
        #endif
    }
}
template void fill_glorot(memorymanager<int>&, const std::vector<int>&);
template void fill_glorot(memorymanager<float>&, const std::vector<float>&);
template void fill_glorot(memorymanager<double>&, const std::vector<double>&);


template <typename T>
void fill_constant(memorymanager<T> &m, const std::vector<T>& params) {
    assert( params.size() >= 1 );

    // assume first param is constant value
    T val = (T) params[0];

    switch (m.get_memory_type()) {
        case HOST:
            for (int i = 0; i < (int) m.get_size(); i++) m.get_host_ptr()[i] = val;
            break;
            
        #if defined(_HAS_CUDA_)
        case DEVICE:
	        fill_constant_device(m, val);	// fill device pointer
            break;
        case MANAGED:
	        fill_constant_device(m, val);	// fill device
	        for (int i = 0; i < (int) m.get_size(); i++) m.get_host_ptr()[i] = val; // fill host
            break;
        case CUDA_MANAGED:
	        // fill host and sync
	        for (int i = 0; i < (int) m.get_size(); i++) m.get_cuda_managed_ptr()[i] = val;
            m.sync(false);
            break;
        #endif
    }
}
template void fill_constant(memorymanager<int>&, const std::vector<int>&);
template void fill_constant(memorymanager<float>&, const std::vector<float>&);
template void fill_constant(memorymanager<double>&, const std::vector<double>&);

} // namespace internal
} // namespace skepsi
