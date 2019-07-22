/**
 * @file fill_internal_host.cpp
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-02-12
 *
 * @copyright Copyright (c) 2019
 */

#include "tensor/fill_internal.h"

namespace magmadnn {
namespace internal {

template <typename T>
void fill_uniform(MemoryManager& m, const std::vector<T>& params) {
    assert(params.size() >= 2);

    T start_val, end_val;
    start_val = params[0];
    end_val = params[1];

    std::default_random_engine random_generator;
    std::uniform_real_distribution<T> uniform_distribution(start_val, end_val);

    switch (m.get_memory_type()) {
        case HOST:
            for (unsigned int i = 0; i < m.get_size(); i++)
                m.get_host_ptr<T>()[i] = uniform_distribution(random_generator);
            break;

#if defined(_HAS_CUDA_)
        case DEVICE: {
            MemoryManager host_mem(m.get_size(), m.dtype(), HOST, 0);
            T* host_ptr = host_mem.get_ptr<T>();

            for (unsigned int i = 0; i < m.get_size(); i++) {
                host_ptr[i] = uniform_distribution(random_generator);
            }

            m.copy_from(host_mem);
        } break;
        case MANAGED:
            for (unsigned int i = 0; i < m.get_size(); i++)
                m.get_host_ptr<T>()[i] = uniform_distribution(random_generator);
            m.sync(false);
            break;
        case CUDA_MANAGED:
            for (unsigned int i = 0; i < m.get_size(); i++)
                m.get_cuda_managed_ptr<T>()[i] = uniform_distribution(random_generator);
            m.sync(false);
            break;
#endif
    }
}

#if defined(_HAS_CUDA_)
#    define SPECIALIZE_FILLUNIFORM_INT(type)                                                        \
        template <>                                                                                 \
        void fill_uniform(MemoryManager& m, const std::vector<type>& params) {                      \
            /* define special for `int` because uniform_real_distribution is only for REAL_TYPE */  \
            assert(params.size() >= 2);                                                             \
                                                                                                    \
            type start_val, end_val;                                                                \
            start_val = params[0];                                                                  \
            end_val = params[1];                                                                    \
                                                                                                    \
            std::default_random_engine random_generator;                                            \
            std::uniform_int_distribution<type> uniform_distribution(start_val, end_val);           \
                                                                                                    \
            switch (m.get_memory_type()) {                                                          \
                case HOST:                                                                          \
                    for (unsigned int i = 0; i < m.get_size(); i++)                                 \
                        m.get_host_ptr<type>()[i] = uniform_distribution(random_generator);         \
                    break;                                                                          \
                case DEVICE: {                                                                      \
                    MemoryManager host_mem(m.get_size(), m.dtype(), HOST, 0);                       \
                    type* host_ptr = host_mem.get_ptr<type>();                                      \
                                                                                                    \
                    for (unsigned int i = 0; i < m.get_size(); i++) {                               \
                        host_ptr[i] = uniform_distribution(random_generator);                       \
                    }                                                                               \
                                                                                                    \
                    m.copy_from(host_mem);                                                          \
                } break;                                                                            \
                case MANAGED:                                                                       \
                    for (unsigned int i = 0; i < m.get_size(); i++)                                 \
                        m.get_host_ptr<type>()[i] = uniform_distribution(random_generator);         \
                    m.sync(false);                                                                  \
                    break;                                                                          \
                case CUDA_MANAGED:                                                                  \
                    for (unsigned int i = 0; i < m.get_size(); i++)                                 \
                        m.get_cuda_managed_ptr<type>()[i] = uniform_distribution(random_generator); \
                    m.sync(false);                                                                  \
                    break;                                                                          \
            }                                                                                       \
        }
#else
#    define SPECIALIZE_FILLUNIFORM_INT(type)                                                       \
        template <>                                                                                \
        void fill_uniform(MemoryManager& m, const std::vector<type>& params) {                     \
            /* define special for `int` because uniform_real_distribution is only for REAL_TYPE */ \
            assert(params.size() >= 2);                                                            \
                                                                                                   \
            type start_val, end_val;                                                               \
            start_val = params[0];                                                                 \
            end_val = params[1];                                                                   \
                                                                                                   \
            std::default_random_engine random_generator;                                           \
            std::uniform_int_distribution<type> uniform_distribution(start_val, end_val);          \
                                                                                                   \
            switch (m.get_memory_type()) {                                                         \
                case HOST:                                                                         \
                    for (unsigned int i = 0; i < m.get_size(); i++)                                \
                        m.get_host_ptr<type>()[i] = uniform_distribution(random_generator);        \
                    break;                                                                         \
            }                                                                                      \
        }
#endif

#define COMPILE_FILLUNIFORM(type) template void fill_uniform(MemoryManager&, const std::vector<type>&);
CALL_FOR_ALL_FLOAT_TYPES(COMPILE_FILLUNIFORM)
CALL_FOR_ALL_INT_TYPES(SPECIALIZE_FILLUNIFORM_INT)
#undef COMPILE_FILLUNIFORM
#undef SPECIALIZE_FILLUNIFORM_INT

template <typename T>
void fill_glorot(MemoryManager& m, const std::vector<T>& params) {
    assert(params.size() >= 2);

    T mean, std_dev;
    mean = params[0];
    std_dev = params[1];

    std::default_random_engine random_generator;
    std::normal_distribution<T> normal_dist(mean, std_dev);

    switch (m.get_memory_type()) {
        case HOST:
            for (unsigned int i = 0; i < m.get_size(); i++) m.get_host_ptr<T>()[i] = normal_dist(random_generator);
            break;

#if defined(_HAS_CUDA_)
        case DEVICE:
            // TODO replace with kernel call
            for (unsigned int i = 0; i < m.get_size(); i++) m.set(i, normal_dist(random_generator));
            break;
        case MANAGED:
            for (unsigned int i = 0; i < m.get_size(); i++) m.get_host_ptr<T>()[i] = normal_dist(random_generator);
            m.sync(false);
            break;
        case CUDA_MANAGED:
            for (unsigned int i = 0; i < m.get_size(); i++)
                m.get_cuda_managed_ptr<T>()[i] = normal_dist(random_generator);
            m.sync(false);
            break;
#endif
    }
}

#if defined(_HAS_CUDA_)
#    define SPECIALIZE_FILLGLOROT_INT(type)                                                                        \
        template <>                                                                                                \
        void fill_glorot(MemoryManager& m, const std::vector<type>& params) {                                      \
            /* use binomial for integers */                                                                        \
            assert(params.size() >= 2);                                                                            \
                                                                                                                   \
            type mean, std_dev;                                                                                    \
            mean = params[0];                                                                                      \
            std_dev = params[1];                                                                                   \
                                                                                                                   \
            std::default_random_engine random_generator;                                                           \
            std::binomial_distribution<type> normal_dist(mean, std_dev);                                           \
                                                                                                                   \
            switch (m.get_memory_type()) {                                                                         \
                case HOST:                                                                                         \
                    for (unsigned int i = 0; i < m.get_size(); i++)                                                \
                        m.get_host_ptr<type>()[i] = normal_dist(random_generator);                                 \
                    break;                                                                                         \
                                                                                                                   \
                case DEVICE:                                                                                       \
                    for (unsigned int i = 0; i < m.get_size(); i++) m.set<type>(i, normal_dist(random_generator)); \
                    break;                                                                                         \
                case MANAGED:                                                                                      \
                    for (unsigned int i = 0; i < m.get_size(); i++)                                                \
                        m.get_host_ptr<type>()[i] = normal_dist(random_generator);                                 \
                    m.sync(false);                                                                                 \
                    break;                                                                                         \
                case CUDA_MANAGED:                                                                                 \
                    for (unsigned int i = 0; i < m.get_size(); i++)                                                \
                        m.get_cuda_managed_ptr<type>()[i] = normal_dist(random_generator);                         \
                    m.sync(false);                                                                                 \
                    break;                                                                                         \
            }                                                                                                      \
        }
#else
#    define SPECIALIZE_FILLGLOROT_INT(type)                                        \
        template <>                                                                \
        void fill_glorot(MemoryManager& m, const std::vector<type>& params) {      \
            /* use binomial for integers */                                        \
            assert(params.size() >= 2);                                            \
                                                                                   \
            type mean, std_dev;                                                    \
            mean = params[0];                                                      \
            std_dev = params[1];                                                   \
                                                                                   \
            std::default_random_engine random_generator;                           \
            std::binomial_distribution<type> normal_dist(mean, std_dev);           \
                                                                                   \
            switch (m.get_memory_type()) {                                         \
                case HOST:                                                         \
                    for (unsigned int i = 0; i < m.get_size(); i++)                \
                        m.get_host_ptr<type>()[i] = normal_dist(random_generator); \
                    break;                                                         \
            }                                                                      \
        }
#endif

#define COMPILE_FILLGLOROT(type) template void fill_glorot(MemoryManager&, const std::vector<type>&);
CALL_FOR_ALL_FLOAT_TYPES(COMPILE_FILLGLOROT)
CALL_FOR_ALL_INT_TYPES(SPECIALIZE_FILLGLOROT_INT)
#undef COMPILE_FILLGLOROT
#undef SPECIALIZE_FILLGLOROT_INT

template <typename T>
void fill_mask(MemoryManager& m, const std::vector<T>& params) {
    assert(params.size() >= 1);

    T p;
    p = params[0];

    T val = 1;
    if (params.size() >= 2) val = params[1];

    std::random_device rd;
    std::mt19937 random_generator(rd());
    std::bernoulli_distribution bernoulli(p);

    switch (m.get_memory_type()) {
        case HOST:
            for (unsigned int i = 0; i < m.get_size(); i++) m.get_host_ptr<T>()[i] = bernoulli(random_generator) * val;
            break;

#if defined(_HAS_CUDA_)
        case DEVICE:
            // TODO replace with kernel call
            for (unsigned int i = 0; i < m.get_size(); i++) m.set(i, bernoulli(random_generator) * val);
            break;
        case MANAGED:
            for (unsigned int i = 0; i < m.get_size(); i++) m.get_host_ptr<T>()[i] = bernoulli(random_generator) * val;
            m.sync(false);
            break;
        case CUDA_MANAGED:
            for (unsigned int i = 0; i < m.get_size(); i++)
                m.get_cuda_managed_ptr<T>()[i] = bernoulli(random_generator) * val;
            m.sync(false);
            break;
#endif
    }
}
#define COMPILE_FILLMASK(type) template void fill_mask(MemoryManager&, const std::vector<type>&);
CALL_FOR_ALL_TYPES(COMPILE_FILLMASK)
#undef COMPILE_FILLMASK

template <typename T>
void fill_diagonal(MemoryManager& m, const std::vector<T>& params) {
    bool use_constant_value;
    unsigned int root;
    unsigned int m_size, params_size;
    T val;

    // must have some params
    params_size = params.size();
    assert(params_size > 0);

    m_size = m.get_size();
    root = round(sqrt((double) m_size));
    // make sure it's square memory
    assert(m_size == root * root);

    assert(params_size >= (unsigned int) root || params_size == 1);
    if (params_size == 1)
        use_constant_value = true;
    else
        use_constant_value = false;

    for (unsigned int i = 0; i < m_size; i++) {
        /* if we're on a diagonal element */
        if (i % (root + 1) == 0) {
            if (use_constant_value)
                val = params[0];
            else
                val = params[i / (root + 1)];

            m.set<T>(i, val);
        } else {
            m.set<T>(i, (T) 0);
        }
    }
}
#define COMPILE_FILLDIAGONAL(type) template void fill_diagonal(MemoryManager&, const std::vector<type>&);
CALL_FOR_ALL_TYPES(COMPILE_FILLDIAGONAL)
#undef COMPILE_FILLDIAGONAL

template <typename T>
void fill_constant(MemoryManager& m, const std::vector<T>& params) {
    assert(params.size() >= 1);

    // assume first param is constant value
    T val = (T) params[0];

    switch (m.get_memory_type()) {
        case HOST:
            for (int i = 0; i < (int) m.get_size(); i++) m.get_host_ptr<T>()[i] = val;
            break;

#if defined(_HAS_CUDA_)
        case DEVICE:
            fill_constant_device(m, val);  // fill device pointer
            break;
        case MANAGED:
            fill_constant_device(m, val);                                               // fill device
            for (int i = 0; i < (int) m.get_size(); i++) m.get_host_ptr<T>()[i] = val;  // fill host
            break;
        case CUDA_MANAGED:
            // fill host and sync
            for (int i = 0; i < (int) m.get_size(); i++) m.get_cuda_managed_ptr<T>()[i] = val;
            m.sync(false);
            break;
#endif
    }
}
#define COMPILE_FILLCONSTANT(type) template void fill_constant(MemoryManager&, const std::vector<type>&);
CALL_FOR_ALL_TYPES(COMPILE_FILLCONSTANT);
#undef COMPILE_FILLCONSTANT

}  // namespace internal
}  // namespace magmadnn
