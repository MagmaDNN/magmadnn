/**
 * @file mdnn_device_types.h
 * @author Daniel Nichols
 * @version 1.0
 * @date 2019-07-19
 *
 * @copyright Copyright (c) 2019
 */
#pragma once

/* Named mdnn_device_types.h so as not to interfere with cuda's device_types.h 
    TODO -- refactor/rename this file
*/

#include <cstdint>
#include <iostream>
#include "types.h"

namespace magmadnn {

enum DeviceType { CPU = 0, GPU = 1, COMBINED = 2 };

template <DeviceType dev_type>
struct GetMemoryType {};

template <memory_t mem_type>
struct GetDeviceType {};

#define combine(mem_type, dev_type)                   \
    template <>                                       \
    struct GetMemoryType<dev_type> {                  \
        static constexpr memory_t value = mem_type;   \
    };                                                \
    template <>                                       \
    struct GetDeviceType<mem_type> {                  \
        static constexpr DeviceType value = dev_type; \
    }

combine(HOST, CPU);
combine(DEVICE, GPU);
combine(MANAGED, COMBINED);

#undef combine

#define CALL_FOR_ALL_DEVICES(func) func(CPU) func(GPU) func(COMBINED)

#define FOR_ALL_DEVICE_TYPES(enum_type, dev_type, ...)                      \
    switch (enum_type) {                                                    \
        case CPU: {                                                         \
            const DeviceType dev_type = ::magmadnn::CPU;                    \
            { __VA_ARGS__ }                                                 \
        } break;                                                            \
        case GPU: {                                                         \
            const DeviceType dev_type = ::magmadnn::GPU;                    \
            { __VA_ARGS__ }                                                 \
        } break;                                                            \
        case COMBINED: {                                                    \
            const DeviceType dev_type = ::magmadnn::COMBINED;               \
            { __VA_ARGS__ }                                                 \
        } break;                                                            \
        default:                                                            \
            std::cerr << "Unsupported device type:  " << enum_type << "\n"; \
    }

inline DeviceType getDeviceType(memory_t mem_type) {
#define CASE(mem) \
    case mem:     \
        return GetDeviceType<mem>::value;

    switch (mem_type) {
        CASE(HOST)
        CASE(DEVICE)
        CASE(MANAGED)
        default:
            return CPU;
    }
}

}  // namespace magmadnn