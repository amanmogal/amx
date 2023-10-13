// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/runtime/properties.hpp>
#include <string>

namespace ov {

/**
 * @brief Namespace with Intel AUTO specific properties
 */
namespace intel_auto {
/**
 * @brief auto/multi device setting that enables performance improvement by binding buffer to hw infer request
 */
static constexpr Property<bool> device_bind_buffer{"DEVICE_BIND_BUFFER"};

/**
 * @brief auto device setting that enable/disable CPU as acceleration (or helper device) at the beginning
 */
static constexpr Property<bool> enable_startup_fallback{"ENABLE_STARTUP_FALLBACK"};

/**
 * @brief auto device setting that enable/disable runtime fallback to other devices when infer fails on current
 * selected device
 */
static constexpr Property<bool> enable_runtime_fallback{"ENABLE_RUNTIME_FALLBACK"};

/**
 * @brief Enum to define schedult policy of inference request for MULTI
 * @ingroup ov_runtime_cpp_prop_api
 */
enum class SchedulePolicy {
    ROUND_ROBIN = 0,
    DEVICE_POLICY = 1,
    DEFAULT = ROUND_ROBIN,  //!<  Default schedule policy is ROUND_ROBIN
};

/** @cond INTERNAL */
inline std::ostream& operator<<(std::ostream& os, const SchedulePolicy& policy) {
    switch (policy) {
    case SchedulePolicy::ROUND_ROBIN:
        return os << "ROUND_ROBIN";
    case SchedulePolicy::DEVICE_POLICY:
        return os << "DEVICE_POLICY";
    default:
        OPENVINO_THROW("Unsupported schedule policy value");
    }
}

inline std::istream& operator>>(std::istream& is, SchedulePolicy& policy) {
    std::string str;
    is >> str;
    if (str == "ROUND_ROBIN") {
        policy = SchedulePolicy::ROUND_ROBIN;
    } else if (str == "DEVICE_PRIORITY") {
        policy = SchedulePolicy::DEVICE_POLICY;
    } else if (str == "DEFAULT") {
        policy = SchedulePolicy::DEFAULT;
    } else {
        OPENVINO_THROW("Unsupported schedule policy: ", str);
    }
    return is;
}
/** @endcond */

/**
 * @brief High-level OpenVINO model policy hint
 * Defines what model should be provided with more performant bounded resource first
 * @ingroup ov_runtime_cpp_prop_api
 */
static constexpr Property<SchedulePolicy> schedule_policy{"SCHEDULE_POLICY"};
}  // namespace intel_auto
}  // namespace ov