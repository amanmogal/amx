// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <map>
#include <string>
#include "ie_icore.hpp"
#include "ie_metric_helpers.hpp"
#include <ie_plugin_config.hpp>
#include "cpp_interfaces/impl/ie_executable_network_thread_safe_default.hpp"
#include "threading/ie_executor_manager.hpp"
#include "threading/ie_immediate_executor.hpp"
#include "threading/ie_istreams_executor.hpp"
#include "threading/ie_itask_executor.hpp"
#include "threading/ie_thread_safe_containers.hpp"
#include "utils/log_util.hpp"
#include <ie_performance_hints.hpp>
#include "openvino/runtime/properties.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "transformations/utils/utils.hpp"
#include "utils/log_util.hpp"
#include "itt.hpp"

#ifdef  MULTIUNITTEST
#define MOCKTESTMACRO virtual
#define MultiDevicePlugin MockMultiDevicePlugin
#else
#define MOCKTESTMACRO
#endif

namespace MultiDevicePlugin {
namespace IE = InferenceEngine;
using DeviceName = std::string;
using IInferPtr = IE::IInferRequestInternal::Ptr;
using IExecNetwork = IE::IExecutableNetworkInternal;
using SoInfer = IE::SoIInferRequestInternal;
using SoExecNetwork = IE::SoExecutableNetworkInternal;
template<typename T>
using DeviceMap = std::unordered_map<DeviceName, T>;
struct DeviceInformation {
    DeviceName deviceName;
    std::map<std::string, std::string> config;
    int numRequestsPerDevices;
    std::string defaultDeviceID;
    DeviceName uniqueName;
    unsigned int devicePriority;
};

class Context : public std::enable_shared_from_this<Context>  {
public:
    using Ptr = std::shared_ptr<Context>;
    std::shared_ptr<IE::ICore> _core;
    std::weak_ptr<IExecNetwork> _executableNetwork;
    virtual ~Context() = default;
};

class MultiContext : public Context {
public:
    using Ptr = std::shared_ptr<MultiContext>;
    std::vector<DeviceInformation>  _devicePriorities;
    std::vector<DeviceInformation>  _devicePrioritiesInitial;
    std::unordered_map<std::string, IE::Parameter> _config;
    DeviceMap<SoExecNetwork> _networksPerDevice;
    std::mutex _mutex;
    bool _needPerfCounters;
    virtual ~MultiContext() = default;
};

class MultiDeviceInferencePlugin;
class AutoContext : public MultiContext {
public:
    using Ptr = std::shared_ptr<AutoContext>;
    std::string _modelPath;
    IE::CNNNetwork _network;
    std::string _strDevices;
    unsigned int _modelPriority = 0;
    bool _batchingDisabled = {false};
    std::mutex _confMutex;
    MultiDeviceInferencePlugin* _plugin;
    virtual ~AutoContext() = default;
};

struct WorkerInferRequest {
    SoInfer _inferRequest;
    IE::Task _task;
    std::exception_ptr _exceptionPtr = nullptr;
    unsigned int _inferCount = 0;
    int _index = 0;
};
}  // namespace MultiDevicePlugin
