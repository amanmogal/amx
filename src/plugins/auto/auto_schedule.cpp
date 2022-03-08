// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include <mutex>
#include <string>
#include <vector>
#include <memory>
#include <utility>
#include <map>
#include <unordered_map>

#include "ie_icore.hpp"
#include "ie_metric_helpers.hpp"
#include <ie_plugin_config.hpp>
#include "auto_executable_network.hpp"
#include "base_async_infer_request.hpp"
#include "plugin.hpp"

#include "ngraph/opsets/opset1.hpp"
#include "transformations/utils/utils.hpp"
#include "utils/log_util.hpp"

#include "itt.hpp"
// ------------------------------AutoSchedule----------------------------
namespace MultiDevicePlugin {
using namespace InferenceEngine;

namespace {
std::string GetNetworkPrecision(const InferenceEngine::CNNNetwork &network) {
    auto nGraphFunc = network.getFunction();
    bool isINTModel = ngraph::op::util::has_op_with_type<ngraph::op::FakeQuantize>(nGraphFunc);
    if (isINTModel) {
        return METRIC_VALUE(INT8);
    }
    for (auto & node : nGraphFunc->get_ordered_ops()) {
        if (std::dynamic_pointer_cast<ngraph::opset1::Convolution>(node) ||
            std::dynamic_pointer_cast<ngraph::opset1::GroupConvolution>(node) ||
            std::dynamic_pointer_cast<ngraph::opset1::GroupConvolutionBackpropData>(node) ||
            std::dynamic_pointer_cast<ngraph::opset1::ConvolutionBackpropData>(node)) {
            auto layerType = node->input(1).get_element_type().get_type_name();
            if (layerType == "f32")
                return METRIC_VALUE(FP32);
            if (layerType == "f16")
                return METRIC_VALUE(FP16);
        }
    }
    return METRIC_VALUE(FP32);
}
}  // namespace

void AutoSchedule::GenerateWorkers(const std::string& device, const SoExecutableNetworkInternal& executableNetwork) {
    std::string realDeviceName;
    if (device == "CPU_HELP") {
        realDeviceName = "CPU";
    } else {
        realDeviceName = device;
    }
    auto itNumRequests = std::find_if(_autoContext->_devicePriorities.cbegin(),
                                      _autoContext->_devicePriorities.cend(),
                                      [&realDeviceName](const DeviceInformation& d){ return d.deviceName == realDeviceName;});
    unsigned int optimalNum = 0;
    try {
        optimalNum = executableNetwork->GetMetric(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)).as<unsigned int>();
    } catch (const InferenceEngine::Exception &iie) {
        IE_THROW()
            << "Every device used with the Multi-Device should "
            << "support OPTIMAL_NUMBER_OF_INFER_REQUESTS ExecutableNetwork metric. "
            << "Failed to query the metric for the " << device << " with error:" << iie.what();
    }
    const auto numRequests = (_autoContext->_devicePriorities.end() == itNumRequests ||
                              itNumRequests->numRequestsPerDevices == -1) ? optimalNum : itNumRequests->numRequestsPerDevices;
    auto& workerRequests = _workerRequests[device];
    auto& idleWorkerRequests = _idleWorkerRequests[device];
    workerRequests.resize(numRequests);
    _inferPipelineTasksDeviceSpecific[device] = std::unique_ptr<ThreadSafeQueue<Task>>(new ThreadSafeQueue<Task>);
    auto* idleWorkerRequestsPtr = &(idleWorkerRequests);
    idleWorkerRequests.set_capacity(numRequests);
    int num = 0;
    for (auto&& workerRequest : workerRequests) {
        workerRequest._inferRequest = {executableNetwork->CreateInferRequest(), executableNetwork._so};
        auto* workerRequestPtr = &workerRequest;
        workerRequestPtr->_index = num++;
        IE_ASSERT(idleWorkerRequests.try_push(std::make_pair(workerRequestPtr->_index, workerRequestPtr)) == true);
        workerRequest._inferRequest->SetCallback(
            [workerRequestPtr, this, device, idleWorkerRequestsPtr] (std::exception_ptr exceptionPtr) mutable {
                IdleGuard idleGuard{workerRequestPtr, *idleWorkerRequestsPtr};
                workerRequestPtr->_exceptionPtr = exceptionPtr;
                {
                    auto capturedTask = std::move(workerRequestPtr->_task);
                    capturedTask();
                }
                // try to return the request to the idle list (fails if the overall object destruction has began)
                if (idleGuard.Release()->try_push(std::make_pair(workerRequestPtr->_index, workerRequestPtr))) {
                    // let's try to pop a task, as we know there is at least one idle request, schedule if succeeded
                    // if no device-agnostic tasks, let's try pop the device specific task, schedule if succeeded
                    Task t;
                    if (_inferPipelineTasks.try_pop(t))
                        ScheduleToWorkerInferRequest(std::move(t));
                    else if (_inferPipelineTasksDeviceSpecific[device]->try_pop(t))
                        ScheduleToWorkerInferRequest(std::move(t), device);
                }
            });
    }
}

void AutoSchedule::init(const Context::Ptr& context) {
    LOG_INFO("[AUTOPLUGIN]ExecutableNetwork start");
    Schedule::init(context);
    _multiContext = std::dynamic_pointer_cast<MultiContext>(_context);
    _autoContext = std::dynamic_pointer_cast<AutoContext>(_context);
    if (_autoContext->_core == nullptr) {
        IE_THROW() << "Please, work with Auto device via InferencEngine::Core object";
    }

    if (_autoContext->_modelPath.empty() && _autoContext->_network.getFunction() == nullptr) {
        IE_THROW() << "AUTO device supports just ngraph network representation";
    }

    _autoContext->_config[MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES] = _autoContext->_strDevices;
    std::string profilingTask = "AutoSchedule::AutoSchedule:AutoMode";

    // loadContext[ACTUALDEVICE] is always enabled,
    // when there is CPU and there are more than two devices, loadContext[CPU] is enabled
    _loadContext[ACTUALDEVICE].isEnabled = true;
    _loadContext[ACTUALDEVICE].networkPrecision = GetNetworkPrecision(_autoContext->_network);
    _loadContext[ACTUALDEVICE].metaDevices = _autoContext->_devicePriorities;
    _loadContext[ACTUALDEVICE].deviceInfo = _autoContext->_plugin->SelectDevice(_autoContext->_devicePriorities,
            _loadContext[ACTUALDEVICE].networkPrecision, _autoContext->_modelPriority);
    LOG_INFO("[AUTOPLUGIN]:select device:%s", _loadContext[ACTUALDEVICE].deviceInfo.deviceName.c_str());
    bool isActualDevCPU =
        _loadContext[ACTUALDEVICE].deviceInfo.deviceName.find("CPU") != std::string::npos;
    // if Actual device is CPU, disabled _loadContext[CPU], only use _loadContext[ACTUALDEVICE]
    if (isActualDevCPU) {
        _loadContext[CPU].isEnabled = false;
    } else {
        const auto CPUIter = std::find_if(_autoContext->_devicePriorities.begin(),
                _autoContext->_devicePriorities.end(),
                [=](const DeviceInformation& d)->bool{return d.deviceName.find("CPU") != std::string::npos;});
        // if have CPU Device,  enable _loadContext[CPU]
        if (CPUIter != _autoContext->_devicePriorities.end()) {
            _loadContext[CPU].isEnabled = true;
            _loadContext[CPU].deviceInfo = *CPUIter;
            _loadContext[CPU].deviceInfo.config[CONFIG_KEY(PERFORMANCE_HINT)] =
                InferenceEngine::PluginConfigParams::LATENCY;
            _loadContext[CPU].workName = "CPU_HELP";
            LOG_INFO("[AUTOPLUGIN]:will load CPU for accelerator");
        } else {
            _loadContext[CPU].isEnabled = false;
        }
    }

    // initialize the rest members of load context
    for (int i = 0; i < CONTEXTNUM; i++) {
         if (_loadContext[i].isEnabled) {
             _loadContext[i].future =  _loadContext[i].promise.get_future();
              auto* contextPtr = &_loadContext[i];
              auto modelPath = _autoContext->_modelPath;
              auto network = _autoContext->_network;
             _loadContext[i].task = [this, contextPtr, modelPath, network]() mutable {
                      TryToLoadNetWork(*contextPtr, modelPath, network);
                      if (contextPtr->isLoadSuccess) {
                          if (contextPtr->workName.empty()) {
                                contextPtr->workName = contextPtr->deviceInfo.deviceName;
                          }
                          GenerateWorkers(contextPtr->workName, contextPtr->executableNetwork);
                          //need lock
                          {
                             std::lock_guard<std::mutex> lock(_autoContext->_confMutex);
                             _autoContext->_config.insert(contextPtr->deviceInfo.config.begin(),
                                     contextPtr->deviceInfo.config.end());
                          }
                          contextPtr->isAlready = true;
                          auto& deviceName = contextPtr->deviceInfo.deviceName;
                          LOG_INFO("[AUTOPLUGIN]:device:%s loading Network finished",
                                  deviceName.c_str());
                          auto supported_config_keys =
                              _autoContext->_core->GetMetric(deviceName,
                                      METRIC_KEY(SUPPORTED_CONFIG_KEYS)).as<std::vector<std::string>>();
                          // there is log mutex in LOG_DEBUG, add _configMutex just want to print them all together
                          // toDo maybe neet to implement LOG_RUN(task, LOG_LEVEL) to run some debug code.
                          std::lock_guard<std::mutex> lock(_autoContext->_confMutex);
                          for (const auto& cfg : supported_config_keys) {
                              try {
                                  LOG_DEBUG("[AUTOPLUGIN]:device:%s, GetConfig:%s=%s", deviceName.c_str(),
                                          cfg.c_str(), contextPtr->executableNetwork->GetConfig(cfg).as<std::string>().c_str());
                              } catch (...) {
                              }
                          }
                      }
                      contextPtr->promise.set_value();
                      // the first load network process finished
                      std::call_once(_firstLoadOC, [this] () {
                              _firstLoadPromise.set_value();
                              });
             };
         }
    }

    OV_ITT_SCOPED_TASK(itt::domains::MULTIPlugin, openvino::itt::handle(profilingTask));
    if (_loadContext[CPU].isEnabled) {
        _firstLoadFuture = _firstLoadPromise.get_future();
        // will not wait for loading accelerator network,
        // so the executor can't be destroyed before finished the task,
        // so use executor as a member of AutoSchedule.
        _executor = _autoContext->_plugin->executorManager()->getIdleCPUStreamsExecutor(
                IStreamsExecutor::Config{"AutoDeviceAsyncLoad",
                static_cast<int>(std::thread::hardware_concurrency()) /* max possible #streams*/,
                0 /*default threads per stream, workaround for ticket 62376*/,
                IStreamsExecutor::ThreadBindingType::NONE});
        for (auto&& device : _autoContext->_devicePriorities) {
            // initialize containers before run async task
            _idleWorkerRequests[device.deviceName];
            _workerRequests[device.deviceName];
            _inferPipelineTasksDeviceSpecific[device.deviceName] = nullptr;
        }
        _idleWorkerRequests["CPU_HELP"];
        _workerRequests["CPU_HELP"];
        _inferPipelineTasksDeviceSpecific["CPU_HELP"] = nullptr;
        _executor->run(_loadContext[CPU].task);
        _executor->run(_loadContext[ACTUALDEVICE].task);
        auto recycleTask = [this]() mutable {
            WaitActualNetworkReady();
            while (!_exitFlag && _loadContext[ACTUALDEVICE].isAlready) {
                // handle the case of ACTUAL faster than CPU
                _loadContext[CPU].future.wait();
                // clean up helper infer requests
                // first, wait for all the remaining requests to finish
                for (auto& iter : _workerRequests["CPU_HELP"]) {
                    iter._inferRequest._ptr->Wait(InferRequest::WaitMode::RESULT_READY);
                }
                // late enough to check the idle queue now
                // second, check the idle queue if all requests are in place
                size_t destroynum = 0;
                std::pair<int, WorkerInferRequest*> worker;
                while (_idleWorkerRequests["CPU_HELP"].try_pop(worker)) {
                    destroynum++;
                    _cpuHelpInferCount += worker.second->_inferCount;
                }
                if (destroynum == _workerRequests["CPU_HELP"].size()) {
                    std::lock_guard<std::mutex> lock(_autoContext->_confMutex);
                    _workerRequests["CPU_HELP"].clear();
                    _loadContext[CPU].executableNetwork._ptr.reset();
                    _loadContext[CPU].executableNetwork._so.reset();
                    break;
                }
            }
        };
        _executor->run(std::move(recycleTask));
    } else {
        // only one device need to load network, do not need to load it async
        _loadContext[ACTUALDEVICE].task();
    }
    WaitFirstNetworkReady();
}
void AutoSchedule::TryToLoadNetWork(AutoLoadContext& context,
                                                    const std::string& modelPath,
                                                    const InferenceEngine::CNNNetwork& network) {
    auto& device = context.deviceInfo.deviceName;
    auto& deviceConfig = context.deviceInfo.config;
    auto& deviceList = context.metaDevices;
    bool curDevIsCPU = (device.find("CPU") != std::string::npos);
    bool curDevIsGPU = (device.find("GPU") != std::string::npos);
    {
        std::lock_guard<std::mutex> lock(_autoContext->_confMutex);
        if (curDevIsGPU && _loadContext[CPU].isEnabled) {
            // user does not set the compiling threads
            // limit the threads num for compiling
            int maxNumThreads = 0;
            try {
                maxNumThreads = _autoContext->_core->GetConfig(device,
                        GPU_CONFIG_KEY(MAX_NUM_THREADS)).as<int>();
            } catch (...) {
                LOG_DEBUG("[AUTOPLUGIN]: cannot get MAX_NUM_THREADS from GPU");
            }
            if (maxNumThreads == static_cast<int>(std::thread::hardware_concurrency())) {
                int threadNum = maxNumThreads / 2;
                deviceConfig[GPU_CONFIG_KEY(MAX_NUM_THREADS)] = std::to_string(threadNum).c_str();
                LOG_DEBUG("[AUTO PLUGIN]:gpu streams number for compiling: %s", deviceConfig[GPU_CONFIG_KEY(MAX_NUM_THREADS)].c_str());
            } else {
                // user set the compiling threads num
                // use the user's val anyway
                LOG_DEBUG("[AUTOPLUGIN]:user defined compiling threads: %d", maxNumThreads);
            }
        }
    }
    try {
        if (!modelPath.empty()) {
            context.executableNetwork = _autoContext->_core->LoadNetwork(modelPath, device, deviceConfig);
        } else {
            context.executableNetwork = _autoContext->_core->LoadNetwork(network, device, deviceConfig);
        }
        context.isLoadSuccess = true;
    } catch (const std::exception& e) {
        context.errMessage += device + ":" + e.what();
        context.isLoadSuccess = false;
    }

    if (context.isLoadSuccess || curDevIsCPU) {
        return;
    }

    // need to reload network, unregister it's priority
    // there maybe potential issue.
    // for example they are dGPU, VPUX, iGPU, customer want to LoadNetwork with
    // configure 0 dGPU, 1 VPUX, if dGPU load failed,
    // the result will be not sure, maybe two network are loaded into VPUX,
    // maybe 0 is loaded to VPUX, 1 is loaded to iGPU
    _autoContext->_plugin->UnregisterPriority(_autoContext->_modelPriority,
            context.deviceInfo.uniqueName);
    // remove the current device from deviceList
    auto eraseDevice = std::find_if(deviceList.begin(), deviceList.end(),
            [device](DeviceInformation& d){
            return d.deviceName == device;
            });
    deviceList.erase(eraseDevice);

    if (deviceList.empty()) {
        return;
    }

    // select next candidate device
    try {
        std::lock_guard<std::mutex> lock(_autoContext->_confMutex);
        context.deviceInfo = _autoContext->_plugin->SelectDevice(deviceList,
                context.networkPrecision, _autoContext->_modelPriority);
    }
    catch (const std::exception& e) {
        return;
    }

    // if the select device is CPU, need to check the config of _loadContext[CPU]
    // if they are same, do not need to load again
    curDevIsCPU = (context.deviceInfo.deviceName.find("CPU") != std::string::npos);
    if (curDevIsCPU) {
        auto compare = [](std::map<std::string, std::string>& a,
                std::map<std::string, std::string>& b) -> bool {
            if (a.size() != b.size()) {
                return false;
            }
            for (auto& item : a) {
                auto bIter = b.find(item.first);
                if (bIter != b.end()) {
                    if (bIter->second != item.second) {
                        return false;
                    }
                } else {
                    return false;
                }
            }
            return true;
        };
        if (compare(context.deviceInfo.config, _loadContext[CPU].deviceInfo.config)) {
            return;
        }
    }

    LOG_DEBUG("[AUTOPLUGIN] try to load %s", context.deviceInfo.deviceName.c_str());
    // try to load this candidate device
    TryToLoadNetWork(context, modelPath, network);
}

void AutoSchedule::WaitFirstNetworkReady() {
    if (_firstLoadFuture.valid()) {
        // wait for the first loading finished
        _firstLoadFuture.wait();
    }

    // check if there is any device that have loaded network successfully
    for (int i = CONTEXTNUM - 1; i >= 0; i--) {
        if (_loadContext[i].isEnabled && _loadContext[i].isAlready) {
            return;
        }
    }

    // the first loading is failed, wait for another loading
    for (int i = CONTEXTNUM - 1; i >= 0; i--) {
        if (_loadContext[i].isEnabled) {
            _loadContext[i].future.wait();
            // check if loading is successful
            if (_loadContext[i].isAlready) {
                return;
            }
        }
    }

    //print errMessage
    for (int i = CONTEXTNUM - 1; i >= 0; i--) {
        if (_loadContext[i].isEnabled) {
            LOG_ERROR("[AUTOPLUGIN] load failed, %s", _loadContext[i].errMessage.c_str());
        }
    }

    IE_THROW() << "[AUTOPLUGIN] load all devices failed";
}

void AutoSchedule::WaitActualNetworkReady() const {
    OV_ITT_SCOPED_TASK(itt::domains::MULTIPlugin, "AutoSchedule::WaitActualNetworkReady");
    // Maybe different API will call this function, so add call once here
    // for every AutoSchedule instance
    std::call_once(_oc, [this] () {
               if (_loadContext[ACTUALDEVICE].future.valid()) {
                   _loadContext[ACTUALDEVICE].future.wait();
               }
               });
}

void AutoSchedule::ScheduleToWorkerInferRequest(Task inferPipelineTask, DeviceName preferred_device) {
    std::vector<DeviceInformation> devices;
    // AUTO work mode
    if (!preferred_device.empty()) {
        // if the device needed by customer is not ready, need to wait for it
        WaitActualNetworkReady();
        // the preferred_device should be the selected device in AUTO work mode
        if (preferred_device != _loadContext[ACTUALDEVICE].deviceInfo.deviceName) {
            IE_THROW(NotFound) << "The preferred device should be the selected device";
        }
        devices.push_back(_loadContext[ACTUALDEVICE].deviceInfo);
    } else {
        // _acceleratorDevice could be the same as _cpuDevice, such as AUTO:CPU
        if (_loadContext[ACTUALDEVICE].isAlready) {
            devices.push_back(_loadContext[ACTUALDEVICE].deviceInfo);
        } else {
            // replace deviceName with workName, so schedule can select correct
            // idleWorkerQueue
            auto deviceInfo =  _loadContext[CPU].deviceInfo;
            deviceInfo.deviceName = _loadContext[CPU].workName;
            devices.push_back(std::move(deviceInfo));
        }
    }

    for (auto&& device : devices) {
        if (!preferred_device.empty() && (device.deviceName != preferred_device))
            continue;
        if (RunPipelineTask(inferPipelineTask, _idleWorkerRequests[device.deviceName], preferred_device)) {
            return;
        }
    }

    // no vacant requests this time, storing the task to the respective queue
    if (!preferred_device.empty())
        _inferPipelineTasksDeviceSpecific[preferred_device]->push(std::move(inferPipelineTask));
    else
        _inferPipelineTasks.push(std::move(inferPipelineTask));
}


AutoSchedule::~AutoSchedule() {
    // this is necessary to guarantee member destroyed after getting future
    if (_loadContext[CPU].isEnabled) {
        _exitFlag = true;
        _loadContext[CPU].future.wait();
        WaitActualNetworkReady();
        // it's necessary to wait the loading network threads to stop here.
        _autoContext->_plugin->executorManager()->clear("AutoDeviceAsyncLoad");
        _executor.reset();
    }
    _autoContext->_plugin->UnregisterPriority(_autoContext->_modelPriority,
            _loadContext[ACTUALDEVICE].deviceInfo.uniqueName);
    for (auto&& _workerRequest : _workerRequests) {
         unsigned int count = 0;
         for (auto& request : _workerRequest.second) {
             count += request._inferCount;
         }
         if (_workerRequest.first == "CPU_HELP") {
             LOG_INFO("[AUTOPLUGIN]CPU_HELP:infer:%ld", _cpuHelpInferCount + count);
         }
    }

    LOG_INFO("[AUTOPLUGIN]ExecutableNetwork end");
}


IInferPtr AutoSchedule::CreateInferRequestImpl(
    const std::vector<std::shared_ptr<const ov::Node>>& inputs,
    const std::vector<std::shared_ptr<const ov::Node>>& outputs) {
    auto num = _numRequestsCreated++;
    InferenceEngine::SoIInferRequestInternal request_to_share_blobs_with;
    InferenceEngine::RemoteContext::Ptr ctx = nullptr;

    if (!_loadContext[CPU].isEnabled && _loadContext[ACTUALDEVICE].isAlready) {
        try {
            ctx = _autoContext->_core->GetDefaultContext(_loadContext[ACTUALDEVICE].deviceInfo.deviceName);
        } catch (InferenceEngine::Exception& ex) {
            // plugin does not support context, say CPU
            LOG_DEBUG("[AUTOPLUGIN]context not supported for %s, fallback to default memory",
                    _loadContext[ACTUALDEVICE].deviceInfo.deviceName.c_str());
            // for dynamic shape support
            auto& dev_requests = _workerRequests[_loadContext[ACTUALDEVICE].deviceInfo.deviceName];
            if (num < dev_requests.size()) {
                request_to_share_blobs_with = dev_requests.at(num)._inferRequest;
            }
        }
    }
    return std::make_shared<MultiDeviceInferRequest>(inputs, outputs, request_to_share_blobs_with, ctx);
}

IInferPtr AutoSchedule::CreateInferRequestImpl(InferenceEngine::InputsDataMap networkInputs,
        InferenceEngine::OutputsDataMap networkOutputs) {
    auto num = _numRequestsCreated++;
    InferenceEngine::SoIInferRequestInternal request_to_share_blobs_with;
    InferenceEngine::RemoteContext::Ptr ctx = nullptr;

    if (!_loadContext[CPU].isEnabled && _loadContext[ACTUALDEVICE].isAlready) {
        try {
            ctx = _autoContext->_core->GetDefaultContext(_loadContext[ACTUALDEVICE].deviceInfo.deviceName);
        } catch (InferenceEngine::Exception& ex) {
            // plugin does not support context
            LOG_DEBUG("[AUTOPLUGIN]context not supported for %s, fallback to default memory",
                    _loadContext[ACTUALDEVICE].deviceInfo.deviceName.c_str());
            auto& dev_requests = _workerRequests[_loadContext[ACTUALDEVICE].deviceInfo.deviceName];
            if (num < dev_requests.size()) {
                request_to_share_blobs_with = dev_requests.at(num)._inferRequest;
            }
        }
    }
    return std::make_shared<MultiDeviceInferRequest>(networkInputs, networkOutputs, request_to_share_blobs_with, ctx);
}

IInferPtr AutoSchedule::CreateInferRequest() {
    IInferRequestInternal::Ptr syncRequestImpl;
    auto execNetwork = std::dynamic_pointer_cast<AutoExecutableNetwork>(
            _autoContext->_executableNetwork.lock());
    if (_multiContext->_core && _multiContext->_core->isNewAPI())
        syncRequestImpl = CreateInferRequestImpl(execNetwork->_parameters,
                execNetwork->_results);

    if (!syncRequestImpl)
        syncRequestImpl = CreateInferRequestImpl(execNetwork->_networkInputs,
                execNetwork->_networkOutputs);
    syncRequestImpl->setPointerToExecutableNetworkInternal(execNetwork);
    return std::make_shared<BaseAsyncInferRequest>(shared_from_this(),
                                                   syncRequestImpl,
                                                   execNetwork->_callbackExecutor);
}
}  // namespace MultiDevicePlugin
