// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include <string>
#include <vector>
#include <iostream>
#include <memory>
#include <utility>
#include <map>
#include <unordered_map>
#include <unordered_set>

#include "ie_metric_helpers.hpp"
#include <cpp_interfaces/interface/ie_internal_plugin_config.hpp>
#include <legacy/ie_util_internal.hpp>
#include <ie_plugin_config.hpp>
#include <ie_icore.hpp>
#include <ie_performance_hints.hpp>
#include "auto_batch.hpp"

namespace AutoBatchPlugin {
    using namespace InferenceEngine;

    template <Precision::ePrecision precision>
    Blob::Ptr create_shared_blob_on_top_of_batched_blob(Blob::Ptr batched_blob, size_t batch_id, size_t batch_num) {
        typedef typename PrecisionTrait<precision>::value_type TYPE;
        typedef typename std::add_pointer<TYPE>::type TYPEPTR;
        auto ptr = batched_blob->buffer().as<TYPEPTR>();
        auto sizePerBatch = batched_blob->size() / batch_num;
        auto layout = batched_blob->getTensorDesc().getLayout();
        SizeVector dims = batched_blob->getTensorDesc().getDims();

        if (layout == InferenceEngine::Layout::NC || layout == InferenceEngine::Layout::NCDHW
            || layout == InferenceEngine::Layout::NCHW || layout == InferenceEngine::Layout::NHWC
            || layout == InferenceEngine::Layout::NDHWC) {
            dims[0] = 1;
            assert(batched_blob->getTensorDesc().getPrecision() == precision);
            return make_shared_blob<TYPE>({precision, dims, batched_blob->getTensorDesc().getLayout()},
                                          ptr + sizePerBatch * batch_id, sizePerBatch);
        } else {
            // same blob for all requests (e.g. constants)
            return make_shared_blob<TYPE>({precision, dims, batched_blob->getTensorDesc().getLayout()},
                                          ptr);
        }
    }

// ------------------------------AutoBatchInferRequest----------------------------
AutoBatchInferRequest::AutoBatchInferRequest(const InputsDataMap&   networkInputs,
                                             const OutputsDataMap&  networkOutputs,
                                             AutoBatchExecutableNetwork::WorkerInferRequest* workerRequestPtr,
                                             int batch_id, int num_batch,
                                             bool needPerfCounters)
        : IInferRequestInternal(networkInputs, networkOutputs), _workerInferRequest(workerRequestPtr),
        _needPerfCounters(needPerfCounters), _batchId(batch_id), _batchSize(num_batch) {
    // Allocate all input blobs
    for (const auto &it : networkInputs) {
        auto blob = workerRequestPtr->_inferRequest->GetBlob(it.first);
        Blob::Ptr res;
        switch (it.second->getTensorDesc().getPrecision()) {
            case InferenceEngine::Precision::FP32:
                res = create_shared_blob_on_top_of_batched_blob<InferenceEngine::Precision::FP32>
                        (workerRequestPtr->_inferRequest->GetBlob(it.first), batch_id, num_batch);
                break;
            case InferenceEngine::Precision::I32:
                res = create_shared_blob_on_top_of_batched_blob<InferenceEngine::Precision::I32>
                        (workerRequestPtr->_inferRequest->GetBlob(it.first), batch_id, num_batch);
                break;
            case InferenceEngine::Precision::I8:
                res = create_shared_blob_on_top_of_batched_blob<InferenceEngine::Precision::I8>
                        (workerRequestPtr->_inferRequest->GetBlob(it.first), batch_id, num_batch);
                break;
            case InferenceEngine::Precision::U16:
                res = create_shared_blob_on_top_of_batched_blob<InferenceEngine::Precision::U16>
                        (workerRequestPtr->_inferRequest->GetBlob(it.first), batch_id, num_batch);
                break;

            case InferenceEngine::Precision::I16:
                res = create_shared_blob_on_top_of_batched_blob<InferenceEngine::Precision::I16>
                        (workerRequestPtr->_inferRequest->GetBlob(it.first), batch_id, num_batch);

                break;
            case InferenceEngine::Precision::U8:
            case InferenceEngine::Precision::BOOL:
                res = create_shared_blob_on_top_of_batched_blob<InferenceEngine::Precision::U8>
                        (workerRequestPtr->_inferRequest->GetBlob(it.first), batch_id, num_batch);
                break;
            default:
                IE_THROW() <<"Unsupported input precision " << it.second->getTensorDesc().getPrecision();
        }
        _inputs[it.first] = res;
    }
    // Allocate all output blobs
    for (const auto &it : networkOutputs) {
        auto blob = workerRequestPtr->_inferRequest->GetBlob(it.first);
        Blob::Ptr res;
        switch (it.second->getTensorDesc().getPrecision()) {
            case InferenceEngine::Precision::FP32:
                res = create_shared_blob_on_top_of_batched_blob<InferenceEngine::Precision::FP32>
                        (workerRequestPtr->_inferRequest->GetBlob(it.first), batch_id, num_batch);
                break;
            case InferenceEngine::Precision::I32:
                res = create_shared_blob_on_top_of_batched_blob<InferenceEngine::Precision::I32>
                        (workerRequestPtr->_inferRequest->GetBlob(it.first), batch_id, num_batch);
                break;
            case InferenceEngine::Precision::I8:
                res = create_shared_blob_on_top_of_batched_blob<InferenceEngine::Precision::I8>
                        (workerRequestPtr->_inferRequest->GetBlob(it.first), batch_id, num_batch);
                break;
            case InferenceEngine::Precision::U16:
                res = create_shared_blob_on_top_of_batched_blob<InferenceEngine::Precision::U16>
                        (workerRequestPtr->_inferRequest->GetBlob(it.first), batch_id, num_batch);
                break;

            case InferenceEngine::Precision::I16:
                res = create_shared_blob_on_top_of_batched_blob<InferenceEngine::Precision::I16>
                        (workerRequestPtr->_inferRequest->GetBlob(it.first), batch_id, num_batch);

                break;
            case InferenceEngine::Precision::U8:
            case InferenceEngine::Precision::BOOL:
                res = create_shared_blob_on_top_of_batched_blob<InferenceEngine::Precision::U8>
                        (workerRequestPtr->_inferRequest->GetBlob(it.first), batch_id, num_batch);
                break;
            default:
                IE_THROW(NotImplemented) << "Unsupported input precision " << it.second->getTensorDesc().getPrecision();
        }
        _outputs[it.first] = res;
    }
}

void AutoBatchInferRequest::SetBlobsToAnotherRequest(SoIInferRequestInternal& req) {
    for (const auto &it : _networkInputs) {
        auto &name = it.first;
        // this request is already in BUSY state, so using the internal functions safely
        auto blob = GetBlob(name);
        if (req->GetBlob(name) != blob)
            req->SetBlob(name, blob);
    }
    for (const auto &it : _networkOutputs) {
        auto &name = it.first;
        // this request is already in BUSY state, so using the internal functions safely
        auto blob = GetBlob(name);
        if (req->GetBlob(name) != blob)
            req->SetBlob(name, blob);
    }
}

void AutoBatchInferRequest::CopyInputsIfNeeded() {
    for (const auto &it : _networkInputs) {
        auto &name = it.first;
        // this request is already in BUSY state, so using the internal functions safely
        CopyBlobIfNeeded(GetBlob(name), _workerInferRequest->_inferRequest->GetBlob(name), true);
    }
}

void AutoBatchInferRequest::CopyBlobIfNeeded(InferenceEngine::Blob::CPtr src, InferenceEngine::Blob::Ptr dst, bool bInput) {
    auto bufferDst = dst->buffer();
    auto ptrDst = bufferDst.as<char*>();
    auto bufferSrc = src->cbuffer();
    auto ptrSrc = bufferSrc.as<const char*>();
    ptrdiff_t szDst = dst->byteSize();
    ptrdiff_t szSrc = src->byteSize();
    if (bInput) {
        ptrdiff_t offset = szSrc != szDst ? _batchId*szDst/_batchSize : 0;
        if (ptrDst - ptrSrc < szDst)
            return;
        else
           memcpy(ptrDst + offset, ptrSrc, src->byteSize());
    } else {
        ptrdiff_t offset = szSrc != szDst ? _batchId*szSrc/_batchSize : 0;
        if (ptrSrc - ptrDst < szDst)
            return;
        else
            memcpy(ptrDst, ptrSrc + offset, dst->byteSize());
    }
    // std::cout << "!!! COPY !!!" << std::endl;
}

void AutoBatchInferRequest::CopyOutputsIfNeeded() {
    for (const auto &it : _networkOutputs) {
        auto &name = it.first;
        // this request is already in BUSY state, so using the internal functions safely
        CopyBlobIfNeeded(_workerInferRequest->_inferRequest->GetBlob(name), GetBlob(name), false);
    }
}

std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> AutoBatchInferRequest::GetPerformanceCounts() const {
    return _perfMap;
}
void AutoBatchInferRequest::InferImpl() {
    std::unique_lock<std::mutex> lock(_workerInferRequest->_mutex);
    int sz = _workerInferRequest->_tasks.unsafe_size();
    if (sz == _workerInferRequest->_batchSize) {
        printf("!!! BATCH : %ld \n", _workerInferRequest->_tasks.unsafe_size());
        std::pair<AutoBatchAsyncInferRequest*, InferenceEngine::Task> t;
        for (int c = 0; c < _workerInferRequest->_batchSize; c++) {
            if (_workerInferRequest->_tasks.try_pop(t)) {
                _workerInferRequest->_completionTasks[c] = std::move(t.second);
                t.first->_inferRequest->CopyInputsIfNeeded();
            } else {
                printf("!!! BUG !!! \n");
            }
        }
        _workerInferRequest->_inferRequest->StartAsync();
    }
}

AutoBatchAsyncInferRequest::AutoBatchAsyncInferRequest(
    const AutoBatchInferRequest::Ptr&           inferRequest,
    const bool                                  needPerfCounters,
    InferenceEngine::SoIInferRequestInternal& inferRequestWithoutBatch,
    const ITaskExecutor::Ptr&                   callbackExecutor) :
    AsyncInferRequestThreadSafeDefault(inferRequest, nullptr, callbackExecutor),
    _inferRequestWithoutBatch(inferRequestWithoutBatch),
    _inferRequest{inferRequest} {
    // this executor starts the inference while  the task (checking the result) is passed to the next stage
    struct ThisRequestExecutor : public ITaskExecutor {
        explicit ThisRequestExecutor(AutoBatchAsyncInferRequest* _this_) : _this{_this_} {}
        void run(Task task) override {
            auto& workerInferRequest = _this->_inferRequest->_workerInferRequest;
            std::pair<AutoBatchAsyncInferRequest*, InferenceEngine::Task> t;
            t.first = _this;
            t.second = std::move(task);
            workerInferRequest->_tasks.push(t);
            _this->_inferRequest->InferImpl();
        };
        AutoBatchAsyncInferRequest* _this = nullptr;
    };
        _pipeline = {
            { /*TaskExecutor*/ std::make_shared<ThisRequestExecutor>(this), /*task*/ [this, needPerfCounters] {
                // TODO: exception checking
                this->_inferRequest->CopyOutputsIfNeeded();
//                if (needPerfCounters)
//                    _inferRequest->_perfMap = _inferRequest->_workerInferRequest->_inferRequest->GetPerformanceCounts();
            }}
    };
  }

void AutoBatchAsyncInferRequest::Infer_ThreadUnsafe() {
    InferUsingAsync();
}

AutoBatchAsyncInferRequest::~AutoBatchAsyncInferRequest() {
    StopAndWait();
}

// ------------------------------AutoBatchExecutableNetwork----------------------------
AutoBatchExecutableNetwork::AutoBatchExecutableNetwork(const InferenceEngine::SoExecutableNetworkInternal& networkForDevice,
        const InferenceEngine::SoExecutableNetworkInternal& networkWithoutBatch,
        const DeviceInformation& networkDevice,
        const std::unordered_map<std::string, InferenceEngine::Parameter>& config,
        const bool needPerfCounters) :
    InferenceEngine::ExecutableNetworkThreadSafeDefault(
            nullptr,
            std::make_shared<InferenceEngine::ImmediateExecutor>()),
    _device{networkDevice},
    _network{networkForDevice},
    _networkWithoutBatch{networkWithoutBatch},
    _config{config},
    _needPerfCounters{needPerfCounters} {
}

AutoBatchExecutableNetwork::~AutoBatchExecutableNetwork() {
//    {
//        std::lock_guard<std::mutex> lock(_mutex);
//        _device = {};
//    }
    _terminate = true;
    /* NOTE: The only threads that use `AutoBatchExecutableNetwork` Context are those that are used by Worker infer requests.
     *       But AsyncInferRequest destructor should waits for all asynchronous tasks that are used by the request
     */
    for (auto w : _workerRequests) {
        w->_thread.join();
    }
    _workerRequests.clear();
}

InferenceEngine::IInferRequestInternal::Ptr AutoBatchExecutableNetwork::CreateInferRequestImpl(InferenceEngine::InputsDataMap networkInputs,
                                                                                                InferenceEngine::OutputsDataMap networkOutputs) {
        // todo : guard request creation from another thread/on-the-fly
        auto num = _numRequestsCreated++;
        auto batch_id = num % _device.batchForDevice;
        if (!batch_id) {  //need new request
            _workerRequests.push_back(std::make_shared<WorkerInferRequest>());
            auto workerRequestPtr = _workerRequests.back();
            workerRequestPtr->_inferRequest = {_network._so, _network->CreateInferRequest()};
            workerRequestPtr->_batchSize = _device.batchForDevice;
            workerRequestPtr->_completionTasks.resize(workerRequestPtr->_batchSize);
            workerRequestPtr->_inferRequest->SetCallback(
                [workerRequestPtr, this] (std::exception_ptr exceptionPtr) mutable {
                    IE_ASSERT(workerRequestPtr->_completionTasks.size() == (size_t)workerRequestPtr->_batchSize);
                    // notify the ibvidual requests on the completion
                    for (int c = 0; c < workerRequestPtr->_batchSize; c++) {
                        workerRequestPtr->_completionTasks[c]();
                    }
                    // reset the timeout
                    workerRequestPtr->_cond.notify_one();
                });

            workerRequestPtr->_thread = std::thread([workerRequestPtr, this] {
                while (!_terminate) {
                    std::unique_lock<std::mutex> lock(workerRequestPtr->_mutex);
                    auto status = workerRequestPtr->_cond.wait_for(lock, std::chrono::milliseconds(100));
                    if (!_terminate && status == std::cv_status::timeout) {
                        // timeout to collect the batch is over, have to execute the requests in the batch1 mode
                        auto sz =  workerRequestPtr->_tasks.unsafe_size();
                        IE_ASSERT(sz < (size_t)_device.batchForDevice);
                        if (sz)
                            std::cout << "TIME_OUT with tasks: " << sz << std::endl;
                        std::pair<AutoBatchAsyncInferRequest*, InferenceEngine::Task> t;
                        // popping all tasks and execute with batch1
                        while (workerRequestPtr->_tasks.try_pop(t)) {
                            t.first->_inferRequestWithoutBatch->SetCallback([t](std::exception_ptr){t.second();});
                            t.first->_inferRequest->SetBlobsToAnotherRequest(t.first->_inferRequestWithoutBatch);
                            t.first->_inferRequestWithoutBatch->StartAsync();
                        }
                    }
                }
            });
       }
    return std::make_shared<AutoBatchInferRequest>(networkInputs, networkOutputs, _workerRequests.back().get(),
            batch_id, _device.batchForDevice, _needPerfCounters);
}

InferenceEngine::IInferRequestInternal::Ptr AutoBatchExecutableNetwork::CreateInferRequest() {
    auto syncRequestImpl = CreateInferRequestImpl(_networkInputs, _networkOutputs);
    syncRequestImpl->setPointerToExecutableNetworkInternal(shared_from_this());
    InferenceEngine::SoIInferRequestInternal inferRequestWithoutBatch = {_networkWithoutBatch._so,
                                                                         _networkWithoutBatch->CreateInferRequest()};
    return std::make_shared<AutoBatchAsyncInferRequest>(std::static_pointer_cast<AutoBatchInferRequest>(syncRequestImpl),
                                                                             _needPerfCounters,
                                                                             inferRequestWithoutBatch,
                                                                             _callbackExecutor);
}

void AutoBatchExecutableNetwork::SetConfig(const std::map<std::string, InferenceEngine::Parameter> &config) {
    // TODO
    IE_THROW(NotImplemented);
}

InferenceEngine::Parameter AutoBatchExecutableNetwork::GetConfig(const std::string &name) const {
    auto res = _config.find(name);
    if (res != _config.end()) {
        return res->second;
    } else {
        IE_THROW(NotFound) << name <<" not found in the ExecutableNetwork config";
    }
}

InferenceEngine::Parameter AutoBatchExecutableNetwork::GetMetric(const std::string &name) const {
    if (name == METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)) {
        auto reqs = 0;
        try {
            auto hint  = _network->GetConfig(CONFIG_KEY(PERFORMANCE_HINT_NUM_REQUESTS)).as<std::string>();
            reqs = InferenceEngine::PerfHintsConfig::CheckPerformanceHintRequestValue(hint);
            if (!reqs) // no limitations from user, let's deduce the full blown #requests
                // (multiplied by the devices capabilities to run multiple <batched> requests for further perf)
                reqs = _device.batchForDevice *
                        _network->GetMetric(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)).as<unsigned int>();
        } catch (const InferenceEngine::Exception &iie) {
        }
        reqs = std::max(reqs, _device.batchForDevice); // round up to the possible  user's value
        IE_SET_METRIC_RETURN(OPTIMAL_NUMBER_OF_INFER_REQUESTS, reqs);
    } else if (name == METRIC_KEY(NETWORK_NAME)) {
        IE_SET_METRIC_RETURN(NETWORK_NAME, _network->GetMetric(
                METRIC_KEY(NETWORK_NAME)).as<std::string>());
    } else if (name == METRIC_KEY(SUPPORTED_METRICS)) {
        IE_SET_METRIC_RETURN(SUPPORTED_METRICS, {
            METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS),
            METRIC_KEY(SUPPORTED_METRICS),
            METRIC_KEY(NETWORK_NAME),
            METRIC_KEY(SUPPORTED_CONFIG_KEYS)
        });
    } else if (name == METRIC_KEY(SUPPORTED_CONFIG_KEYS)) {
        std::vector<std::string> configKeys = { CONFIG_KEY(AUTO_BATCH) };
        IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, configKeys);
    } else {
        IE_THROW() <<"Unsupported Network metric: " << name;
    }
}

// ------------------------------AutoBatchInferencePlugin----------------------------

namespace {

std::map<std::string, std::string> mergeConfigs(std::map<std::string, std::string> config,
                                                const std::map<std::string, std::string> & local) {
    for (auto && kvp : local) {
        config[kvp.first] = kvp.second;
    }
    return config;
}

}  // namespace

std::map<std::string, std::string> AutoBatchInferencePlugin::GetSupportedConfig(
    const std::map<std::string, std::string> & config, const std::string & deviceName) const {
    std::vector<std::string> supportedConfigKeys = GetCore()->GetMetric(deviceName, METRIC_KEY(SUPPORTED_CONFIG_KEYS));
    std::map<std::string, std::string> supportedConfig;
    for (auto&& key : supportedConfigKeys) {
        auto itKey = config.find(key);
        if (config.end() != itKey) {
            supportedConfig[key] = itKey->second;
        }
    }
    return supportedConfig;
}

DeviceInformation AutoBatchInferencePlugin::ParseMetaDevice(const std::string& devicesBatchCfg,
                                                                          const std::map<std::string, std::string> & config) const {
    DeviceInformation metaDevice;
    auto getDeviceConfig = [&] (const DeviceName & deviceWithID) {
        DeviceIDParser deviceParser(deviceWithID);
        std::string deviceName = deviceParser.getDeviceName();
        std::map<std::string, std::string> tconfig = mergeConfigs(_config, config);

        // set device ID if any
        std::string deviceIDLocal = deviceParser.getDeviceID();
        if (!deviceIDLocal.empty()) {
            tconfig[PluginConfigParams::KEY_DEVICE_ID] = deviceIDLocal;
        }

        return GetSupportedConfig(tconfig, deviceName);
    };

    auto && d = devicesBatchCfg;
    {
        auto openingBracket = d.find_first_of('(');
        auto closingBracket = d.find_first_of(')', openingBracket);
        auto deviceName = d.substr(0, openingBracket);

        int batch = -1;
        if (closingBracket != std::string::npos && openingBracket < closingBracket) {
            batch = std::stol(d.substr(openingBracket + 1, closingBracket - 1));

            if (batch <= 0) {
                IE_THROW() <<"Batch value for '" << deviceName << "' must be > 0, while " << batch
                    << "is passed";
            }
        }

        // create meta device
        auto cfg = getDeviceConfig(deviceName);
        metaDevice = { deviceName, cfg, batch };
    }

    return metaDevice;
}

Parameter AutoBatchInferencePlugin::GetConfig(const std::string& name,
        const std::map<std::string, Parameter> & options) const {
    if (name == CONFIG_KEY(AUTO_BATCH)) {
        auto it = _config.find(CONFIG_KEY(AUTO_BATCH));
        if (it == _config.end()) {
            IE_THROW() <<"Value for KEY_AUTO_BATCH is not set";
        } else {
            return { it->second };
        }
    } else {
        IE_THROW() <<"Unsupported config key: " << name;
    }
}

void AutoBatchInferencePlugin::SetConfig(const std::map<std::string, std::string> & config) {
    for (auto && kvp : config) {
        _config[kvp.first] = kvp.second;
    }
}

static const Version version = {{2, 1}, CI_BUILD_NUMBER, "AutoBatchPlugin"};
IE_DEFINE_PLUGIN_CREATE_FUNCTION(AutoBatchInferencePlugin, version)

AutoBatchInferencePlugin::AutoBatchInferencePlugin() {
    _pluginName = "BATCH";
}

InferenceEngine::Parameter AutoBatchInferencePlugin::GetMetric(const std::string& name,
                                         const std::map<std::string, InferenceEngine::Parameter> & options) const {
    if (name == METRIC_KEY(SUPPORTED_METRICS)) {
        std::vector<std::string> metrics;
        metrics.push_back(METRIC_KEY(SUPPORTED_METRICS));
        metrics.push_back(METRIC_KEY(FULL_DEVICE_NAME));
        metrics.push_back(METRIC_KEY(SUPPORTED_CONFIG_KEYS));
        IE_SET_METRIC_RETURN(SUPPORTED_METRICS, metrics);
    } else if (name == METRIC_KEY(FULL_DEVICE_NAME)) {
        IE_SET_METRIC_RETURN(FULL_DEVICE_NAME, _pluginName);
    } else if (name == METRIC_KEY(SUPPORTED_CONFIG_KEYS)) {
        std::vector<std::string> configKeys = PerfHintsConfig::SupportedKeys();
        IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, configKeys);
    } else {
        IE_THROW(NotFound) << "Unsupported metric key " << name;
    }
}

IExecutableNetworkInternal::Ptr AutoBatchInferencePlugin::LoadExeNetworkImpl(const InferenceEngine::CNNNetwork&network,
                                                                              const std::map<std::string, std::string>& config) {
    if (GetCore() == nullptr) {
        IE_THROW() << "Please, work with MULTI device via InferencEngine::Core object";
    }

    auto fullConfig = mergeConfigs(_config, config);
    auto device_batch = fullConfig.find(CONFIG_KEY(AUTO_BATCH));
    if (device_batch == fullConfig.end()) {
        IE_THROW() << "KEY_AUTO_BATCH key is not set for BATCH device";
    }

    auto metaDevice = ParseMetaDevice(device_batch->second, fullConfig);
    const auto &deviceName = metaDevice.deviceName;
    const auto &deviceConfig = metaDevice.config;
    const auto perfConfig = fullConfig.find(PluginConfigParams::KEY_PERF_COUNT);
    const bool enablePerfCounters = (fullConfig.end() != perfConfig) && (perfConfig->second == PluginConfigParams::YES);

    auto networkWithoutBatch = GetCore()->LoadNetwork(network, deviceName, deviceConfig);
    // device settings + auto-batch settings
    std::unordered_map<std::string, InferenceEngine::Parameter> networkConfig;
    networkConfig.insert(*device_batch);
    networkConfig.insert(deviceConfig.begin(), deviceConfig.end());

    const uint64_t total_mem = GetCore()->GetMetric(deviceName, GPU_METRIC_KEY(DEVICE_TOTAL_MEM_SIZE));
    // TODO: remove this experimental code that does loop rather than use the batch1 footprint only
    do {
        CNNNetwork clonedNetwork(InferenceEngine::cloneNetwork(network));
        const InputsDataMap inputInfo = clonedNetwork.getInputsInfo();
        ICNNNetwork::InputShapes shapes = clonedNetwork.getInputShapes();
        for (const InputsDataMap::value_type &item : inputInfo) {
            auto layout = item.second->getTensorDesc().getLayout();
            if (layout == InferenceEngine::Layout::NC || layout == InferenceEngine::Layout::NCDHW
                || layout == InferenceEngine::Layout::NCHW || layout == InferenceEngine::Layout::NHWC
                || layout == InferenceEngine::Layout::NDHWC) {
                shapes[item.first][0] = metaDevice.batchForDevice;
                std::cout << "  reshaping the input " << item.first << " (layout " << layout << ")" << " by the batch"
                          << std::endl;
            }
        }
        std::cout << "Reshaped network by batch to  " << metaDevice.batchForDevice << std::endl;
        clonedNetwork.reshape(shapes);
        auto executableNetworkForDevice = GetCore()->LoadNetwork(CNNNetwork{clonedNetwork}, deviceName, deviceConfig);
        if (executableNetworkForDevice == nullptr)
            IE_THROW(NotFound) << "Failed to load Executable network the device "
                               << "that the BATCH device is initialized to work with";
        uint64_t footprint = executableNetworkForDevice->GetMetric(GPU_METRIC_KEY(NETWORK_MEM_FOOTPRINT));
        std::cout << "!!!!!!!!!!!!!! (BATCHED):" << footprint << std::endl;

        if (footprint < total_mem) {
            return std::make_shared<AutoBatchExecutableNetwork>(executableNetworkForDevice,
                                                                networkWithoutBatch,
                                                                metaDevice,
                                                                networkConfig,
                                                                enablePerfCounters);
        } else { // WA for inaccurate footprint estimations
            std::cout << "WA for inaccurate footprint estimations!!!" << std::endl;
            metaDevice.batchForDevice /= 2;
        }
    } while (1);
}

InferenceEngine::QueryNetworkResult AutoBatchInferencePlugin::QueryNetwork(const InferenceEngine::CNNNetwork& network,
                                              const std::map<std::string, std::string>& config) const {
//    IE_THROW() <<NOT_IMPLEMENTED_str;
    const std::map<std::string, std::string> cfg;
    return GetCore()->QueryNetwork(network, "CPU", cfg);
}
}  // namespace AutoBatchPlugin
