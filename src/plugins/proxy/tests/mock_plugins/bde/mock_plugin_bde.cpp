// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mock_plugin_bde.hpp"

#include <iostream>
#include <map>
#include <string>
#include <utility>

#include "description_buffer.hpp"
#include "mock_compiled_model.hpp"
#include "openvino/core/op_extension.hpp"
#include "openvino/runtime/common.hpp"

using namespace std;
using namespace InferenceEngine;
namespace {
bool support_model(const std::shared_ptr<const ov::Model>& model,
                   const InferenceEngine::QueryNetworkResult& supported_ops) {
    for (const auto& op : model->get_ops()) {
        if (supported_ops.supportedLayersMap.find(op->get_friendly_name()) == supported_ops.supportedLayersMap.end())
            return false;
    }
    return true;
}

bool string_to_bool(const std::string& s) {
    return s == "YES";
}
}  // namespace

MockPluginBde::MockPluginBde() {}

void MockPluginBde::SetConfig(const std::map<std::string, std::string>& _config) {
    for (const auto& it : _config) {
        if (it.first == ov::enable_profiling.name())
            m_profiling = string_to_bool(it.second);
        else if (it.first == ov::device::id.name())
            continue;
        else
            throw ov::Exception("BDE set config: " + it.first);
    }
}

Parameter MockPluginBde::GetConfig(const std::string& name,
                                   const std::map<std::string, InferenceEngine::Parameter>& options) const {
    std::string device_id;
    if (options.find(ov::device::id.name()) != options.end()) {
        device_id = options.find(ov::device::id.name())->second.as<std::string>();
    }
    if (name == ov::device::id) {
        return device_id;
    } else if (name == "SUPPORTED_METRICS") {
        std::vector<std::string> metrics;
        metrics.push_back("AVAILABLE_DEVICES");
        metrics.push_back("SUPPORTED_METRICS");
        metrics.push_back(ov::device::uuid.name());
        return metrics;
    } else if (name == "SUPPORTED_CONFIG_KEYS") {
        std::vector<std::string> configs;
        configs.push_back("PERF_COUNT");
        return configs;
    } else if (name == ov::device::uuid) {
        ov::device::UUID uuid;
        for (size_t i = 0; i < uuid.MAX_UUID_SIZE; i++) {
            if (device_id == "bde_b")
                uuid.uuid[i] = i * 2;
            else if (device_id == "bde_d")
                uuid.uuid[i] = i * 4;
            else if (device_id == "bde_e")
                uuid.uuid[i] = i * 5;
        }
        return decltype(ov::device::uuid)::value_type{uuid};
    }

    IE_THROW(NotImplemented) << "BDE config: " << name;
}

Parameter MockPluginBde::GetMetric(const std::string& name,
                                   const std::map<std::string, InferenceEngine::Parameter>& options) const {
    auto RO_property = [](const std::string& propertyName) {
        return ov::PropertyName(propertyName, ov::PropertyMutability::RO);
    };
    auto RW_property = [](const std::string& propertyName) {
        return ov::PropertyName(propertyName, ov::PropertyMutability::RW);
    };

    std::string device_id = GetConfig(ov::device::id.name(), options);

    if (name == ov::supported_properties) {
        std::vector<ov::PropertyName> roProperties{
            RO_property(ov::supported_properties.name()),
            RO_property(ov::available_devices.name()),
            RO_property(ov::device::uuid.name()),
        };
        // the whole config is RW before network is loaded.
        std::vector<ov::PropertyName> rwProperties{
            RW_property(ov::enable_profiling.name()),
        };

        std::vector<ov::PropertyName> supportedProperties;
        supportedProperties.reserve(roProperties.size() + rwProperties.size());
        supportedProperties.insert(supportedProperties.end(), roProperties.begin(), roProperties.end());
        supportedProperties.insert(supportedProperties.end(), rwProperties.begin(), rwProperties.end());

        return decltype(ov::supported_properties)::value_type(supportedProperties);
    } else if (name == "SUPPORTED_METRICS") {
        std::vector<std::string> metrics;
        metrics.push_back("AVAILABLE_DEVICES");
        metrics.push_back("SUPPORTED_METRICS");
        metrics.push_back(ov::device::uuid.name());
        return metrics;
    } else if (name == "PERF_COUNT") {
        return m_profiling;
    } else if (name == "SUPPORTED_CONFIG_KEYS") {
        std::vector<std::string> configs;
        configs.push_back("NUM_STREAMS");
        configs.push_back("PERF_COUNT");
        return configs;
    } else if (name == ov::device::uuid) {
        ov::device::UUID uuid;
        for (size_t i = 0; i < uuid.MAX_UUID_SIZE; i++) {
            if (device_id == "bde_b")
                uuid.uuid[i] = i * 2;
            else if (device_id == "bde_d")
                uuid.uuid[i] = i * 4;
            else if (device_id == "bde_e")
                uuid.uuid[i] = i * 5;
        }
        return decltype(ov::device::uuid)::value_type{uuid};
    } else if (name == ov::available_devices) {
        const std::vector<std::string> availableDevices = {"bde_b", "bde_d", "bde_e"};
        return decltype(ov::available_devices)::value_type(availableDevices);
    } else if (name == ov::device::capabilities) {
        std::vector<std::string> capabilities;
        capabilities.push_back(ov::device::capability::EXPORT_IMPORT);
        return decltype(ov::device::capabilities)::value_type(capabilities);
    }

    IE_THROW(NotImplemented) << "BDE metric: " << name;
}

std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> MockPluginBde::LoadNetwork(
    const CNNNetwork& network,
    const std::map<std::string, std::string>& config) {
    auto model = network.getFunction();

    OPENVINO_ASSERT(model);
    if (!support_model(model, QueryNetwork(network, config)))
        throw ov::Exception("Unsupported model");

    return std::make_shared<MockCompiledModel>(model, config);
}

std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> MockPluginBde::LoadNetwork(
    const CNNNetwork& network,
    const std::map<std::string, std::string>& config,
    const std::shared_ptr<RemoteContext>& context) {
    IE_THROW(NotImplemented);
}

ov::SoPtr<InferenceEngine::IExecutableNetworkInternal> MockPluginBde::LoadNetwork(
    const std::string& modelPath,
    const std::map<std::string, std::string>& config) {
    return InferenceEngine::IInferencePlugin::LoadNetwork(modelPath, config);
}

std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> MockPluginBde::LoadExeNetworkImpl(
    const CNNNetwork& network,
    const std::map<std::string, std::string>& config) {
    auto model = network.getFunction();

    OPENVINO_ASSERT(model);
    if (!support_model(model, QueryNetwork(network, config)))
        throw ov::Exception("Unsupported model");

    return std::make_shared<MockCompiledModel>(model, config);
}

std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> MockPluginBde::ImportNetwork(
    std::istream& networkModel,
    const std::map<std::string, std::string>& config) {
    IE_THROW(NotImplemented);
}

std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> MockPluginBde::ImportNetwork(
    std::istream& networkModel,
    const std::shared_ptr<InferenceEngine::RemoteContext>& context,
    const std::map<std::string, std::string>& config) {
    IE_THROW(NotImplemented);
}

std::shared_ptr<InferenceEngine::RemoteContext> MockPluginBde::GetDefaultContext(
    const InferenceEngine::ParamMap& params) {
    IE_THROW(NotImplemented);
}

InferenceEngine::QueryNetworkResult MockPluginBde::QueryNetwork(
    const InferenceEngine::CNNNetwork& network,
    const std::map<std::string, std::string>& config) const {
    auto model = network.getFunction();

    OPENVINO_ASSERT(model);

    std::unordered_set<std::string> supported_ops = {"Parameter", "Result", "Add", "Constant", "Subtract"};

    InferenceEngine::QueryNetworkResult res;
    for (const auto& op : model->get_ordered_ops()) {
        if (supported_ops.find(op->get_type_info().name) == supported_ops.end())
            continue;
        res.supportedLayersMap.emplace(op->get_friendly_name(), GetName());
    }
    return res;
}

void MockPluginBde::SetCore(std::weak_ptr<InferenceEngine::ICore> core) noexcept {
    InferenceEngine::IInferencePlugin::SetCore(core);
}

void MockPluginBde::SetName(const std::string& name) noexcept {
    InferenceEngine::IInferencePlugin::SetName(name);
}

std::string MockPluginBde::GetName() const noexcept {
    return InferenceEngine::IInferencePlugin::GetName();
}

static const Version version = {{2, 1}, "test_plugin", "MockPluginBde"};
IE_DEFINE_PLUGIN_CREATE_FUNCTION(MockPluginBde, version)
