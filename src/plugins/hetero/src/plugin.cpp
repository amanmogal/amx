// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include <memory>
#include <vector>
#include <map>
#include <string>
#include <utility>
#include <fstream>
#include <unordered_set>

// LEGACY
#include "ie_metric_helpers.hpp"
// LEGACY

#include "openvino/runtime/device_id_parser.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/util/common_util.hpp"

#include "plugin.hpp"
// #include "compiled_model.hpp"
#include "itt.hpp"
// #include "properties.hpp"

// TODO (vurusovs) required for conversion to legacy API 1.0
#include "converter_utils.hpp"
#include "plugin.hpp"
#include "../executable_network.hpp"
#include "../internal_properties.hpp"

#include "iplugin_wrapper.hpp"
// TODO (vurusovs) required for conversion to legacy API 1.0


ov::hetero::Plugin::Plugin() {
    set_device_name("HETERO");
}

std::shared_ptr<ov::ICompiledModel> ov::hetero::Plugin::compile_model(
    const std::shared_ptr<const ov::Model>& model,
    const ov::AnyMap& properties) const {
    OV_ITT_SCOPED_TASK(itt::domains::Hetero, "Plugin::compile_model");

    // auto fullConfig = ov::hetero::Configuration{properties, m_cfg};
    // auto compiled_model = std::make_shared<CompiledModel>(
    //     model->clone(),
    //     shared_from_this(),
    //     fullConfig);
    // return compiled_model;
    
    auto shared_this = std::const_pointer_cast<ov::IPlugin>(shared_from_this());
    auto plugin_p = ov::legacy_convert::convert_plugin(shared_this);
    auto network = ov::legacy_convert::convert_model(model, this->is_new_api());
    auto legacy_compiled_model = std::make_shared<HeteroPlugin::HeteroExecutableNetwork>(network, properties, std::dynamic_pointer_cast<ov::hetero::Plugin>(shared_this));
    legacy_compiled_model->SetPointerToPlugin(plugin_p);
    legacy_compiled_model->setNetworkInputs(InferenceEngine::copyInfo(network.getInputsInfo()));
    legacy_compiled_model->setNetworkOutputs(InferenceEngine::copyInfo(network.getOutputsInfo()));
    InferenceEngine::SetExeNetworkInfo(legacy_compiled_model, model, this->is_new_api());
    auto compiled_model = ov::legacy_convert::convert_compiled_model(legacy_compiled_model);
    return compiled_model;

}

std::shared_ptr<ov::ICompiledModel> ov::hetero::Plugin::compile_model(const std::shared_ptr<const ov::Model>& model,
    const ov::AnyMap& properties,
    const ov::RemoteContext& context) const {
    OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<ov::ICompiledModel> ov::hetero::Plugin::import_model(std::istream& model,
    const ov::RemoteContext& context,
    const ov::AnyMap& properties) const  {
    OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<ov::ICompiledModel> ov::hetero::Plugin::import_model(std::istream& model,
                                                                     const ov::AnyMap& properties) const {
    OV_ITT_SCOPED_TASK(itt::domains::Hetero, "Plugin::import_model");


    auto shared_this = std::const_pointer_cast<ov::IPlugin>(shared_from_this());
    auto plugin_p = ov::legacy_convert::convert_plugin(shared_this);
    auto legacy_compiled_model = std::make_shared<HeteroPlugin::HeteroExecutableNetwork>(model, properties, std::dynamic_pointer_cast<ov::hetero::Plugin>(shared_this), true);
    legacy_compiled_model->SetPointerToPlugin(plugin_p);
    // legacy_compiled_model->setNetworkInputs(InferenceEngine::copyInfo(network.getInputsInfo()));
    // legacy_compiled_model->setNetworkOutputs(InferenceEngine::copyInfo(network.getOutputsInfo()));
    // InferenceEngine::SetExeNetworkInfo(legacy_compiled_model, model, this->is_new_api());

    auto compiled_model = ov::legacy_convert::convert_compiled_model(legacy_compiled_model);
    return compiled_model;
}

ov::hetero::Plugin::DeviceProperties ov::hetero::Plugin::get_device_properties(const std::string& device_priorities,
                                                                               const ov::AnyMap& properties) const {
    auto device_names = ov::DeviceIDParser::get_hetero_devices(device_priorities);
    DeviceProperties device_properties;
    for (auto&& device_name : device_names) {
        auto properties_it = device_properties.find(device_name);
        if (device_properties.end() == properties_it) {
            device_properties[device_name] = get_core()->get_supported_property(device_name, properties);
        }
    }
    return device_properties;
}

ov::SupportedOpsMap ov::hetero::Plugin::query_model(const std::shared_ptr<const ov::Model>& model,
                                                           const ov::AnyMap& properties) const {
    OV_ITT_SCOPED_TASK(itt::domains::Hetero, "Plugin::query_model");

    Configuration fullConfig{properties, m_cfg};
    
    OPENVINO_ASSERT(model, "OpenVINO Model is empty!");

    std::string fallbackDevicesStr = fullConfig.device_priorities;
    
    DeviceProperties metaDevices = get_device_properties(fallbackDevicesStr, fullConfig.GetDeviceConfig());

    std::map<std::string, ov::SupportedOpsMap> queryResults;
    for (auto&& metaDevice : metaDevices) {
        const auto& deviceName = metaDevice.first;
        const auto& device_config = metaDevice.second;
        queryResults[deviceName] = get_core()->query_model(model, deviceName, device_config);
    }

    //  WARNING: Here is devices with user set priority
    auto fallbackDevices = ov::DeviceIDParser::get_hetero_devices(fallbackDevicesStr);

    ov::SupportedOpsMap res;
    for (auto&& deviceName : fallbackDevices) {
        for (auto&& layerQueryResult : queryResults[deviceName]) {
            res.emplace(layerQueryResult);
        }
    }

    return res;
}

void ov::hetero::Plugin::set_property(const ov::AnyMap& properties) {
    m_cfg = Configuration{properties, m_cfg};
}

ov::Any ov::hetero::Plugin::get_property(const std::string& name, const ov::AnyMap& arguments) const {
    const auto& add_ro_properties = [](const std::string& name, std::vector<ov::PropertyName>& properties) {
        properties.emplace_back(ov::PropertyName{name, ov::PropertyMutability::RO});
    };

    const auto& default_ro_properties = []() {
        std::vector<ov::PropertyName> ro_properties{ov::supported_properties,
                                                    ov::device::full_name,
                                                    ov::device::capabilities,
                                                    ov::caching_properties};
                                                    //ov::available_devices,
                                                    //ov::device::architecture,
                                                    //ov::range_for_async_infer_requests
        return ro_properties;
    };
    const auto& default_rw_properties = []() {
        std::vector<ov::PropertyName> rw_properties{ov::device::priorities};
                                                    //ov::device::id,
                                                    //ov::enable_profiling,
                                                    //ov::hint::performance_mode,
                                                    //ov::exclusive_async_requests,
        return rw_properties;
    };
    const auto& to_string_vector = [](const std::vector<ov::PropertyName>& properties) {
        std::vector<std::string> ret;
        for (const auto& property : properties) {
            ret.emplace_back(property);
        }
        return ret;
    };
    if (METRIC_KEY(SUPPORTED_METRICS) == name) {
        auto metrics = default_ro_properties();

        add_ro_properties(METRIC_KEY(SUPPORTED_METRICS), metrics);
        add_ro_properties(METRIC_KEY(SUPPORTED_CONFIG_KEYS), metrics);
        add_ro_properties(METRIC_KEY(IMPORT_EXPORT_SUPPORT), metrics);
        return to_string_vector(metrics);
        // IE_SET_METRIC_RETURN(SUPPORTED_METRICS,
        //                     // TODO: check list
        //                      std::vector<std::string>{METRIC_KEY(SUPPORTED_METRICS),
        //                                               ov::device::full_name.name(),
        //                                               METRIC_KEY(SUPPORTED_CONFIG_KEYS),
        //                                               METRIC_KEY(IMPORT_EXPORT_SUPPORT),
        //                                               ov::caching_properties.name(),
        //                                               ov::device::capabilities.name()});
    } else if (METRIC_KEY(SUPPORTED_CONFIG_KEYS) == name) {
        return to_string_vector(default_rw_properties());
        // IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, getSupportedConfigKeys());
    } else if (ov::supported_properties == name) {
        auto ro_properties = default_ro_properties();
        auto rw_properties = default_rw_properties();

        std::vector<ov::PropertyName> supported_properties;
        supported_properties.reserve(ro_properties.size() + rw_properties.size());
        supported_properties.insert(supported_properties.end(), ro_properties.begin(), ro_properties.end());
        supported_properties.insert(supported_properties.end(), rw_properties.begin(), rw_properties.end());
        return decltype(ov::supported_properties)::value_type(supported_properties);
    } else if (ov::device::full_name == name) {
        return decltype(ov::device::full_name)::value_type{"HETERO"};
    } else if (METRIC_KEY(IMPORT_EXPORT_SUPPORT) == name) {
        return true;
    } else if (ov::caching_properties == name) {
        // TODO vurusovs: RECHECK WITH ov::hetero::caching_device_properties
        return decltype(ov::caching_properties)::value_type{ov::hetero::caching_device_properties.name()};
    } else if (ov::hetero::caching_device_properties == name) {
        // std::string targetFallback = GetTargetFallback(user_options);
        // it = hetero_config.find(ov::device::priorities.name());

        // TODO vurusovs: CHECK `target_fallback` is empty or not
        // TODO vurusovs: RECHECK WITH ov::caching_properties
        auto target_fallback = m_cfg.device_priorities;
        return decltype(ov::hetero::caching_device_properties)::value_type{DeviceCachingProperties(target_fallback)};
    } else if (ov::device::capabilities == name) {
        return decltype(ov::device::capabilities)::value_type{{ov::device::capability::EXPORT_IMPORT}};
    } else {
        return m_cfg.Get(name);
    }
}

std::string ov::hetero::Plugin::DeviceCachingProperties(const std::string& targetFallback) const {
    // TODO: CHECK FUNCTION WORKS CORRECTLY

    auto fallbackDevices = ov::DeviceIDParser::get_hetero_devices(targetFallback);
    // Vector of caching configs for devices
    std::vector<ov::AnyMap> result = {};
    for (const auto& device : fallbackDevices) {
        ov::DeviceIDParser parser(device);
        ov::AnyMap properties = {};
        // Use name without id
        auto device_name = parser.get_device_name();
        auto supported_properties =
            get_core()->get_property(device, ov::supported_properties);
        if (ov::util::contains(supported_properties, ov::caching_properties)) {
            auto caching_properties =
                get_core()->get_property(device, ov::caching_properties);
            for (auto& property_name : caching_properties) {
                properties[property_name] = get_core()->get_property(device, std::string(property_name), {});
            }
            // If caching properties are not supported by device, try to add at least device architecture
        } else if (ov::util::contains(supported_properties, ov::device::architecture)) {
            auto device_architecture = get_core()->get_property(device, ov::device::architecture);
            properties = ov::AnyMap{{ov::device::architecture.name(), device_architecture}};
            // Device architecture is not supported, add device name as achitecture
        } else {
            properties = ov::AnyMap{{ov::device::architecture.name(), device_name}};
        }
        result.emplace_back(properties);
    }
    return result.empty() ? "" : ov::Any(result).as<std::string>();
}


std::shared_ptr<ov::IRemoteContext> ov::hetero::Plugin::create_context(const ov::AnyMap& remote_properties) const {
    OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<ov::IRemoteContext> ov::hetero::Plugin::get_default_context(const ov::AnyMap& remote_properties) const {
    OPENVINO_NOT_IMPLEMENTED;
}

static const ov::Version version = {CI_BUILD_NUMBER, "openvino_hetero_plugin"};
OV_DEFINE_PLUGIN_CREATE_FUNCTION(ov::hetero::Plugin, version)