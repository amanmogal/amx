// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "config.hpp"
#include "plugin.hpp"
#include "openvino/runtime/icompiled_model.hpp"
// #include "openvino/runtime/iinfer_request.hpp"
#include "openvino/runtime/so_ptr.hpp"
#include "openvino/runtime/isync_infer_request.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace hetero {

class Plugin;

/**
 * @class CompiledModel
 * @brief Implementation of compiled model
 */
class CompiledModel : public ov::ICompiledModel {
public:
    CompiledModel(const std::shared_ptr<ov::Model>& model,
                  const std::shared_ptr<const ov::IPlugin>& plugin,
                  const Configuration& cfg,
                  bool loaded_from_cache = false);

    CompiledModel(std::istream& model,
                  const std::shared_ptr<const ov::IPlugin>& plugin,
                  const Configuration& cfg,
                  bool loaded_from_cache = false);

    // Methods from a base class ov::ICompiledModel
    void export_model(std::ostream& model) const override;

    std::shared_ptr<const ov::Model> get_runtime_model() const override;

    void set_property(const ov::AnyMap& properties) override;

    ov::Any get_property(const std::string& name) const override;

    std::shared_ptr<ov::IAsyncInferRequest> create_infer_request() const override;

    // TODO (vurusovs) to be changed with more robust solution with InferRequest implementation
    std::unordered_map<std::string, std::string> _blobNameMap;

protected:
    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override;

private:
    friend class Plugin;

    std::shared_ptr<const Plugin> get_hetero_plugin() const;

    Configuration m_cfg;
    std::shared_ptr<ov::Model> m_model;
    const bool m_loaded_from_cache;

    struct NetworkDesc {
        std::string _device;
        std::shared_ptr<ov::Model> _clonedNetwork;
        ov::SoPtr<ov::ICompiledModel> _network;
    };

    std::vector<NetworkDesc> m_networks;
    std::string m_name;

    
};
}  // namespace hetero
}  // namespace ov