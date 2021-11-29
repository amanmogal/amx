// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <list>
#include <string>
#include <utility>

#include <ie_common.h>
#include <legacy/ie_layers.h>
#include "dnn.hpp"

namespace GNAPluginNS {
namespace backend {


struct DnnComponentExtra {
    std::string name;
    intel_dnn_component_t dnnComponent;
    bool isDelayed;
    uint16_t execOrder;
    DnnComponentExtra(std::string name,
                      intel_dnn_component_t dnnComponent,
                      bool isDelayed) :
                      name(name), dnnComponent(dnnComponent), isDelayed(isDelayed) {}
};

/**
 * maps layer name to dnn.component, in topological order, or execution order
 */
struct DnnComponents {
    using storage_type = std::list<DnnComponentExtra>;
    storage_type components;
    /**
     * @brief initializes new empty intel_dnn_component_t object
     * @param layerName - layer name in IR
     * @param layerMetaType - usually either gna of original layer type
     * @return
     */
    intel_dnn_component_t & addComponent(const std::string layerName, const std::string layerMetaType);
    /**
     * @brief returns corresponding dnn layer for topology layer
     * @return
     */
    intel_dnn_component_t * findComponent(InferenceEngine::CNNLayerPtr layer);
    DnnComponentExtra * findFirstComponentWithPtr(const void *ptr);
    /**
     * @brief extract components in execution order
     */
    std::vector<intel_dnn_component_t> getExecutionOrder();

private:
    uint32_t delayedOperations = 0;
};
}  // namespace backend
}  // namespace GNAPluginNS
