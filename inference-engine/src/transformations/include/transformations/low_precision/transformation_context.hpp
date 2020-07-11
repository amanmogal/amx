// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include "transformations/low_precision/quantization_details.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

class TRANSFORMATIONS_API TransformationContext {
public:
    explicit TransformationContext(std::shared_ptr<Function> network);

    std::shared_ptr<Function> network;
    std::unordered_set<std::string> quantizedFakeQuantizeNames;
    std::unordered_set<std::string> dequantizationLayersNames;

    inline ngraph::element::Type getOriginalLayerPrecision(const std::string& layer_name, const size_t output_index = 0) {
        const auto& data_map = _original_precisions_map.find(layer_name);
        if (data_map == _original_precisions_map.end())
            return element::undefined;
        if (data_map->second.find(output_index) == data_map->second.end())
            return element::undefined;
        return data_map->second[output_index];
    }

private:
    std::unordered_map<std::string, std::unordered_map<size_t, ngraph::element::Type>> _original_precisions_map;
};

} // namespace low_precision
} // namespace pass
} // namespace ngraph
