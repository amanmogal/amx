// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/mark_precision_sensitive_divides.hpp"

#include <memory>
#include <vector>

#include "openvino/op/util/precision_sensitive_attribute.hpp"
#include "openvino/opsets/opset8.hpp"
#include "transformations/rt_info/nonconvertible_divide.hpp"
#include "transformations/utils/utils.hpp"

bool ov::pass::MarkPrecisionSensitiveDivides::run_on_model(const std::shared_ptr<ov::Model>& m) {
    std::deque<std::shared_ptr<Node>> nodes;
    std::unordered_set<std::shared_ptr<Node>> visited, precision_sensitive_visited;
    for (auto& r : m->get_results()) {
        nodes.push_back(r);
        visited.insert(r);
    }
    for (auto& r : m->get_sinks()) {
        nodes.emplace_back(r);
        visited.insert(r);
    }

    auto markup_func = [](std::shared_ptr<Node> node) {
        if (ov::is_type<ov::opset8::Divide>(node) && node->get_output_element_type(0) == ngraph::element::f16) {
            ov::disable_divide_conversion(node);
        }
    };

    while (!nodes.empty()) {
        auto curr_node = nodes.front();
        nodes.pop_front();
        for (auto& input : curr_node->inputs()) {
            if (ov::is_precision_sensitive(input)) {
                visited.insert(input.get_source_output().get_node_shared_ptr());
                // visit_shape_path shouldn't depend on "visited" nodes because we can approach Divide
                // earlier from some non precision sensitive path. So we use dedicated "precision_sensitive_visited"
                // set for precision sensitive nodes, so they can be visited twice and finally marked-up.
                ngraph::op::util::visit_shape_path(input.get_source_output().get_node_shared_ptr(),
                                                   precision_sensitive_visited,
                                                   markup_func);
            }
        }

        for (auto& input_value : curr_node->input_values()) {
            // continue searching
            const auto& input_node = input_value.get_node_shared_ptr();
            if (visited.count(input_node)) continue;
            nodes.push_front(input_node);
            visited.insert(input_node);
        }
    }
    return true;
}
