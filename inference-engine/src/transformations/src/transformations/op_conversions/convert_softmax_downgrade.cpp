// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_softmax_downgrade.hpp"
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include "itt.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(pass::ConvertSoftmax8ToSoftmax1, "ConvertSoftmax8ToSoftmax1", 0);

pass::ConvertSoftmax8ToSoftmax1::ConvertSoftmax8ToSoftmax1() {
    MATCHER_SCOPE(ConvertSoftmax8ToSoftmax1);

    auto softmax_v8_pattern = pattern::wrap_type<opset8::Softmax>();

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto softmax_v8_node = dynamic_pointer_cast<opset8::Softmax>(m.get_match_root());
        if (!softmax_v8_node)
            return false;

        auto v8_axis = softmax_v8_node->get_axis();
        size_t rank = static_cast<size_t>(softmax_v8_node->get_input_partial_shape(0).rank().get_length());
        size_t v1_axis = v8_axis < 0 ? static_cast<size_t>(rank + v8_axis) : static_cast<size_t>(v8_axis);

        auto softmax_v1_node = make_shared<opset1::Softmax>(softmax_v8_node->input_value(0), v1_axis);
        softmax_v1_node->set_friendly_name(softmax_v8_node->get_friendly_name());
        copy_runtime_info(softmax_v8_node, softmax_v1_node);
        replace_node(softmax_v8_node, softmax_v1_node);

        return true;
    };

    auto m = make_shared<pattern::Matcher>(softmax_v8_pattern, matcher_name);
    register_matcher(m, callback);
}
