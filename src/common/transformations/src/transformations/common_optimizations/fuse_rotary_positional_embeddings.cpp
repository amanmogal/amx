// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/fuse_rotary_positional_embeddings.hpp"

#include <openvino/core/validation_util.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/op.hpp>
#include <openvino/op/variadic_split.hpp>
#include <openvino/pass/pattern/op/pattern.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>
#include <ov_ops/rotary_positional_embeddings.hpp>

#include "itt.hpp"

ov::pass::pattern::op::ValuePredicate constant_predicate(
    const std::function<bool(const std::vector<int64_t>&)>& predicate) {
    return ov::pass::pattern::op::as_value_predicate([&](std::shared_ptr<ov::Node> n) -> bool {
        if (const auto& constant = ov::as_type_ptr<ov::op::v0::Constant>(n)) {
            auto values = constant->cast_vector<int64_t>();
            return predicate(values);
        }
        return false;
    });
}

#define CONSTANT_WITH_PREDICATE(expression)                                                                 \
    pattern::wrap_type<op::v0::Constant>(constant_predicate([](const std::vector<int64_t>& value) -> bool { \
        return expression;                                                                                  \
    }))

ov::pass::RPE_Fusion::RPE_Fusion() {
    MATCHER_SCOPE(RPE_Fusion);

    auto sin = pattern::any_input();
    auto cos = pattern::any_input();

    // FIXME: should be a single node match
    auto source_1 = pattern::any_input();
    auto source = pattern::any_input();
    // BEGIN: rotate_half

    // Variadic Split into two equal parts
    auto axis = pattern::any_input();
    auto split_length = CONSTANT_WITH_PREDICATE(value.size() == 2 && value[0] == value[1]);
    auto vsplit = pattern::wrap_type<op::v1::VariadicSplit>({source, axis, split_length});
    vsplit->set_output_size(2);

    // Negate
    auto minus_1 = CONSTANT_WITH_PREDICATE(value.size() == 1 && value[0] == -1);
    auto neg = pattern::wrap_type<op::v1::Multiply>({vsplit->output(1), minus_1});

    // Concat two splitted parts in the opposite order, first of them is negated
    auto concat = pattern::wrap_type<op::v0::Concat>({neg, vsplit->output(0)});  // make sure axis eq to vsplit eq -1

    // END: rotate half

    auto mul_sin = pattern::wrap_type<op::v1::Multiply>({concat, sin});
    auto mul_cos = pattern::wrap_type<op::v1::Multiply>({source_1, cos});
    auto add = pattern::wrap_type<op::v1::Add>({mul_cos, mul_sin});

    ov::matcher_pass_callback matcher_pass_callback = [=](pattern::Matcher& m) {
        auto value_map = m.get_pattern_value_map();

        auto actual_source = value_map.at(vsplit).get_node_shared_ptr()->input_value(0);
        auto potential_source = value_map.at(mul_cos).get_node_shared_ptr()->input_value(0);
        auto cos_output = value_map.at(mul_cos).get_node_shared_ptr()->input_value(1);

        if (actual_source != potential_source && actual_source != cos_output)
            return false;  // flawed match
        if (actual_source == potential_source && actual_source == cos_output)
            return false;  // flawed match
        if (actual_source != potential_source && actual_source == cos_output)
            cos_output = potential_source;

        auto input = value_map.at(source);
        auto concat_node = ov::as_type_ptr<op::v0::Concat>(value_map.at(concat).get_node_shared_ptr());
        if (!concat_node)
            return false;
        OPENVINO_SUPPRESS_DEPRECATED_START
        auto split_axis_node = ov::get_constant_from_source(value_map.at(axis));
        OPENVINO_SUPPRESS_DEPRECATED_END
        if (!split_axis_node)
            return false;
        auto value = split_axis_node->cast_vector<int64_t>();
        if (value.size() != 1)
            return false;
        auto concat_axis = concat_node->get_concatenation_axis();
        auto split_axis = value[0];
        if (concat_axis != split_axis) {
            if (input.get_partial_shape().is_static()) {
                auto rank = input.get_partial_shape().rank().get_length();
                concat_axis = concat_axis < 0 ? concat_axis + rank : concat_axis;
                split_axis = split_axis < 0 ? split_axis + rank : split_axis;
            }
            if (concat_axis != split_axis)
                return false;
        }
        auto rope =
            std::make_shared<ov::op::internal::RPE>(input, value_map.at(sin), cos_output, concat_node->get_axis());
        ov::replace_output_update_name(value_map.at(add), rope->output(0));
        return true;
    };
    auto m = std::make_shared<pattern::Matcher>(add, matcher_name);
    register_matcher(m, matcher_pass_callback);
}
