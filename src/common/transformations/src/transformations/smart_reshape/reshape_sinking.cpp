// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset9.hpp>
#include <ngraph/pattern/matcher.hpp>
#include <ngraph/pattern/op/or.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <transformations/smart_reshape/reshape_sinking.hpp>

#include "itt.hpp"

ngraph::pass::ReshapeSinkingMatMul::ReshapeSinkingMatMul() {
    MATCHER_SCOPE(ReshapeSinkingMatMul);
    /*  Original graph:                         Transformed graph:
     *
     * any_input                                any_input
     *     |    shape=[B, S, K]                      |    shape=[B, S, K]
     *  Reshape output_pattern=(-1, K)          Reshape output_pattern=(0, 0, K)
     *     |    shape=[B * S, K]                     |    shape=[B, S, K]
     *  MatMul  constant_shape=[K, O]           MatMul  constant_shape=[K, O]
     *     |    shape=[B * S, O]                     |    shape=[B, S, O]
     *  Reshape output_pattern=(B=1, S, O)      Reshape output_pattern=(0, S, O)
     *     |    shape=[1, S, O]                      |    shape=[B, S, O]
     */
    auto any_input = pattern::any_input(pattern::has_static_rank());
    auto reshape_label = ngraph::pattern::wrap_type<opset9::Reshape>(
        {pattern::any_input(), ngraph::pattern::wrap_type<opset9::Constant>()},
        pattern::rank_equals(2));

    auto matmul_label =
        ngraph::pattern::wrap_type<opset9::MatMul>({reshape_label, ngraph::pattern::wrap_type<opset9::Constant>()},
                                                   pattern::rank_equals(2));
    auto add_label =
        ngraph::pattern::wrap_type<opset9::Add>({matmul_label, ngraph::pattern::wrap_type<opset9::Constant>()},
                                                pattern::rank_equals(2));

    auto matmul_or_matmul_add_label = std::make_shared<pattern::op::Or>(OutputVector{add_label, matmul_label});

    auto reshape_1_label = ngraph::pattern::wrap_type<opset9::Reshape>(
        {matmul_or_matmul_add_label, ngraph::pattern::wrap_type<opset9::Constant>()},
        pattern::has_static_rank());

    matcher_pass_callback callback = [=](pattern::Matcher& m) -> bool {
        auto pattern_to_node = m.get_pattern_map();

        // check first Reshape eligibility: has a constant output pattern in a form of [-1, K]
        auto reshape = pattern_to_node.at(reshape_label);
        int64_t K = -1;
        if (const auto& constant = std::dynamic_pointer_cast<opset9::Constant>(reshape->get_input_node_shared_ptr(1))) {
            auto output_pattern_vector = constant->cast_vector<int64_t>();
            if (output_pattern_vector.size() != 2 || output_pattern_vector[0] != -1)
                return false;
            K = output_pattern_vector[1];
        }
        if (K == -1)
            return false;

        // check input shape eligibility: has a form of [x1, x2, ..., xn, K]
        auto input_pshape = reshape->get_input_partial_shape(0);
        if (input_pshape.rank().is_dynamic() || input_pshape.rank().get_length() <= 2)
            return false;
        auto input_rank = input_pshape.size();
        if (input_pshape[input_rank - 1] != K)
            return false;

        // check matmul eligibility: has constant second input in a form of [O, K]
        auto matmul = std::dynamic_pointer_cast<opset9::MatMul>(pattern_to_node.at(matmul_label));
        if (!matmul || matmul->get_transpose_a())
            return false;
        int64_t O = -1;
        if (const auto& constant = std::dynamic_pointer_cast<opset9::Constant>(matmul->get_input_node_shared_ptr(1))) {
            const auto& constant_shape = constant->get_shape();
            if (constant_shape.size() != 2)
                return false;
            const auto& desired_K_index = matmul->get_transpose_b() ? 1 : 0;
            const auto& O_index = matmul->get_transpose_b() ? 0 : 1;
            if (constant_shape[desired_K_index] != K)
                return false;
            O = static_cast<int64_t>(constant_shape[O_index]);
        }
        if (O == -1)
            return false;

        // check add eligibility if present: has constant second input that has a form of [1, 1, ..., O] (doesn't
        // broadcast first input)
        if (pattern_to_node.count(add_label)) {
            auto add = std::dynamic_pointer_cast<opset9::Add>(pattern_to_node.at(add_label));
            if (!add || add->get_autob() != ov::op::AutoBroadcastType::NUMPY)
                return false;
            const auto& constant = std::dynamic_pointer_cast<opset9::Constant>(add->get_input_node_shared_ptr(1));
            if (!constant)
                return false;
            const auto& constant_shape = constant->get_shape();
            auto desired_ones_shape = ov::Shape(constant_shape.size(), 1);
            auto desired_shape = ov::Shape(constant_shape.size() - 1, 1);
            desired_shape.push_back(O);
            OPENVINO_ASSERT(constant_shape.size() == desired_ones_shape.size() &&
                            constant_shape.size() == desired_shape.size());
            if (constant_shape != desired_shape && constant_shape != desired_ones_shape)
                return false;
        }

        // check second Reshape eligibility: has hard-coded output pattern constant which is almost the same as
        // input_shape of the pattern except for the batch and last dimension
        auto reshape_1 = m.get_match_root();

        const auto& constant = std::dynamic_pointer_cast<opset9::Constant>(reshape_1->get_input_node_shared_ptr(1));
        if (constant == nullptr)
            return false;
        auto output_pattern = constant->cast_vector<int64_t>();
        if (!std::all_of(output_pattern.begin(), output_pattern.end(), [](const int64_t& i) {
                return i > 0;
            }))
            return false;
        if (output_pattern.size() != input_rank)
            return false;
        for (size_t i = 0; i < input_rank; ++i) {
            if (i == 0)
                continue;
            if (i + 1 == input_rank) {
                if (output_pattern[i] != O)
                    return false;
                else
                    continue;
            }
            if (input_pshape[i] != output_pattern[i])
                return false;
        }

        // this is the pattern we are looking for! performing the transformation
        auto first_reshape = std::dynamic_pointer_cast<opset9::Reshape>(reshape);
        if (!first_reshape)
            return false;
        first_reshape->set_special_zero(true);
        auto second_reshape = std::dynamic_pointer_cast<opset9::Reshape>(reshape_1);
        if (!second_reshape)
            return false;
        second_reshape->set_special_zero(true);

        std::vector<int64_t> output_pattern_vector(input_rank - 1, 0);
        output_pattern_vector.push_back(K);
        auto new_reshape_constant =
            opset9::Constant::create(ov::element::i64, Shape{input_rank}, output_pattern_vector);
        reshape->input(1).replace_source_output(new_reshape_constant->output(0));

        output_pattern[0] = 0;
        auto new_reshape_1_constant = opset9::Constant::create(ov::element::i64, Shape{input_rank}, output_pattern);
        reshape_1->input(1).replace_source_output(new_reshape_1_constant->output(0));

        return true;
    };
    auto m = std::make_shared<ngraph::pattern::Matcher>(reshape_1_label, matcher_name);
    register_matcher(m, callback);
}
