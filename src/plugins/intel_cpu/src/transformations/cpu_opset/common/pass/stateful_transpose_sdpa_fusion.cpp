// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "stateful_transpose_sdpa_fusion.hpp"

#include <cstdint>
#include <limits>
#include <openvino/core/rt_info.hpp>
#include <openvino/opsets/opset1.hpp>
#include <openvino/opsets/opset13.hpp>
#include <openvino/opsets/opset6.hpp>
#include <openvino/opsets/opset8.hpp>
#include <openvino/pass/pattern/op/or.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>
#include <transformations/utils/utils.hpp>

#include "itt.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "transformations/cpu_opset/common/op/sdpa.hpp"

namespace ov {
namespace intel_cpu {

StatefulTransposeSDPAFusion::StatefulTransposeSDPAFusion() {
    MATCHER_SCOPE(StatefulTransposeSDPAFusion);
    using namespace ov::pass::pattern;

    auto past_k = wrap_type<opset6::ReadValue>();
    auto past_v = wrap_type<opset6::ReadValue>();
    auto convert_past_k = wrap_type<opset1::Convert>({past_k});
    auto convert_past_v = wrap_type<opset1::Convert>({past_v});
    auto concat_input_k = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{past_k, convert_past_k});
    auto concat_input_v = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{past_v, convert_past_v});
    auto concat_k = wrap_type<opset6::Concat>({concat_input_k, any_input()});
    auto concat_v = wrap_type<opset6::Concat>({concat_input_v, any_input()});

    // multi-query branch
    auto reshape_k = wrap_type<opset6::Reshape>({concat_k, any_input()});
    auto reshape_v = wrap_type<opset6::Reshape>({concat_v, any_input()});
    auto constant_k = wrap_type<opset6::Constant>();
    auto constant_v = wrap_type<opset6::Constant>();
    auto multiply_k = wrap_type<opset6::Multiply>({reshape_k, constant_k});
    auto multiply_v = wrap_type<opset6::Multiply>({reshape_v, constant_v});
    auto reshape1_k = wrap_type<opset6::Reshape>({multiply_k, any_input()});
    auto reshape1_v = wrap_type<opset6::Reshape>({multiply_v, any_input()});

    auto transpose_k_input = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{reshape1_k, concat_k});
    auto transpose_v_input = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{reshape1_v, concat_v});
    auto order_k = wrap_type<opset6::Constant>();
    auto order_v = wrap_type<opset6::Constant>();
    auto transpose_k = wrap_type<opset6::Transpose>({transpose_k_input, order_k});
    auto transpose_v = wrap_type<opset6::Transpose>({transpose_v_input, order_v});

    auto order_q = wrap_type<opset6::Constant>();
    auto q_input = any_input();
    auto transpose_q = wrap_type<opset6::Transpose>({q_input, order_q});
    auto sdp0 = wrap_type<opset13::ScaledDotProductAttention>({transpose_q, transpose_k, transpose_v});
    auto sdp1 = wrap_type<opset13::ScaledDotProductAttention>({transpose_q, transpose_k, transpose_v, any_input()});
    auto sdp2 = wrap_type<opset13::ScaledDotProductAttention>({transpose_q, transpose_k, transpose_v, any_input(), any_input()});
    auto sdp = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{sdp0, sdp1, sdp2});

    ov::matcher_pass_callback callback = [=](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto root = m.get_match_root();
        auto find_assign = [&](const ov::Output<ov::Node>& out, opset6::Assign*& assign, opset1::Convert*& cvt) {
            auto present_to = out.get_target_inputs();
            if (present_to.size() != 2)
                return;
            for (auto& to : present_to) {
                auto to_node = to.get_node();
                if (auto convert = dynamic_cast<opset1::Convert*>(to_node)) {
                    auto cvt_targets = convert->get_output_target_inputs(0);
                    if (cvt_targets.size() == 1) {
                        to_node = cvt_targets.begin()->get_node();
                        cvt = convert;
                    }
                }
                assign = dynamic_cast<opset6::Assign*>(to_node);
                if (assign)
                    return;
            }
        };

        std::shared_ptr<opset1::Convert> read_cvt_k_node, read_cvt_v_node;
        const auto sdp_node = ov::as_type_ptr<opset13::ScaledDotProductAttention>(root);
        const auto past_k_node = ov::as_type_ptr<opset6::ReadValue>(pattern_map.at(past_k).get_node_shared_ptr());
        const auto past_v_node = ov::as_type_ptr<opset6::ReadValue>(pattern_map.at(past_v).get_node_shared_ptr());
        const auto concat_k_node = ov::as_type_ptr<opset6::Concat>(pattern_map.at(concat_k).get_node_shared_ptr());
        const auto concat_v_node = ov::as_type_ptr<opset6::Concat>(pattern_map.at(concat_v).get_node_shared_ptr());
        if (pattern_map.count(convert_past_k)) {
            read_cvt_k_node = ov::as_type_ptr<opset1::Convert>(pattern_map.at(convert_past_k).get_node_shared_ptr());
            read_cvt_v_node = ov::as_type_ptr<opset1::Convert>(pattern_map.at(convert_past_v).get_node_shared_ptr());
        }

        // check broadcast arg has all ones
        auto check_bcst = [&](const std::shared_ptr<Node>& ptr) {
            const auto constant_node = ov::as_type_ptr<opset6::Constant>(ptr);
            const auto& bcst_arg = constant_node->cast_vector<float>();
            return std::all_of(bcst_arg.begin(), bcst_arg.end(), [](int i) {
                return i == 1.0;
            });
        };

        if (pattern_map.count(constant_k)) {
            if (!check_bcst(pattern_map.at(constant_k).get_node_shared_ptr()))
                return false;
        }

        if (pattern_map.count(constant_v)) {
            if (!check_bcst(pattern_map.at(constant_v).get_node_shared_ptr()))
                return false;
        }

        opset6::Assign* assign_k_node = nullptr, *assign_v_node = nullptr;
        opset1::Convert* assign_cvt_k_node = nullptr, *assign_cvt_v_node = nullptr;
        find_assign(concat_k_node, assign_k_node, assign_cvt_k_node);
        if (!assign_k_node)
            return false;
        if (past_k_node->get_variable_id() != assign_k_node->get_variable_id())
            return false;

        find_assign(concat_v_node, assign_v_node, assign_cvt_v_node);
        if (!assign_v_node)
            return false;
        if (past_v_node->get_variable_id() != assign_v_node->get_variable_id())
            return false;
        auto args = sdp_node->input_values();
        args[0] = pattern_map.at(q_input).get_node_shared_ptr()->output(0);
        args[1] = concat_k_node->input_value(1);
        args[2] = concat_v_node->input_value(1);
        args.push_back(read_cvt_k_node ? read_cvt_k_node->output(0) : past_k_node->output(0));
        args.push_back(read_cvt_v_node ? read_cvt_v_node->output(0) : past_v_node->output(0));
        ov::intel_cpu::ScaledDotProductAttentionWithKVCache::Config config;

        const auto order_q_node = ov::as_type_ptr<opset6::Constant>(pattern_map.at(order_q).get_node_shared_ptr());
        const auto order_k_node = ov::as_type_ptr<opset6::Constant>(pattern_map.at(order_k).get_node_shared_ptr());
        const auto order_v_node = ov::as_type_ptr<opset6::Constant>(pattern_map.at(order_v).get_node_shared_ptr());

        const auto& permute_q = order_q_node->cast_vector<int32_t>();
        const auto& permute_k = order_k_node->cast_vector<int32_t>();
        const auto& permute_v = order_v_node->cast_vector<int32_t>();
        if (permute_q != permute_k || permute_q != permute_v) {
            return false;
        }

        config.is_causal = sdp_node->get_causal();
        config.fuse_concat = true;

        config.permute_axes.resize(permute_q.size());
        for (size_t i = 0; i < permute_q.size(); i++) {
            config.permute_axes[i] = static_cast<size_t>(permute_q[i]);
        }
        auto& old_node = sdp_node;
        auto new_node = std::make_shared<ov::intel_cpu::ScaledDotProductAttentionWithKVCache>(args, config);
        new_node->set_friendly_name(old_node->get_friendly_name());
        ov::replace_node(old_node, {new_node->output(0)});
        if (assign_cvt_k_node)
            assign_cvt_k_node->set_arguments({new_node->output(1)});
        else
            assign_k_node->set_arguments({new_node->output(1)});

        if (assign_cvt_v_node)
            assign_cvt_v_node->set_arguments({new_node->output(2)});
        else
            assign_v_node->set_arguments({new_node->output(2)});

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(sdp, matcher_name);
    this->register_matcher(m, callback);
}

}  // namespace intel_cpu
}  // namespace ov