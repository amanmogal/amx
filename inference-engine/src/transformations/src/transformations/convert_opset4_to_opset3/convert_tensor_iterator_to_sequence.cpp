// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <transformations/convert_opset4_to_opset3/convert_tensor_iterator_to_sequence.hpp>

#include <memory>
#include <vector>

#include <ngraph/node.hpp>
#include <ngraph/pass/manager.hpp>
#include <ngraph/opsets/opset4.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/graph_util.hpp>
#include <ngraph/specialize_function.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>


void ngraph::pass::ConvertTensorIteratorToLSTMSequence::convert_ti_to_lstm_sequence() {
    auto tensor_iterator = std::make_shared<ngraph::pattern::op::Label>(ngraph::element::f32,
                                                                        ngraph::Shape{}, ngraph::pattern::has_class<ngraph::opset4::TensorIterator>());
    ngraph::matcher_pass_callback callback = [this](pattern::Matcher &m) {
        auto ti = std::dynamic_pointer_cast<ngraph::opset4::TensorIterator>(m.get_match_root());
        if (!ti)
            return false;

        // create pattern
        auto data = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::Shape{1, 1, 1});
        auto axis_squeeze = std::make_shared<ngraph::opset4::Constant>(ngraph::element::i64, ngraph::Shape{1}, 1);

        auto input_data = std::make_shared<ngraph::opset4::Squeeze>(data, axis_squeeze);
        auto input_H_state = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::Shape{1, 1});
        auto input_C_state = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::Shape{1, 1});
        auto input_W = std::make_shared<ngraph::opset4::Constant>(ngraph::element::f32, ngraph::Shape{4, 1});
        auto input_R = std::make_shared<ngraph::opset4::Constant>(ngraph::element::f32, ngraph::Shape{4, 1});
        auto input_B = std::make_shared<ngraph::opset4::Constant>(ngraph::element::f32, ngraph::Shape{4});

        auto cell = std::make_shared<ngraph::opset4::LSTMCell>(input_data, input_H_state, input_C_state,
                                                               input_W, input_R, input_B, 1);

        auto axis_unsqueeze = std::make_shared<ngraph::opset4::Constant>(ngraph::element::i64, ngraph::Shape{1}, 1);
        auto unsqueeze = std::make_shared<ngraph::opset4::Unsqueeze>(cell, axis_unsqueeze);
        ngraph::pattern::Matcher matcher(unsqueeze);

        bool flag = false;
        auto func = ti->get_body()->to_function();
        for (const auto& res : func->get_results()) {
            flag = matcher.match((res->get_input_source_output(0)));
            if (flag)
                break;
        }

        // All nodes are in the TI body should be matched in pattern
        if (!flag || (matcher.get_matched_nodes().size() + func->get_results().size()) != func->get_ops().size())
            return false;

        auto seq_lengths = ngraph::opset4::Constant::create(element::i32, Shape{}, {ti->get_num_iterations()});
        auto pattern_map = matcher.get_pattern_map();

        auto params = func->get_parameters();
        std::vector<std::shared_ptr<ngraph::opset4::TensorIterator::InputDescription>> ordered_in_descs(3);
        int64_t stride = 0;
        for (const auto& input_desc : ti->get_input_descriptions()) {
            auto param = params[input_desc->m_body_parameter_index];
            if (param == pattern_map[data]) {
                auto slice_input
                        = std::dynamic_pointer_cast<ngraph::opset4::TensorIterator::SliceInputDescription>(input_desc);
                if (!slice_input)
                    return false;

                stride = slice_input->m_stride;

                ordered_in_descs[0] = input_desc;
            } else if (param == pattern_map[input_H_state]) {
                ordered_in_descs[1] = input_desc;
            } else if (param == pattern_map[input_C_state]) {
                ordered_in_descs[2] = input_desc;
            } else {
                return false;
            }
        }

        auto results = func->get_results();
        std::vector<std::shared_ptr<ngraph::opset4::TensorIterator::OutputDescription>> ordered_out_descs(3);
        for (const auto& output_desc : ti->get_output_descriptions()) {
            std::shared_ptr<opset4::Result> res = results[output_desc->m_body_value_index];
            if (res->get_input_source_output(0) == pattern_map[unsqueeze]) {
                auto concat_output
                        = std::dynamic_pointer_cast<ngraph::opset4::TensorIterator::ConcatOutputDescription>(output_desc);
                if (!concat_output)
                    return false;

                stride = concat_output->m_stride;
                ordered_out_descs[0] = output_desc;
            } else if (res->get_input_source_output(0) == pattern_map[cell]->output(0)) {
                ordered_out_descs[1] = output_desc;
            } else if (res->get_input_source_output(0) == pattern_map[cell]->output(1)) {
                ordered_out_descs[2] = output_desc;
            } else {
                return false;
            }
        }

        const auto& lstm_cell = std::dynamic_pointer_cast<ngraph::opset4::LSTMCell>(pattern_map[cell]);

        auto axis_1 = ngraph::opset4::Constant::create(ngraph::element::i64, ngraph::Shape{}, {1});
        auto in_1 = std::make_shared<ngraph::opset4::Unsqueeze>(ti->input_values()[ordered_in_descs[1]->m_input_index], axis_1);
        auto in_2 = std::make_shared<ngraph::opset4::Unsqueeze>(ti->input_values()[ordered_in_descs[2]->m_input_index], axis_1);

        auto axis_2 = ngraph::opset4::Constant::create(ngraph::element::i64, ngraph::Shape{}, {0});
        auto in_4 = std::make_shared<ngraph::opset4::Unsqueeze>(pattern_map[input_W]->get_output_as_single_output_node(0), axis_2);
        auto in_5 = std::make_shared<ngraph::opset4::Unsqueeze>(pattern_map[input_R]->get_output_as_single_output_node(0), axis_2);
        auto in_6 = std::make_shared<ngraph::opset4::Unsqueeze>(pattern_map[input_B]->get_output_as_single_output_node(0), axis_2);
        auto sequence = std::make_shared<ngraph::opset4::LSTMSequence>(
                ti->input_values()[ordered_in_descs[0]->m_input_index],
                in_1,
                in_2,
                seq_lengths,
                in_4,
                in_5,
                in_6,
                lstm_cell->get_hidden_size(),
                stride > 0 ? ngraph::op::RecurrentSequenceDirection::FORWARD: ngraph::op::RecurrentSequenceDirection::REVERSE,
                lstm_cell->get_weights_format(),
                lstm_cell->get_activations_alpha(),
                lstm_cell->get_activations_beta(),
                lstm_cell->get_activations(),
                lstm_cell->get_clip(),
                lstm_cell->get_input_forget());

        NodeVector new_nodes = {in_1, in_2, in_4, in_5, in_6, sequence};
        copy_runtime_info(ti, new_nodes);
        for (size_t i = 0; i < ordered_out_descs.size(); ++i) {
            if (ordered_out_descs[i]) {
                for (const auto &input : ti->output(ordered_out_descs[i]->m_output_index).get_target_inputs()) {
                    input.replace_source_output(sequence->output(i));
                }
            }
        }

        std::string type = cell->get_type_name();
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(tensor_iterator, "ConvertTensorIteratorToLSTMSequence");
    register_matcher(m, callback);
}

void ngraph::pass::ConvertTensorIteratorToRNNSequence::convert_ti_to_rnn_sequence() {
    auto tensor_iterator = std::make_shared<ngraph::pattern::op::Label>(ngraph::element::f32,
                                                                        ngraph::Shape{}, ngraph::pattern::has_class<ngraph::opset4::TensorIterator>());
    ngraph::matcher_pass_callback callback = [this](pattern::Matcher &m) {
        auto ti = std::dynamic_pointer_cast<ngraph::opset4::TensorIterator>(m.get_match_root());
        if (!ti)
            return false;

        // create pattern
        auto data = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::Shape{1, 1, 1});
        auto axis_squeeze = std::make_shared<ngraph::opset4::Constant>(ngraph::element::i64, ngraph::Shape{1}, 0);
        auto input_data = std::make_shared<ngraph::opset4::Squeeze>(data, axis_squeeze);

        auto input_H_state = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::Shape{1, 1});
        auto input_W = std::make_shared<ngraph::opset4::Constant>(ngraph::element::f32, ngraph::Shape{1, 1});
        auto input_R = std::make_shared<ngraph::opset4::Constant>(ngraph::element::f32, ngraph::Shape{1, 1});
        auto input_B = std::make_shared<ngraph::opset4::Constant>(ngraph::element::f32, ngraph::Shape{1});

        auto cell = std::make_shared<ngraph::opset4::RNNCell>(input_data, input_H_state, input_W, input_R, input_B, 1);

        auto axis_unsqueeze = std::make_shared<ngraph::opset4::Constant>(ngraph::element::i64, ngraph::Shape{1}, 0);
        auto unsqueeze = std::make_shared<ngraph::opset4::Unsqueeze>(cell, axis_unsqueeze);
        ngraph::pattern::Matcher matcher(unsqueeze);

        bool flag = false;
        auto func = ti->get_body()->to_function();
        for (const auto& res : func->get_results()) {
            flag = matcher.match((res->get_input_source_output(0)));
            if (flag)
                break;
        }

        // All nodes are in the TI body should be matched in pattern
        if (!flag || (matcher.get_matched_nodes().size() + func->get_results().size()) != func->get_ops().size())
            return false;

        auto seq_lengths = ngraph::opset4::Constant::create(element::i32, Shape{}, {ti->get_num_iterations()});
        auto pattern_map = matcher.get_pattern_map();

        auto params = func->get_parameters();
        std::vector<std::shared_ptr<ngraph::opset4::TensorIterator::InputDescription>> ordered_in_descs(3);
        int64_t stride = 0;
        for (const auto& input_desc : ti->get_input_descriptions()) {
            auto param = params[input_desc->m_body_parameter_index];
            if (param == pattern_map[data]) {
                auto slice_input
                        = std::dynamic_pointer_cast<ngraph::opset4::TensorIterator::SliceInputDescription>(input_desc);
                if (!slice_input)
                    return false;

                stride = slice_input->m_stride;

                ordered_in_descs[0] = input_desc;
            } else if (param == pattern_map[input_H_state]) {
                ordered_in_descs[1] = input_desc;
            } else {
                return false;
            }
        }

        auto results = func->get_results();
        std::vector<std::shared_ptr<ngraph::opset4::TensorIterator::OutputDescription>> ordered_out_descs(2);
        for (const auto& output_desc : ti->get_output_descriptions()) {
            std::shared_ptr<opset4::Result> res = results[output_desc->m_body_value_index];
            if (res->get_input_source_output(0) == pattern_map[unsqueeze]) {
                auto concat_output
                        = std::dynamic_pointer_cast<ngraph::opset4::TensorIterator::ConcatOutputDescription>(output_desc);
                if (!concat_output)
                    return false;

                stride = concat_output->m_stride;
                ordered_out_descs[0] = output_desc;
            } else if (res->get_input_source_output(0) == pattern_map[cell]->output(0)) {
                ordered_out_descs[1] = output_desc;
            } else {
                return false;
            }
        }

        const auto& rnn_cell = std::dynamic_pointer_cast<ngraph::opset4::RNNCell>(pattern_map[cell]);

        auto axis_1 = ngraph::opset4::Constant::create(ngraph::element::i64, ngraph::Shape{}, {1});
        auto in_1 = std::make_shared<ngraph::opset4::Unsqueeze>(ti->input_values()[ordered_in_descs[1]->m_input_index], axis_1);

        auto axis_2 = ngraph::opset4::Constant::create(ngraph::element::i64, ngraph::Shape{}, {0});
        auto in_3 = std::make_shared<ngraph::opset4::Unsqueeze>(pattern_map[input_W]->get_output_as_single_output_node(0), axis_2);
        auto in_4 = std::make_shared<ngraph::opset4::Unsqueeze>(pattern_map[input_R]->get_output_as_single_output_node(0), axis_2);
        auto in_5 = std::make_shared<ngraph::opset4::Unsqueeze>(pattern_map[input_B]->get_output_as_single_output_node(0), axis_2);
        auto sequence = std::make_shared<ngraph::opset4::RNNSequence>(
                ti->input_values()[ordered_in_descs[0]->m_input_index],
                in_1,
                seq_lengths,
                in_3,
                in_4,
                in_5,
                rnn_cell->get_hidden_size(),
                stride > 0 ? ngraph::op::RecurrentSequenceDirection::FORWARD: ngraph::op::RecurrentSequenceDirection::REVERSE,
                rnn_cell->get_activations(),
                rnn_cell->get_activations_alpha(),
                rnn_cell->get_activations_beta(),
                rnn_cell->get_clip());

        NodeVector new_nodes = {in_1, in_3, in_4, in_5, sequence};
        copy_runtime_info(ti, new_nodes);
        for (size_t i = 0; i < ordered_out_descs.size(); ++i) {
            if (ordered_out_descs[i]) {
                for (const auto &input : ti->output(ordered_out_descs[i]->m_output_index).get_target_inputs()) {
                    input.replace_source_output(sequence->output(i));
                }
            }
        }

        std::string type = cell->get_type_name();
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(tensor_iterator, "ConvertTensorIteratorToRNNSequence");
    register_matcher(m, callback);
}

void ngraph::pass::ConvertTensorIteratorToGRUSequence::convert_ti_to_gru_sequence() {
    auto tensor_iterator = std::make_shared<ngraph::pattern::op::Label>(ngraph::element::f32,
                                                                        ngraph::Shape{}, ngraph::pattern::has_class<ngraph::opset4::TensorIterator>());
    ngraph::matcher_pass_callback callback = [this](pattern::Matcher &m) {
        auto ti = std::dynamic_pointer_cast<ngraph::opset4::TensorIterator>(m.get_match_root());
        if (!ti)
            return false;

        // create pattern
        auto data = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::Shape{1, 1, 1});
        auto axis_squeeze = std::make_shared<ngraph::opset4::Constant>(ngraph::element::i64, ngraph::Shape{1}, 0);
        auto input_data = std::make_shared<ngraph::opset4::Squeeze>(data, axis_squeeze);

        auto input_H_state = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::Shape{1, 1});
        auto input_W = std::make_shared<ngraph::opset4::Constant>(ngraph::element::f32, ngraph::Shape{3, 1});
        auto input_R = std::make_shared<ngraph::opset4::Constant>(ngraph::element::f32, ngraph::Shape{3, 1});
        auto input_B = std::make_shared<ngraph::opset4::Constant>(ngraph::element::f32, ngraph::Shape{3});

        auto cell = std::make_shared<ngraph::opset4::GRUCell>(input_data, input_H_state, input_W, input_R, input_B, 1);

        auto axis_unsqueeze = std::make_shared<ngraph::opset4::Constant>(ngraph::element::i64, ngraph::Shape{1}, 0);
        auto unsqueeze = std::make_shared<ngraph::opset4::Unsqueeze>(cell, axis_unsqueeze);
        ngraph::pattern::Matcher matcher(unsqueeze);

        bool flag = false;
        auto func = ti->get_body()->to_function();
        for (const auto& res : func->get_results()) {
            flag = matcher.match((res->get_input_source_output(0)));
            if (flag)
                break;
        }

        // All nodes are in the TI body should be matched in pattern
        if (!flag || (matcher.get_matched_nodes().size() + func->get_results().size()) != func->get_ops().size())
            return false;

        auto seq_lengths = ngraph::opset4::Constant::create(element::i32, Shape{}, {ti->get_num_iterations()});
        auto pattern_map = matcher.get_pattern_map();

        auto params = func->get_parameters();
        std::vector<std::shared_ptr<ngraph::opset4::TensorIterator::InputDescription>> ordered_in_descs(3);
        int64_t stride = 0;
        for (const auto& input_desc : ti->get_input_descriptions()) {
            auto param = params[input_desc->m_body_parameter_index];
            if (param == pattern_map[data]) {
                auto slice_input
                        = std::dynamic_pointer_cast<ngraph::opset4::TensorIterator::SliceInputDescription>(input_desc);
                if (!slice_input)
                    return false;

                stride = slice_input->m_stride;

                ordered_in_descs[0] = input_desc;
            } else if (param == pattern_map[input_H_state]) {
                ordered_in_descs[1] = input_desc;
            } else {
                return false;
            }
        }

        auto results = func->get_results();
        std::vector<std::shared_ptr<ngraph::opset4::TensorIterator::OutputDescription>> ordered_out_descs(2);
        for (const auto& output_desc : ti->get_output_descriptions()) {
            std::shared_ptr<opset4::Result> res = results[output_desc->m_body_value_index];
            if (res->get_input_source_output(0) == pattern_map[unsqueeze]) {
                auto concat_output
                        = std::dynamic_pointer_cast<ngraph::opset4::TensorIterator::ConcatOutputDescription>(output_desc);
                if (!concat_output)
                    return false;

                stride = concat_output->m_stride;
                ordered_out_descs[0] = output_desc;
            } else if (res->get_input_source_output(0) == pattern_map[cell]->output(0)) {
                ordered_out_descs[1] = output_desc;
            } else {
                return false;
            }
        }

        const auto& rnn_cell = std::dynamic_pointer_cast<ngraph::opset4::GRUCell>(pattern_map[cell]);

        auto axis_1 = ngraph::opset4::Constant::create(ngraph::element::i64, ngraph::Shape{}, {1});
        auto in_1 = std::make_shared<ngraph::opset4::Unsqueeze>(ti->input_values()[ordered_in_descs[1]->m_input_index], axis_1);

        auto axis_2 = ngraph::opset4::Constant::create(ngraph::element::i64, ngraph::Shape{}, {0});
        auto in_3 = std::make_shared<ngraph::opset4::Unsqueeze>(pattern_map[input_W]->get_output_as_single_output_node(0), axis_2);
        auto in_4 = std::make_shared<ngraph::opset4::Unsqueeze>(pattern_map[input_R]->get_output_as_single_output_node(0), axis_2);
        auto in_5 = std::make_shared<ngraph::opset4::Unsqueeze>(pattern_map[input_B]->get_output_as_single_output_node(0), axis_2);
        auto sequence = std::make_shared<ngraph::opset4::GRUSequence>(
                ti->input_values()[ordered_in_descs[0]->m_input_index],
                in_1,
                seq_lengths,
                in_3,
                in_4,
                in_5,
                rnn_cell->get_hidden_size(),
                stride > 0 ? ngraph::op::RecurrentSequenceDirection::FORWARD: ngraph::op::RecurrentSequenceDirection::REVERSE,
                rnn_cell->get_activations(),
                rnn_cell->get_activations_alpha(),
                rnn_cell->get_activations_beta(),
                rnn_cell->get_clip(),
                rnn_cell->get_linear_before_reset());

        NodeVector new_nodes = {in_1, in_3, in_4, in_5, sequence};
        copy_runtime_info(ti, new_nodes);
        for (size_t i = 0; i < ordered_out_descs.size(); ++i) {
            if (ordered_out_descs[i]) {
                for (const auto &input : ti->output(ordered_out_descs[i]->m_output_index).get_target_inputs()) {
                    input.replace_source_output(sequence->output(i));
                }
            }
        }

        std::string type = cell->get_type_name();
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(tensor_iterator, "ConvertTensorIteratorToGRUSequence");
    register_matcher(m, callback);
}