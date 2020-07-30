// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/unroll_tensor_iterator.hpp"
#include "transformations/utils/utils.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset3.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/graph_util.hpp>
#include <ngraph/specialize_function.hpp>

void ngraph::pass::UnrollTensorIterator::unroll_tensor_iterator() {
    auto tensor_iterator = std::make_shared<ngraph::pattern::op::Label>(ngraph::element::f32,
            ngraph::Shape{}, ngraph::pattern::has_class<ngraph::opset3::TensorIterator>());

    ngraph::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        auto ti = std::dynamic_pointer_cast<ngraph::opset3::TensorIterator>(m.get_match_root());
        if (!ti) {
            return false;
        }

        auto body = ti->get_body();
        const auto function = std::make_shared<ngraph::Function>(body->get_results(),
                ngraph::ParameterVector{body->get_parameters()});

        auto num_iter = ti->get_num_iterations(); // -1 means inconsistent TI
        if (num_iter == -1) return false;
        std::vector<std::shared_ptr<ngraph::Function>> body_functions(num_iter);
        for (uint64_t idx = 0; idx < num_iter; ++idx) {
            // body_functions[idx] = clone_function(*function);

            std::vector<element::Type> paramElementTypes;
            std::vector<PartialShape> paramShapes;
            for (const auto &param : function->get_parameters()) {
                paramElementTypes.emplace_back(param->get_element_type());
                paramShapes.emplace_back(param->get_shape());
            }
            auto inBuffers = std::vector<void *>(function->get_parameters().size());
            body_functions[idx] = specialize_function(function, paramElementTypes, paramShapes,
                                                      inBuffers, false, true);
            for (auto &node : body_functions[idx]->get_ops()) {
                node->set_friendly_name(ti->get_friendly_name() + "/" + std::to_string(idx + 1) + "/" + node->get_friendly_name());
                copy_runtime_info(ti, node);
            }
        }

        // Port map : inputs and back edges
        for (const auto& desc : ti->get_input_descriptions()) {
            std::string type_name = desc->get_type_info().name;

            if (type_name == "SliceInputDescription") {
                auto input_desc = std::dynamic_pointer_cast<ngraph::opset3::TensorIterator::SliceInputDescription>(desc);
                if (!input_desc) {
                    return false;
                }

                auto in_data = ti->input_values()[input_desc->m_input_index];
                const auto const_axis = opset3::Constant::create(element::i64, Shape{}, {input_desc->m_axis});
                auto split = std::make_shared<ngraph::opset3::Split>(in_data, const_axis, num_iter);
                copy_runtime_info(ti, split);
                auto stride = input_desc->m_stride;
                // connect to the body
                for (uint64_t j = 0; j < num_iter; j++) {
                    auto idx = stride > 0? j: num_iter - j - 1;
                    auto param = body_functions[j]->get_parameters()[input_desc->m_body_parameter_index];
                    for (auto &output : param->outputs()) {
                        output.replace(split->output(idx));
                    }
                }
            } else if (type_name == "MergedInputDescription") {
                auto input_desc = std::dynamic_pointer_cast<ngraph::opset3::TensorIterator::MergedInputDescription>(desc);
                if (!input_desc) {
                    return false;
                }

                // connect to the body
                auto in_data = ti->input_values()[input_desc->m_input_index].get_node_shared_ptr();
                auto param = body_functions[0]->get_parameters()[input_desc->m_body_parameter_index];
                for (auto &output : param->outputs()) {
                    output.replace(in_data);
                }

                for (uint64_t j = 1; j < num_iter; j++) {
                    auto cur_param = body_functions[j]->get_parameters()[input_desc->m_body_parameter_index];
                    auto prev_val = body_functions[j - 1]->get_results()[input_desc->m_body_value_index];
                    for (auto &output : cur_param->outputs()) {
                        output.replace(prev_val->get_input_source_output(0));
                    }
                }
            } else if (type_name == "InvariantInputDescription") {
                auto input_desc = std::dynamic_pointer_cast<ngraph::opset3::TensorIterator::InvariantInputDescription>(
                        desc);
                if (!input_desc) {
                    return false;
                }

                // connect to the body
                auto in_data = ti->input_values()[input_desc->m_input_index].get_node_shared_ptr();
                for (uint64_t j = 0; j < num_iter; j++) {
                    auto param = body_functions[j]->get_parameters()[input_desc->m_body_parameter_index];
                    for (auto &output : param->outputs()) {
                        output.replace(in_data);
                    }
                }
            } else {
                // "Incorrect type of the input description.";
                return false;
            }
        }

        // Port map: outputs
        for (const auto& desc : ti->get_output_descriptions()) {
            std::string type_name = desc->get_type_info().name;
            if (type_name == "ConcatOutputDescription") {
                auto output_desc = std::dynamic_pointer_cast<ngraph::opset3::TensorIterator::ConcatOutputDescription>(desc);
                if (!output_desc) {
                    return false;
                }

                ngraph::OutputVector to_concat(num_iter);
                auto stride = output_desc->m_stride;
                for (uint64_t j = 0; j < num_iter; j++) {
                    auto idx = stride > 0? j: num_iter - j - 1;
                    std::shared_ptr<opset3::Result> result = body_functions[idx]->get_results()[output_desc->m_body_value_index];
                    auto input_to_res = result->get_input_source_output(0);
                    to_concat[j] = input_to_res;
                }
                auto concat = std::make_shared<ngraph::opset3::Concat>(to_concat, output_desc->m_axis);
                copy_runtime_info(ti, concat);
                concat->output(0).get_tensor().set_name(op::util::create_ie_output_name(ti->output(output_desc->m_output_index)));
                for (auto &input : ti->output(output_desc->m_output_index).get_target_inputs()) {
                    input.replace_source_output(concat);
                }
            } else if (type_name == "BodyOutputDescription") {
                auto output_desc = std::dynamic_pointer_cast<ngraph::opset3::TensorIterator::BodyOutputDescription>(desc);
                if (!output_desc) {
                    return false;
                }
                auto iter = output_desc->m_iteration;
                iter = iter >= 0? iter: num_iter - 1;
                std::shared_ptr<opset3::Result> result = body_functions[iter]->get_results()[output_desc->m_body_value_index];
                const auto& in_value = result->input_value(0);

                in_value.get_tensor().set_name(op::util::create_ie_output_name(ti->output(output_desc->m_output_index)));
                for (const auto &input : ti->output(output_desc->m_output_index).get_target_inputs()) {
                    input.replace_source_output(result->get_input_source_output(0));
                }
            } else {
                // "Incorrect type of the output description."
                return false;
            }
        }

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(tensor_iterator, "UnrollTensorIterator");
    this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
}