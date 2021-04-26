// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/if.hpp"
#include <ngraph/validation_util.hpp>
#include "itt.hpp"
#include "ngraph/factory.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/specialize_function.hpp"
#include "ngraph/op/util/multi_subgraph_base.hpp"
#include <algorithm>
#include <iterator>

#include "ngraph/runtime/reference/if.hpp"

using namespace std;
using namespace ngraph;
constexpr NodeTypeInfo op::v0::If::type_info;


op::v0::If::If()
    : If(OutputVector())
{
}

op::v0::If::If(const OutputVector& values)
    : op::util::MultiSubGraphOp(values, 2)
{
    m_bodies.resize(2);
    m_input_descriptions.resize(2);
    m_output_descriptions.resize(2);
}


op::v0::If::If(const Output<Node>& execution_condition): If() {
    set_argument(0, execution_condition);
  
}

bool op::v0::If::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v0_If_visit_attributes);
    if (m_bodies.size() != 2) {
        m_bodies.resize(2);
        m_input_descriptions.resize(2);
        m_output_descriptions.resize(2);
    }
    m_bodies[then_body_index] = std::make_shared<ngraph::Function>(OutputVector{}, ParameterVector{}, "then_branch");
    m_bodies[else_body_index] = std::make_shared<ngraph::Function>(OutputVector{}, ParameterVector{}, "else_branch");
    visitor.on_attribute("then_body", m_bodies[then_body_index]);
    visitor.on_attribute("else_body", m_bodies[else_body_index]);
    visitor.on_attribute("then_inputs", m_input_descriptions[then_body_index]);
    visitor.on_attribute("else_inputs", m_input_descriptions[else_body_index]);
    visitor.on_attribute("then_outputs", m_output_descriptions[then_body_index]);
    visitor.on_attribute("else_outputs", m_output_descriptions[else_body_index]);
    return true;
}

ngraph::Rank resolve_dynamic_rank(ngraph::Output<ngraph::Node>& then_node, ngraph::Output<ngraph::Node> else_node)
{
    //TODO: Resolve dynamic ranks
    return ngraph::Rank();
}

ngraph::Output<Node> op::v0::If::set_output(ngraph::Output<ngraph::Node> then_output,
                 ngraph::Output<ngraph::Node> else_output)
{
    auto output_index = get_output_size();
    m_output_descriptions[then_body_index].push_back(std::make_shared<BodyOutputDescription>(
        m_bodies[then_body_index]->get_result_index(then_output), output_index));
    m_output_descriptions[else_body_index].push_back(std::make_shared<BodyOutputDescription>(
        m_bodies[else_body_index]->get_result_index(else_output), output_index));
    set_output_size(output_index + 1);
    validate_and_infer_types();
    return ngraph::Output<Node>(shared_from_this(), output_index);
}

void op::v0::If::validate_and_infer_type_body(std::shared_ptr<Function> body,
                                  ngraph::op::util::MultiSubgraphInputDescriptionVector& input_descriptors)
{
    for (const auto& input_description : input_descriptors)
    {
        auto index = input_description->m_input_index;
        auto body_parameter = body->get_parameters().at(input_description->m_body_parameter_index);
        auto input_partial_shape = inputs().at(index).get_source_output().get_partial_shape();
        if (input_partial_shape.is_static()) 
        {
            auto input_shape = input_partial_shape.to_shape();
            Shape out_shape{input_shape};
            body_parameter->set_partial_shape(out_shape);
        }
        else
        {
            body_parameter->set_partial_shape(PartialShape::dynamic(input_partial_shape.rank()));
        }

    }
    body->validate_nodes_and_infer_types();
}
void op::v0::If::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v0_If_validate_and_infer_types);
  //  NODE_VALIDATION_CHECK(this,
 //                         get_input_size() == m_input_descriptions.size(),
//                          "Number of inputs must be the same as number of input descriptions");
    auto cond_output = inputs().at(0).get_source_output();

    auto cond_partial_shape = cond_output.get_partial_shape();
    auto cond_rank = cond_partial_shape.rank();
    if (cond_rank.is_static()) 
    {
        NODE_VALIDATION_CHECK(this, cond_rank.get_length()<2, "Incorrect condition");
    }
    /* TODO: add dynamic rank implementation*/
    if (cond_partial_shape.is_static()) 
    {
        auto cond_shape = cond_partial_shape.to_shape();
        if (cond_rank.get_length() == 1) 
        {
            NODE_VALIDATION_CHECK(this, cond_shape.at(0)==1, "Incorrect shape of condition");
        }
    }
    auto cond_type = cond_output.get_element_type();
    NODE_VALIDATION_CHECK(this, cond_type == ngraph::element::boolean, "Incorrect shape of condition");
    //Trying to get cond as const value
    if (const auto& cond_value = get_constant_from_source(cond_output)) {
        auto val = cond_value->cast_vector<bool>();
        int cond_index = val[0] ? then_body_index : else_body_index;
        auto body = m_bodies[cond_index];
        auto input_descriptors = m_input_descriptions[cond_index];
        validate_and_infer_type_body(body, input_descriptors);
        auto output_nodes = outputs();
        for (auto output_descr : m_output_descriptions[cond_index]) 
        {
            auto body_value = body->get_results().at(output_descr->m_body_value_index)->input_value(0);
            auto body_value_partial_shape = body_value.get_partial_shape();
            set_output_type(output_descr->m_body_value_index, body_value.get_element_type(), PartialShape::dynamic());
            if (body_value_partial_shape.is_static())
            {
                auto body_value_shape = body_value_partial_shape.to_shape();
                Shape out_shape{body_value_shape};

                if (body_value_shape.empty())
                {
                    out_shape = Shape(1);
                }

                set_output_type(output_descr->m_output_index, body_value.get_element_type(), out_shape);
            }
            else
            {
                set_output_type(output_descr->m_output_index, body_value.get_element_type(),
                                PartialShape::dynamic(body_value.get_partial_shape().rank()));
            }
        }
    }
    else //condition is non constant
    {
        validate_and_infer_type_body(get_then_body(), m_input_descriptions[then_body_index]);
        validate_and_infer_type_body(get_else_body(), m_input_descriptions[else_body_index]);

        std::set<int64_t> then_output_indexes{};
        std::set<int64_t> else_output_indexes{};
        auto output_nodes = outputs();
        for (auto then_output_description : m_output_descriptions[then_body_index]) {
            auto out_index = then_output_description->m_output_index;
            auto cond = [=](Output<Node>& node) { return node.get_index() == out_index; };
            auto it = std::find_if(output_nodes.begin(), output_nodes.end(), cond);
            NGRAPH_CHECK(it != output_nodes.end(), "Incorrect output with index %i i n \'then_body\'", out_index);
            then_output_indexes.insert(then_output_description->m_output_index);
        }

        NGRAPH_CHECK(
            then_output_indexes.size() == output_nodes.size(),
            "Incorect then_body! Number of then_body outputs must be same as number If outputs");

        for (auto else_output_description : m_output_descriptions[else_body_index])
        {
            auto out_index = else_output_description->m_output_index;
        
            NGRAPH_CHECK(then_output_indexes.find(out_index) != then_output_indexes.end(),
                         "Incorrect output with index %i in \'else_body\'", out_index);
            else_output_indexes.insert(else_output_description->m_output_index);
        }

        NGRAPH_CHECK(
            else_output_indexes.size() == else_output_indexes.size(),
            "Incorect else_body! Number of then_body outputs must be same as number If outputs");

        for (auto output_index : then_output_indexes) 
        {
            auto description_find_lambda = [=](MultiSubgraphOutputDescriptionPtr& descr) {
                return descr->m_output_index == output_index;
            };

            auto then_output_description = *find_if(m_output_descriptions[then_body_index].begin(), 
                m_output_descriptions[then_body_index].end(), description_find_lambda);
            auto else_output_description = *find_if(m_output_descriptions[else_body_index].begin(),
                m_output_descriptions[else_body_index].end(), description_find_lambda);
            auto then_out_node = m_bodies[then_body_index]->get_results()
                .at(then_output_description->m_body_value_index)->input_value(0);
            auto else_out_node = m_bodies[else_body_index]->get_results()
                .at(else_output_description->m_body_value_index)->input_value(0);
            //TODO: check_types
            auto then_node_partial_shape = then_out_node.get_partial_shape();
            auto else_node_partial_shape = else_out_node.get_partial_shape();
            bool is_static_ranks = then_node_partial_shape.rank().is_static() &&
                                   else_node_partial_shape.rank().is_static();
            bool is_static_shapes =
                then_node_partial_shape.is_static() && else_node_partial_shape.is_static();

            if (is_static_ranks && is_static_shapes)
            {
                auto then_shape = then_node_partial_shape.to_shape();
                auto else_shape = else_node_partial_shape.to_shape();
                if (std::equal(then_shape.begin(), then_shape.end(), else_shape.begin())) 
                {
                    set_output_type(output_index, then_out_node.get_element_type(), then_shape);
                }
                else
                {
                    set_output_type(output_index, then_out_node.get_element_type(),
                        PartialShape::dynamic(resolve_dynamic_rank(then_out_node, else_out_node)));
                }
            }
            else
            {
                set_output_type(output_index, then_out_node.get_element_type(),
                    PartialShape::dynamic(resolve_dynamic_rank(then_out_node, else_out_node)));
            }
        }
    }

    //NODE_VALIDATION_CHECK(this,
   //                       get_output_size() == m_output_descriptions.size(),
   //                       "Number of outputs must be the same as number of output descriptions");
}

void op::v0::If::fill_body(std::shared_ptr<op::v0::If> new_op,
                             size_t branch_index,
                             const OutputVector& new_args) const 
{
    auto body = m_bodies[branch_index];
    auto param_size = body->get_parameters().size();
    std::vector<::ngraph::element::Type> types(param_size);
    std::vector<::ngraph::PartialShape> new_shapes(param_size);
    auto& input_descriptions = m_input_descriptions[branch_index];
    size_t parameters_num = 0;
    for (auto& input_description : input_descriptions)
    {
        if (input_description->m_input_index < new_args.size()) 
{
            types[input_description->m_body_parameter_index] =
                new_args[input_description->m_input_index].get_element_type();
            new_shapes[input_description->m_body_parameter_index] =
                new_args[input_description->m_input_index].get_partial_shape();
            parameters_num++;
        }
    }
    auto func =
        std::make_shared<Function>(body->get_results(), body->get_sinks(), body->get_parameters());
    auto spec_func =
        specialize_function(func, types, new_shapes, std::vector<void*>(parameters_num, nullptr));
    new_op->m_bodies[branch_index] = std::make_shared<Function>(
        spec_func->get_results(), spec_func->get_sinks(), spec_func->get_parameters());

    for (auto& input_description : input_descriptions)
    {
        new_op->m_input_descriptions[branch_index].push_back(input_description->copy());
    }
    for (auto& output_description : m_output_descriptions[branch_index])
    {
        new_op->m_output_descriptions[branch_index].push_back(output_description->copy());
    }
}

std::shared_ptr<Node>
    op::v0::If::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v0_If_clone_with_new_inputs);
    auto op = make_shared<op::v0::If>(new_args);
    NGRAPH_CHECK(op.get(),
                 op != nullptr,
                 "Cannot clone ",
                 description(),
                 " operation with name ",
                 get_friendly_name());

  //TODO: check size of output
    op->m_bodies = std::vector<std::shared_ptr<ngraph::Function>>(2);
    op->m_input_descriptions = std::vector<MultiSubgraphInputDescriptionVector>(2);
    op->m_output_descriptions = std::vector<MultiSubgraphOutputDescriptionVector>(2);
    op->set_output_size(m_output_descriptions[0].size());
    fill_body(op, then_body_index, new_args);
    fill_body(op, else_body_index, new_args);
    op->validate_and_infer_types();
    return op;
}
void op::v0::If::set_invariant_input(
    const Output<Node>& value,
    const std::shared_ptr<Parameter>& then_parameter,
    const std::shared_ptr<Parameter>& else_parameter)
{
    auto input_index = input_for_value(value).get_index();
    if (then_parameter != nullptr)
    {
        m_input_descriptions[then_body_index].push_back(
            std::make_shared<MultiSubGraphOp::InvariantInputDescription>(
                input_index, m_bodies[then_body_index]->get_parameter_index(then_parameter)));
    }
    if (else_parameter != nullptr) {
        m_input_descriptions[else_body_index].push_back(
            std::make_shared<MultiSubGraphOp::InvariantInputDescription>(
                input_index, m_bodies[else_body_index]->get_parameter_index(else_parameter)));
    }
    validate_and_infer_types();
}

bool op::v0::If::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const
{
    NGRAPH_OP_SCOPE(v0_If_evaluate);
    runtime::reference::if_reference(
        m_bodies, m_output_descriptions, m_input_descriptions, outputs, inputs);
    return true;
}

