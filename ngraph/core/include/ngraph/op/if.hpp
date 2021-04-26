// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "ngraph/function.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/util/multi_subgraph_base.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            /// \brief  write something
            class NGRAPH_API If : public util::MultiSubGraphOp
            {
            public:
                static constexpr NodeTypeInfo type_info{"If", 0};
                static constexpr char then_body_index = 0;
                static constexpr char else_body_index = 1;

                
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                bool visit_attributes(AttributeVisitor& visitor) override;
                If(const Output<Node>& execution_condition);
                If();
                explicit If(const OutputVector& values);

                std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;
                /// \return the body of the iteration
                std::shared_ptr<Function> get_then_body() const { return m_bodies[then_body_index]; }
                std::shared_ptr<Function> get_else_body() const
                {
                    return m_bodies[else_body_index];
                }
                /// \param body set the body of the iteration
                void set_then_body(const std::shared_ptr<Function>& body) {m_bodies[then_body_index] = body; }
                void set_else_body(const std::shared_ptr<Function>& body)
                {
                    m_bodies[else_body_index] = body;
                }    
                ngraph::Output<Node> set_output(ngraph::Output<ngraph::Node> then_output,
                                 ngraph::Output<ngraph::Node> else_output);
                void validate_and_infer_types() override;
                void set_invariant_input(const Output<Node>& value,
                                         const std::shared_ptr<Parameter>& then_parameter,
                                         const std::shared_ptr<Parameter>& else_parameter);
                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) const override;
            private:
                void validate_and_infer_type_body(
                    std::shared_ptr<Function> body,
                    ngraph::op::util::MultiSubgraphInputDescriptionVector& input_descriptors);
                void fill_body(std::shared_ptr<op::v0::If> new_op,
                                           size_t branch_index,
                                           const OutputVector& new_args) const;
            };
        }
        using v0::If;
    }
}