// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_ops/lstm_sequence_ie.hpp"

#include <memory>
#include <string>
#include <vector>

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(op::LSTMSequenceIE, "LSTMSequenceIE", 1);

op::LSTMSequenceIE::LSTMSequenceIE(const Output<Node> &X,
                                   const Output<Node> &H_t,
                                   const Output<Node> &C_t,
                                   const Output<Node> &WR,
                                   const Output<Node> &B,
                                   std::size_t hidden_size,
                                   ngraph::opset4::LSTMSequence::direction direction,
                                   const std::vector<std::string> &activations,
                                   const std::vector<float> &activations_alpha,
                                   const std::vector<float> &activations_beta,
                                   float clip)
        : Op({X, H_t, C_t, WR, B}),
          RNNCellBase(hidden_size, clip, activations, activations_alpha, activations_beta),
          m_direction(direction) {
    constructor_validate_and_infer_types();
}

void op::LSTMSequenceIE::validate_and_infer_types() {
    element::Type arg_type = get_input_element_type(0);
    PartialShape output_shape_0{PartialShape::dynamic(3)};
    PartialShape output_shape_1{PartialShape::dynamic(2)};
    if (get_input_partial_shape(0).is_static()) {
        size_t batch_size = get_input_partial_shape(0).get_shape()[0];
        size_t seq_length = get_input_partial_shape(0).get_shape()[1];
        output_shape_0 = Shape{batch_size, seq_length, m_hidden_size};
        output_shape_1 = Shape{batch_size, m_hidden_size};
    }
    set_output_type(0, arg_type, output_shape_0);
    set_output_type(1, arg_type, output_shape_1);
    set_output_type(2, arg_type, output_shape_1);
}

bool ngraph::op::LSTMSequenceIE::visit_attributes(AttributeVisitor& visitor) {
    visitor.on_attribute("direction", m_direction);
    return op::util::RNNCellBase::visit_attributes(visitor);
}

shared_ptr<Node> op::LSTMSequenceIE::clone_with_new_inputs(const OutputVector &new_args) const {
    check_new_args_count(this, new_args);
    return make_shared<op::LSTMSequenceIE>(new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3),
            new_args.at(4), m_hidden_size, m_direction, m_activations, m_activations_alpha, m_activations_beta,
            m_clip);
}
