// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/ctc_greedy_decoder_seq_len.hpp"

#include <ctc_greedy_decoder_seq_len_shape_inference.hpp>

#include "itt.hpp"

using namespace std;
using namespace ngraph;

BWDCMP_RTTI_DEFINITION(op::v6::CTCGreedyDecoderSeqLen);

op::v6::CTCGreedyDecoderSeqLen::CTCGreedyDecoderSeqLen(const Output<Node>& input,
                                                       const Output<Node>& seq_len,
                                                       const bool merge_repeated,
                                                       const element::Type& classes_index_type,
                                                       const element::Type& sequence_length_type)
    : Op({input, seq_len}),
      m_merge_repeated(merge_repeated),
      m_classes_index_type(classes_index_type),
      m_sequence_length_type(sequence_length_type) {
    constructor_validate_and_infer_types();
}

op::v6::CTCGreedyDecoderSeqLen::CTCGreedyDecoderSeqLen(const Output<Node>& input,
                                                       const Output<Node>& seq_len,
                                                       const Output<Node>& blank_index,
                                                       const bool merge_repeated,
                                                       const element::Type& classes_index_type,
                                                       const element::Type& sequence_length_type)
    : Op({input, seq_len, blank_index}),
      m_merge_repeated(merge_repeated),
      m_classes_index_type(classes_index_type),
      m_sequence_length_type(sequence_length_type) {
    constructor_validate_and_infer_types();
}

void op::v6::CTCGreedyDecoderSeqLen::validate_and_infer_types() {
    NGRAPH_OP_SCOPE(v6_CTCGreedyDecoderSeqLen_validate_and_infer_types);
    const auto& logits_pshape = get_input_partial_shape(0);
    const auto& seq_len_pshape = get_input_partial_shape(1);

    // check optional input type: blank index
    if (get_input_size() == 3) {
        const auto& blank_index_type = get_input_element_type(2);
        NODE_VALIDATION_CHECK(this,
                              blank_index_type.is_integral_number(),
                              "The blank index type is expected to be an integer type. Got: ",
                              blank_index_type);

        const auto& blank_index_partial_shape = get_input_partial_shape(2);
        if (blank_index_partial_shape.is_static()) {
            ov::Shape blank_index_shape = blank_index_partial_shape.to_shape();
            NODE_VALIDATION_CHECK(
                this,
                ngraph::is_scalar(blank_index_shape) || (is_vector(blank_index_shape) && (blank_index_shape[0] == 1)),
                "Expected 0D or 1D tensor for the 'blank_index' input. Got: ",
                blank_index_shape);
        }
    }

    std::vector<ov::PartialShape> output_shapes = {ov::PartialShape{}, ov::PartialShape{}};
    std::vector<ov::PartialShape> input_shapes = {logits_pshape,
                                                  seq_len_pshape};
    shape_infer(this, input_shapes, output_shapes);
    NODE_VALIDATION_CHECK(this, output_shapes.size() == 2);
    set_output_type(0, m_classes_index_type, output_shapes[0]);
    set_output_type(1, m_sequence_length_type, output_shapes[1]);
}

bool op::v6::CTCGreedyDecoderSeqLen::visit_attributes(AttributeVisitor& visitor) {
    NGRAPH_OP_SCOPE(v6_CTCGreedyDecoderSeqLen_visit_attributes);
    visitor.on_attribute("merge_repeated", m_merge_repeated);
    visitor.on_attribute("classes_index_type", m_classes_index_type);
    visitor.on_attribute("sequence_length_type", m_sequence_length_type);
    return true;
}

shared_ptr<Node> op::v6::CTCGreedyDecoderSeqLen::clone_with_new_inputs(const OutputVector& new_args) const {
    NGRAPH_OP_SCOPE(v6_CTCGreedyDecoderSeqLen_clone_with_new_inputs);
    check_new_args_count(this, new_args);

    size_t args_size = new_args.size();
    if (args_size == 2) {
        return make_shared<CTCGreedyDecoderSeqLen>(new_args.at(0),
                                                   new_args.at(1),
                                                   m_merge_repeated,
                                                   m_classes_index_type,
                                                   m_sequence_length_type);
    } else if (args_size == 3) {
        return make_shared<CTCGreedyDecoderSeqLen>(new_args.at(0),
                                                   new_args.at(1),
                                                   new_args.at(2),
                                                   m_merge_repeated,
                                                   m_classes_index_type,
                                                   m_sequence_length_type);
    } else {
        throw ngraph_error("Incorrect number of arguments");
    }
}
