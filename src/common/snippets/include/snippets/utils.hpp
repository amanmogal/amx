// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A file contains public utilities.
 * @file utils.hpp
 */
#pragma once

#include "snippets_isa.hpp"
#include "emitter.hpp"

#include "snippets/op/subgraph.hpp"

namespace ngraph {
namespace snippets {
namespace utils {

// Get non-scalar Constant count that will be created after FakeQuantize decomposition.
// This count is needed to know exact count of non-scalar Constants during tokenization.
auto get_non_scalar_constant_count_for_fq(const std::shared_ptr<ngraph::opset1::FakeQuantize>& fq) -> size_t;

inline auto is_scalar_constant(const std::shared_ptr<ngraph::Node>& source_output_node) -> bool {
    return ngraph::is_type<ngraph::opset1::Constant>(source_output_node) && ngraph::shape_size(source_output_node->get_shape()) == 1;
}


ov::PartialShape get_port_planar_shape(const Output<Node>& out);
ov::PartialShape get_reordered_planar_shape(const ov::PartialShape& shape, const std::vector<size_t>& layout);
std::vector<size_t> get_node_output_layout(const std::shared_ptr<Node>& node);
std::vector<size_t> get_node_output_layout(const Node* node);

// Need to update tensor name manually, since intel_cpu::Graph::Replicate() looks at input.get_tensor().get_name();
// If subgraph->get_output_size() == 1, then the name will be restored correctly from the node name
void update_out_tensor_name(const std::shared_ptr<ngraph::snippets::op::Subgraph>& subgraph);

inline ov::Dimension get_inner_dim(const ov::PartialShape &shape) { return *(shape.rbegin()); }
inline ov::Dimension get_outer_dim(const ov::PartialShape &shape) { return *(shape.rbegin() + 1); }

// Non-scalar Constants are tokenized as Parameters inside Subgraph body but some operations with constant inputs
// should have explicit Constants even if they're non-scalar (Reshape, Transpose, Broadcast)
// This check returns True if Constant op which is input of this op should be inside Subgraph body
inline auto constant_input_should_be_inside_body(const std::shared_ptr<ov::Node>& node) -> bool {
    return ov::is_type<ov::op::v1::Transpose>(node) ||
           ov::is_type<ov::op::v1::Broadcast>(node) ||
           ov::is_type<ov::op::v1::Reshape>(node);
}

} // namespace utils
} // namespace snippets
} // namespace ngraph