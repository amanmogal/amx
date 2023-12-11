// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/core/node.hpp"

namespace ov {
namespace util {

/// \brief Returns a vector with unsupported element types. Constant inputs with those types (in general) require
///        conversion before node can be constant folded
OPENVINO_API
element::TypeVector unsupported_types();

/// \brief is_convert returns true if the first parameter is a Convert or ConvertLike node and false otherwise
/// TODO: find better place for this helper function
OPENVINO_API
bool is_convert(const std::shared_ptr<Node>& node);

/// \brief If the node has constant inputs with types that (before constant_fold) should be converted to f32 - the
///        function converts and constantfolds those inputs to f32. Then, the function clones the node with
///        the new inputs and returns the new node.
///
/// \param node
/// \param inputs               - vector of new inputs to the node
/// \param constant_fold_inputs - If true - tries to constant fold converted inputs
///
/// \return New node with f32 inputs if the inputs require conversion or the input node otherwise
OPENVINO_API std::shared_ptr<Node> try_convert_inputs(const std::shared_ptr<Node>& node,
                                                      OutputVector inputs,
                                                      bool constant_fold_inputs = true);

OPENVINO_API std::shared_ptr<Node> try_convert_inputs(const std::shared_ptr<Node>& node,
                                                      bool constant_fold_inputs = true);

/// \brief Function constantfolds a node.
///        It converts the node inputs if necessary, runs Node::constant_fold
///        and then converts its outputs to original type (if necessary).
///
/// \param node              - node to be constant_folded
/// \param output_constants  - output parameter. A vector with constant_folded nodes. Can be empty - see example below.
///
/// Usage example:
///     ```
///         auto abs = std::make_shared<ov::op::v0::Abs>(ov::op::v0::Constant::create(element::f32, Shape{}, {-2}));
///         OutputVector output_constants;
///         bool status = ov::util::constant_fold_node(abs, output_constants);
///         assert(status);
///         assert(output_constants.size() == 1);
///     ```
///
/// \return true if node was successfully constant_folded, false otherwise
OPENVINO_API bool constant_fold_node(const std::shared_ptr<Node>& node, OutputVector& output_constants);

/// \brief Function evaluates a node.
///        It converts the input_tensors if necessary, runs Node::evaluate
///        and then converts its outputs to original type (if necessary).
///
/// \param node               - node to be evaluated.
/// \param input_tensors      - a vector with input tensors.
/// \param output_tensors     - output parameter. A vector with evaluated tensors. Can be empty.
/// \param evaluation_context - map with additional settings
///
/// \return true if node was successfully evaluated, false otherwise
OPENVINO_API bool evaluate_node(const std::shared_ptr<Node>& node,
                                const TensorVector& input_tensors,
                                TensorVector& output_tensors,
                                const EvaluationContext& evaluation_context = {});

}  // namespace util
}  // namespace ov
