// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/shape_of.hpp"

#include <algorithm>
#include <vector>

#include "itt.hpp"
#include "openvino/core/dimension_tracker.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/select.hpp"
#include "openvino/reference/shape_of.hpp"

namespace ov {
namespace op {
namespace shape_of {
namespace {
template <element::Type_t ET>
inline bool evaluate(const Shape& shape, Tensor& output_value) {
    reference::shape_of(shape, output_value.data<fundamental_type_for<ET>>());
    return true;
}

bool evaluate_shape_of(Tensor& output_value, const Shape& input_shape) {
    bool rc;
    output_value.set_shape(Shape{input_shape.size()});
    switch (output_value.get_element_type()) {
        OPENVINO_TYPE_CASE(evaluate_shape_of, i32, input_shape, output_value);
        OPENVINO_TYPE_CASE(evaluate_shape_of, i64, input_shape, output_value);
        OPENVINO_TYPE_CASE(evaluate_shape_of, u32, input_shape, output_value);
        OPENVINO_TYPE_CASE(evaluate_shape_of, u64, input_shape, output_value);
    default:
        rc = false;
        break;
    }
    return rc;
}

bool constant_fold_shape_of(Node* shape_of_node, Output<Node>& replacement, const Output<Node>& shape_of_input) {
    const auto& partial_shape = shape_of_input.get_partial_shape();
    if (partial_shape.is_static()) {
        const auto& output_type = shape_of_node->get_output_element_type(0);
        const auto& output_shape = shape_of_node->get_output_shape(0);
        auto result_tensor = ov::Tensor{output_type, output_shape};
        if (evaluate_shape_of(result_tensor, shape_of_input.get_shape())) {
            replacement = std::make_shared<v0::Constant>(result_tensor);
            return true;
        }
    }
    return false;
}

bool evaluate_bound_shape(const Node* shape_of_node, TensorVector& output_values, bool is_upper) {
    OPENVINO_ASSERT(shape_of_node, output_values.size() == 1);
    const auto& input_partial_shape = shape_of_node->get_input_partial_shape(0);
    if (input_partial_shape.rank().is_dynamic())
        return false;
    const auto rank = input_partial_shape.rank().get_length();
    auto pshape_low = PartialShape::dynamic(rank), pshape_up = PartialShape::dynamic(rank);
    for (Dimension::value_type i = 0; i < rank; ++i) {
        Interval interval = input_partial_shape[i].get_interval();
        pshape_low[i] = interval.get_min_val();
        pshape_up[i] = Dimension(interval.get_max_val()).is_dynamic() ? Dimension(interval.get_max_val() - 1)
                                                                      : interval.get_max_val();
    }
    OPENVINO_ASSERT(pshape_up.is_static() && pshape_low.is_static());
    const auto output_et = output_values[0].get_element_type();

    if (pshape_low.to_shape() == pshape_up.to_shape()) {
        shape_of::evaluate_shape_of(output_values[0], pshape_low.to_shape());
    } else {
        auto&& upper = is_upper ? output_values : TensorVector{{output_et, Shape{pshape_up.to_shape().size()}}};
        shape_of::evaluate_shape_of(upper[0], pshape_up.to_shape());

        auto&& lower = is_upper ? TensorVector{{output_et, Shape{pshape_low.to_shape().size()}}} : output_values;
        shape_of::evaluate_shape_of(lower[0], pshape_low.to_shape());

        std::vector<char> dynamic_mask;  // true if dimension is dynamic
        for (const auto& i : input_partial_shape)
            dynamic_mask.push_back(static_cast<char>(Dimension(i.get_interval().get_max_val()).is_dynamic()));

        const auto mask_const = Tensor(element::boolean, Shape{dynamic_mask.size()}, dynamic_mask.data());

        auto&& min = output_et == element::i64 ? static_cast<int64_t>(0) : static_cast<int32_t>(0);
        auto&& max =
            output_et == element::i64 ? std::numeric_limits<int64_t>::max() : std::numeric_limits<int32_t>::max();

        v1::Select().evaluate(lower, {mask_const, {output_et, Shape{}, &min}, lower.front()});
        v1::Select().evaluate(upper, {mask_const, {output_et, Shape{}, &max}, upper.front()});
    }
    return true;
}

bool evaluate_label(const Node* shape_of_node, TensorLabelVector& output_labels) {
    const auto& shape = shape_of_node->get_input_partial_shape(0);
    OPENVINO_ASSERT(shape.rank().is_static());  // sanity check. at this point value propagation was successful
    output_labels[0].reserve(shape.size());
    bool label_is_set = false;
    for (const auto& d : shape) {
        const auto& label = DimensionTracker::get_label(d);
        if (label)
            label_is_set = true;
        output_labels[0].push_back(label);
    }
    return label_is_set;
}
}  // namespace
}  // namespace shape_of

namespace v3 {
ShapeOf::ShapeOf(const Output<Node>& arg, element::Type output_type) : ShapeOfBase({arg}), m_output_type(output_type) {
    constructor_validate_and_infer_types();
}

void ShapeOf::validate_and_infer_types() {
    OV_OP_SCOPE(v3_ShapeOf_validate_and_infer_types);
    NODE_VALIDATION_CHECK(this,
                          m_output_type == element::i64 || m_output_type == element::i32,
                          "Output type must be i32 or i64");
    set_input_is_relevant_to_value(0, false);
    const auto input_partial_shape = get_input_partial_shape(0);
    set_output_type(0, m_output_type, PartialShape{input_partial_shape.rank()});
}

bool ShapeOf::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v3_ShapeOf_visit_attributes);
    visitor.on_attribute("output_type", m_output_type);
    return true;
}

std::shared_ptr<Node> ShapeOf::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v3_ShapeOf_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<ShapeOf>(new_args.at(0), m_output_type);
}

bool ShapeOf::evaluate(TensorVector& output_values, const TensorVector& input_values) const {
    OV_OP_SCOPE(v0_ShapeOf_evaluate);
    OPENVINO_ASSERT(input_values.size() == 1);
    OPENVINO_ASSERT(output_values.size() == 1);

    return shape_of::evaluate_shape_of(output_values[0], input_values[0].get_shape());
}

bool ShapeOf::has_evaluate() const {
    OV_OP_SCOPE(v3_ShapeOf_has_evaluate);
    switch (get_output_element_type(0)) {
    case element::i32:
    case element::i64:
    case element::u32:
    case element::u64:
        return true;
    default:
        return false;
    }
}

bool ShapeOf::evaluate_lower(TensorVector& output_values) const {
    return shape_of::evaluate_bound_shape(this, output_values, false);
}

bool ShapeOf::evaluate_upper(TensorVector& output_values) const {
    return shape_of::evaluate_bound_shape(this, output_values, true);
}

bool ShapeOf::evaluate_label(TensorLabelVector& output_labels) const {
    return shape_of::evaluate_label(this, output_labels);
}

bool ShapeOf::constant_fold(OutputVector& output_values, const OutputVector& input_values) {
    OV_OP_SCOPE(v3_ShapeOf_constant_fold);
    if (is_const_fold_disabled()) {
        return false;
    }
    return shape_of::constant_fold_shape_of(this, output_values[0], input_values[0]);
}
}  // namespace v3

namespace v0 {
ShapeOf::ShapeOf(const Output<Node>& arg) : ShapeOfBase({arg}) {
    constructor_validate_and_infer_types();
}

void ShapeOf::validate_and_infer_types() {
    OV_OP_SCOPE(v0_ShapeOf_validate_and_infer_types);
    set_input_is_relevant_to_value(0, false);
    set_output_type(0, element::i64, PartialShape{get_input_partial_shape(0).rank()});
}

std::shared_ptr<Node> ShapeOf::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_ShapeOf_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    auto new_shape_of = std::make_shared<v0::ShapeOf>(new_args.at(0));
    OPENVINO_ASSERT(new_shape_of.get(),
                    new_shape_of != nullptr,
                    "Cannot clone ",
                    description(),
                    " operation with name ",
                    get_friendly_name());
    return new_shape_of;
}

bool ShapeOf::evaluate(TensorVector& output_values, const TensorVector& input_values) const {
    OV_OP_SCOPE(v0_ShapeOf_evaluate);
    OPENVINO_ASSERT(input_values.size() == 1);
    OPENVINO_ASSERT(output_values.size() == 1);

    return shape_of::evaluate_shape_of(output_values[0], input_values[0].get_shape());
}

bool ShapeOf::has_evaluate() const {
    OV_OP_SCOPE(v0_ShapeOf_has_evaluate);
    switch (get_output_element_type(0)) {
    case element::i32:
    case element::i64:
    case element::u32:
    case element::u64:
        return true;
    default:
        return false;
    }
}

bool ShapeOf::constant_fold(OutputVector& output_values, const OutputVector& input_values) {
    OV_OP_SCOPE(v0_ShapeOf_constant_fold);
    if (is_const_fold_disabled()) {
        return false;
    }
    return shape_of::constant_fold_shape_of(this, output_values[0], input_values[0]);
}

bool ShapeOf::evaluate_lower(TensorVector& output_values) const {
    return shape_of::evaluate_bound_shape(this, output_values, false);
}

bool ShapeOf::evaluate_upper(TensorVector& output_values) const {
    return shape_of::evaluate_bound_shape(this, output_values, true);
}

bool ShapeOf::evaluate_label(TensorLabelVector& output_labels) const {
    return shape_of::evaluate_label(this, output_labels);
}
}  // namespace v0
}  // namespace op
}  // namespace ov
