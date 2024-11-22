// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/op/reg_spill.hpp"

#include "snippets/utils/utils.hpp"

namespace ov {
namespace snippets {
namespace op {

RegSpillBase::RegSpillBase(const std::vector<Output<Node>> &args) : Op(args) {}

RegSpillBegin::RegSpillBegin() {
    validate_and_infer_types_except_RegSpillEnd();
}

void RegSpillBegin::validate_and_infer_types_except_RegSpillEnd() {
    NODE_VALIDATION_CHECK(this, get_input_size() == 0, "RegSpillBegin doesn't expect any inputs");
    set_output_type(0, element::f32, ov::PartialShape{ov::Shape{}});
}

void RegSpillBegin::validate_and_infer_types() {
    validate_and_infer_types_except_RegSpillEnd();
    OPENVINO_ASSERT(get_output_size() == 1, "RegSpillBegin must have only one output");
    const auto& last_output_inputs = get_output_target_inputs(0);
    OPENVINO_ASSERT(last_output_inputs.size() == 1, "RegSpillBegin must have exactly one input attached to the last output");
    OPENVINO_ASSERT(ov::is_type<RegSpillEnd>(last_output_inputs.begin()->get_node()),
                    "RegSpillBegin must have RegSpillEnd connected to its last output");
}

std::shared_ptr<Node> RegSpillBegin::clone_with_new_inputs(const OutputVector& inputs) const {
    OPENVINO_ASSERT(inputs.empty(), "RegSpillBegin should not contain inputs");
    return std::make_shared<RegSpillBegin>();
}

std::shared_ptr<RegSpillEnd> RegSpillBegin::get_reg_spill_end() const {
    const auto& last_output_inputs = get_output_target_inputs(0);
    OPENVINO_ASSERT(last_output_inputs.size() == 1, "RegSpillBegin has more than one inputs attached to the last output");
    const auto& loop_end = ov::as_type_ptr<RegSpillEnd>(last_output_inputs.begin()->get_node()->shared_from_this());
    OPENVINO_ASSERT(loop_end != nullptr, "RegSpillBegin must have RegSpillEnd connected to its last output");
    return loop_end;
}

RegSpillEnd::RegSpillEnd(const Output<Node>& reg_spill_begin, std::set<Reg> regs_to_spill)  :
        RegSpillBase({reg_spill_begin}),
        m_regs_to_spill(std::move(regs_to_spill)) {
    constructor_validate_and_infer_types();
}

void RegSpillEnd::validate_and_infer_types() {
    NODE_VALIDATION_CHECK(this, get_input_size() == 1 && ov::is_type<RegSpillBegin>(get_input_node_shared_ptr(0)),
                         "RegSpillEnd must have one input of RegSPillBegin type");
    set_output_type(0, element::f32, ov::PartialShape{});
}

bool RegSpillEnd::visit_attributes(AttributeVisitor &visitor) {
    std::stringstream ss;
    for (auto reg_it = m_regs_to_spill.begin(); reg_it != m_regs_to_spill.end(); reg_it++) {
        ss << *reg_it;
        if (std::next(reg_it) != m_regs_to_spill.end())
            ss << ", ";
    }
    std::string spilled = ss.str();
    visitor.on_attribute("regs_to_spill", spilled);
    return true;
}

std::shared_ptr<Node> RegSpillEnd::clone_with_new_inputs(const OutputVector& inputs) const {
    check_new_args_count(this, inputs);
    return std::make_shared<RegSpillEnd>(inputs.at(0), m_regs_to_spill);
}

std::shared_ptr<RegSpillBegin> RegSpillEnd::get_reg_spill_begin() {
    const auto& reg_spill_begin = ov::as_type_ptr<RegSpillBegin>(get_input_source_output(0).get_node_shared_ptr());
    OPENVINO_ASSERT(reg_spill_begin, "RegSpillEnd last input is not connected to RegSpillBegin");
    return reg_spill_begin;
}


} // namespace op
} // namespace snippets
} // namespace ov
