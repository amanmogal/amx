// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/emitter.hpp"

#include "openvino/op/op.hpp"

namespace ov {
namespace snippets {
namespace op {

/**
 * @interface RegSpillBase
 * @brief Base class for RegSpillBegin and RegSpillEnd ops
 * @ingroup snippets
 */
class RegSpillBase : public ov::op::Op {
public:
    OPENVINO_OP("RegSpillBaseBase", "SnippetsOpset");
    RegSpillBase(const std::vector<Output<Node>>& args);
    RegSpillBase() = default;
    virtual std::set<Reg> get_regs_to_spill() const = 0;
protected:
};
class RegSpillBegin;
/**
 * @interface RegSpillEnd
 * @brief Marks the end of the register spill region.
 * @ingroup snippets
 */
class RegSpillEnd : public RegSpillBase {
public:
    OPENVINO_OP("RegSpillEnd", "SnippetsOpset", RegSpillBase);
    RegSpillEnd() = default;
    RegSpillEnd(const Output<Node>& reg_spill_begin, std::set<Reg> regs_to_spill);

    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override;
    std::shared_ptr<RegSpillBegin> get_reg_spill_begin();
    std::set<Reg> get_regs_to_spill() const override { return m_regs_to_spill; }

protected:
    std::set<Reg> m_regs_to_spill = {};
};

/**
 * @interface RegSpillBegin
 * @brief Marks the start of the register spill region.
 * @ingroup snippets
 */
class RegSpillBegin : public RegSpillBase {
public:
    OPENVINO_OP("RegSpillBegin", "SnippetsOpset", RegSpillBase);
    RegSpillBegin();

    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override;
    std::shared_ptr<RegSpillEnd> get_reg_spill_end() const;
    std::set<Reg> get_regs_to_spill() const override {
        return get_reg_spill_end()->get_regs_to_spill();
    }

protected:
    void validate_and_infer_types_except_RegSpillEnd();
};

} // namespace op
} // namespace snippets
} // namespace ov
