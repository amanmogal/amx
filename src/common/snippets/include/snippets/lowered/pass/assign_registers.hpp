// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "pass.hpp"
#include "snippets/generator.hpp"
#include "snippets/lowered/reg_manager.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

/**
 * @interface AssignRegisters
 * @brief Assigns in/out abstract registers indexes to every operation.
 * Note that changing of the IR is likely to invalidate register assignment.
 * @ingroup snippets
 */
class AssignRegisters : public Pass {
public:
    OPENVINO_RTTI("AssignRegisters", "Pass")
    explicit AssignRegisters(RegManager& reg_manager, const size_t reg_cnt)
                            : m_reg_manager(reg_manager), reg_count(reg_cnt) {}
    bool run(LinearIR& linear_ir) override;

private:
    using RegMap = std::map<Reg, Reg>;
    static RegMap assign_regs_manually(const LinearIR& linear_ir);

    RegManager& m_reg_manager;
    size_t reg_count;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
