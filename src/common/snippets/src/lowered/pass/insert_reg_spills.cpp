// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/insert_reg_spills.hpp"

#include "snippets/lowered/linear_ir.hpp"
#include "snippets/op/reg_spill.hpp"
#include "snippets/op/brgemm.hpp"
#include "snippets/itt.hpp"
#include "snippets/utils/utils.hpp"

//todo: remove debug headers
#include "snippets/lowered/pass/serialize_control_flow.hpp"


namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

bool InsertRegSpills::run(LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::InsertRegSpills")

    auto needs_reg_spill = [](const ExpressionPtr& expr) {
        return ov::is_type<snippets::op::Brgemm>(expr->get_node());
    };
//    const auto& loop_mngr = linear_ir.get_loop_manager();
    bool modified = false;
    for (auto it = linear_ir.begin(); it != linear_ir.end(); it++) {
        const auto& expr = *it;
        if (!needs_reg_spill(expr))
            continue;
        auto regs_to_spill = m_reg_manager.get_live_regs(expr);
        auto insert_pos = it;
        // todo: add loop hoisting optimization
//        int num_loops = 0;
//        while (ov::is_type<snippets::op::LoopBegin>(std::prev(insert_pos)->get()->get_node())) {
//            insert_pos--;
//            num_loops++;
//        }
        const auto begin = std::make_shared<op::RegSpillBegin>(regs_to_spill);
        const auto end = std::make_shared<op::RegSpillEnd>(begin);
        const auto spill_begin_expr = *linear_ir.insert_node(begin, std::vector<PortConnectorPtr>{}, expr->get_loop_ids(),
                                                             false, insert_pos, std::vector<std::set<ExpressionPort>>{});
        std::vector<Reg> vregs{regs_to_spill.begin(), regs_to_spill.end()};
        spill_begin_expr->set_reg_info({{}, vregs});

        const auto spill_end_expr = *linear_ir.insert_node(end, spill_begin_expr->get_output_port_connectors(), expr->get_loop_ids(),
                                                           false, std::next(insert_pos), std::vector<std::set<ExpressionPort>>{});
        spill_end_expr->set_reg_info({vregs, {}});
        modified = true;
    }
    return modified;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov

