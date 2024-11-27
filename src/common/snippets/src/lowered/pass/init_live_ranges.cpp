// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/init_live_ranges.hpp"
#include "snippets/itt.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

bool InitLiveRanges::run(LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::InitLiveRanges")
    std::map<RegType, size_t> reg_counter;

    // Note: map expiring time to register
    std::map<double, std::set<Reg>> regs_to_expire;

    const auto always_alive = std::make_pair(linear_ir.front()->get_exec_num(), linear_ir.back()->get_exec_num());

    auto set_global_reg = [&](const std::vector<PortDescriptorPtr>& port_descs){
        const auto reg_type = snippets::RegType::gpr;
        const auto& reg = Reg(reg_type, reg_counter[reg_type]++);
        for (const auto& pd : port_descs)
            pd->set_reg(reg);
        regs_to_expire[always_alive.second].insert(reg);
        m_reg_manager.set_live_range(reg, always_alive);
    };
    // Need to artificially extend lifetime of Parameter consumers and Result producers (including all connected LoopEnds)
    for (const auto& expr : linear_ir.get_parameters()) {
        auto affected_pds = expr->get_output_port_descriptors();
        for (const auto& out : expr->get_output_port_connectors())
            for (const auto& consumer : out->get_consumers())
                affected_pds.push_back(consumer.get_descriptor_ptr());
        set_global_reg(affected_pds);
    }

    for (const auto& expr : linear_ir.get_results()) {
        auto affected_pds = expr->get_input_port_descriptors();
        for (const auto& in : expr->get_input_port_connectors()) {
            const auto& source = in->get_source();
            affected_pds.push_back(source.get_descriptor_ptr());
            for (const auto& sibling : in->get_consumers())
                affected_pds.push_back(sibling.get_descriptor_ptr());
        }
        set_global_reg(affected_pds);
    }

    for (const auto& expr : linear_ir) {
        const auto op = expr->get_node();
        if (ov::is_type<op::LoopEnd>(op) ||
            ov::is_type<ov::op::v0::Result>(op)
#ifdef SNIPPETS_DEBUG_CAPS
            || ov::is_type<op::PerfCountBeginBase>(op)
            || ov::is_type<op::PerfCountEndBase>(op)
#endif
            ) {
            m_reg_manager.set_live_regs(expr, {});
            continue;
        }
        OPENVINO_ASSERT(expr->get_output_count() == op->get_output_size(), "Incorrect count of output port descriptors!");
        const double start = expr->get_exec_num();
        // Remove all regs that expired before start
        regs_to_expire.erase(regs_to_expire.begin(), regs_to_expire.upper_bound(start)); // remove all elements lower than start (not equal)
        std::set<Reg> live_regs;
        for (const auto& time_reg : regs_to_expire)
            live_regs.insert(time_reg.second.begin(), time_reg.second.end());

        m_reg_manager.set_live_regs(expr, std::move(live_regs));

        std::cerr << expr->get_node()->get_friendly_name() << " : ";
        for (auto r : regs_to_expire) {
            for (auto x : r.second)
                std::cerr << x << " ";
        }
        std::cerr << "\n";

        for (size_t i = 0; i < expr->get_output_count(); ++i) {
            const auto& out_pd = expr->get_output_port_descriptor(i);
            if (out_pd->get_reg().is_defined())
                continue;
            const auto reg_type = m_reg_manager.get_reg_type(op->output(i));
            const auto& reg = Reg(reg_type, reg_counter[reg_type]++);
            out_pd->set_reg(reg);
            double stop = start;
            // propogate to consumers
            for (const auto& consumer : expr->get_output_port_connector(i)->get_consumers()) {
                consumer.get_descriptor_ptr()->set_reg(reg);
                stop = std::max(stop, consumer.get_expr()->get_exec_num());
            }
            regs_to_expire[stop].insert(reg);
            m_reg_manager.set_live_range(reg, std::make_pair(start, stop));
        }
    }

    return false;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov

