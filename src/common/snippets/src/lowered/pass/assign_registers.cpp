// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/assign_registers.hpp"

#include "snippets/lowered/linear_ir.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/itt.hpp"
#include "snippets/utils/utils.hpp"

//todo: remove debug headers
#include "snippets/lowered/pass/serialize_control_flow.hpp"


// This header is needed to avoid MSVC warning "C2039: 'inserter': is not a member of 'std'"
#include <iterator>

// todo:
//  1. Move set_reg_types to a separate pass => Ok
//  2. Modify set_reg_types, so it stores a set of live regs for every expression => Ok
//  3. Implement abstract to physical mapping as a separate backend-specific pass => Investigation needed

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

AssignRegisters::RegMap AssignRegisters::assign_regs_manually(const LinearIR& linear_ir, std::set<Reg>& gpr_pool, std::set<Reg>& vec_pool) {
    RegMap manually_assigned;
    for (const auto& param : linear_ir.get_parameters()) {
        const auto& assigned = *gpr_pool.begin();
        manually_assigned[param->get_output_port_descriptor(0)->get_reg()] = assigned;
        // TODO [96434]: Support shape infer ops in arbitrary place in pipeline, not just after inputs
        // shape infer ops sequence after input
        const auto& shape_infer_consumers = utils::get_first_child_shape_infer_expr_seq(param);
        for (const auto& child_shape_infer_expr : shape_infer_consumers) {
            manually_assigned[child_shape_infer_expr->get_output_port_descriptor(0)->get_reg()] = assigned;
        }
        gpr_pool.erase(gpr_pool.begin());
    }
    for (const auto& result : linear_ir.get_results()) {
        const auto& assigned = *gpr_pool.begin();
        manually_assigned[result->get_input_port_descriptor(0)->get_reg()] = assigned;
        // shape infer ops sequence before result
        const auto& shape_infer_sources = utils::get_first_parent_shape_infer_expr_seq(result);
        for (const auto& parent_shape_infer_expr : shape_infer_sources) {
            manually_assigned[parent_shape_infer_expr->get_input_port_descriptor(0)->get_reg()] = assigned;
        }
        gpr_pool.erase(gpr_pool.begin());
    }

    long int max_buffer_group = -1;
    for (const auto& expr : linear_ir.get_ops()) {
        auto op = expr->get_node();
        if (const auto& buffer = ov::as_type_ptr<BufferExpression>(expr)) {
            // All buffers have one common data pointer
            const auto reg_group = static_cast<long int>(buffer->get_reg_group());
            max_buffer_group = std::max(max_buffer_group, reg_group);
            const auto& assigned = *std::next(gpr_pool.begin(), reg_group);
            for (const auto& input : expr->get_input_port_descriptors()) {
                manually_assigned[input->get_reg()] = assigned;
                // shape infer ops in the middle of subgraph. Buffer is inserted before reshape as new loop should start.
                // child shape info ops share the same memory as Buffer.
                const auto& shape_infer_consumers = utils::get_first_child_shape_infer_expr_seq(expr);
                for (const auto& child_shape_infer_expr : shape_infer_consumers) {
                    manually_assigned[child_shape_infer_expr->get_input_port_descriptor(0)->get_reg()] =
                    manually_assigned[child_shape_infer_expr->get_output_port_descriptor(0)->get_reg()] = assigned;
                }
            }
            manually_assigned[expr->get_output_port_descriptor(0)->get_reg()] = assigned;
        } else if (ov::is_type<op::HorizonMax>(op) || ov::is_type<op::HorizonSum>(op)) {
            // Only in ReduceDecomposition Reduce ops use HorizonMax/HorizonSum and VectorBuffer.
            // We should manually set the one vector register for VectorBuffer and Max/Sum output to simulate a accumulator
            // TODO [96351]: We should rewrite accumulator pattern using another way
            const auto& input_tensor = expr->get_input_port_connector(0);
            const auto& input = input_tensor->get_source();
            const auto& assigned = *vec_pool.begin();
            for (const auto& tensor : input.get_expr()->get_input_port_connectors()) {
                const auto parent = tensor->get_source();
                const auto parent_expr = parent.get_expr();
                if (ov::is_type<op::Fill>(parent_expr->get_node())) {
                    if (ov::is_type<op::VectorBuffer>(parent_expr->get_input_port_connector(0)->get_source().get_expr()->get_node())) {
                        manually_assigned[parent.get_descriptor_ptr()->get_reg()] =
                        manually_assigned[parent_expr->get_input_port_descriptor(0)->get_reg()] = assigned;
                    }
                }
            }
            manually_assigned[input.get_descriptor_ptr()->get_reg()] = assigned;
            vec_pool.erase(vec_pool.begin());
        }
    }
    gpr_pool.erase(gpr_pool.begin(), std::next(gpr_pool.begin(), max_buffer_group + 1));
    return manually_assigned;
}

bool AssignRegisters::run(LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::AssignRegisters")

    const auto& exprs = linear_ir.get_ops();

    std::set<Reg> gpr_pool;
    for (size_t idx = 0; idx < m_reg_manager.get_gp_reg_count(); idx++)
        gpr_pool.insert(Reg(RegType::gpr, idx));
    std::set<Reg> vec_pool;
    for (size_t idx = 0; idx < m_reg_manager.get_vec_reg_count(); idx++)
        vec_pool.insert(Reg(RegType::vec, idx));

    auto assigned_reg_map = assign_regs_manually(linear_ir, gpr_pool, vec_pool);

    struct by_starting {
        auto operator()(const LiveInterval& lhs, const LiveInterval& rhs) const -> bool {
            return lhs.first < rhs.first|| (lhs.first == rhs.first && lhs.second < rhs.second);
        }
    };

    struct by_ending {
        auto operator()(const LiveInterval& lhs, const LiveInterval& rhs) const -> bool {
            return lhs.second < rhs.second || (lhs.second == rhs.second && lhs.first < rhs.first);
        }
    };

    // A variable live LiveInterval - is a range (start, stop) of op indexes, such that
    // the variable is alive within this range (defined but not used by the last user)
    std::map<LiveInterval, Reg, by_starting> live_intervals_vec, live_intervals_gpr;
    for (const auto& regint : m_reg_manager.get_live_range_map()) {
        const auto& reg = regint.first;
        const auto& interval = regint.second;
        // If a register is assigned manually, we should ignore it during automatic assignment
        if (assigned_reg_map.count(reg))
            continue;
        switch (reg.type) {
            case (RegType::gpr):
                OPENVINO_ASSERT(!live_intervals_gpr.count(interval), "GPR live interval is already in the map");
                live_intervals_gpr[interval] = reg;
                break;
            case (RegType::vec):
                OPENVINO_ASSERT(!live_intervals_gpr.count(interval), "VEC live interval is already in the map");
                live_intervals_vec[interval] = reg;
                break;
            case (RegType::undefined):
            default:
                OPENVINO_THROW("Unhandled register type");
        }
    }

    auto linescan_assign_registers = [](const decltype(live_intervals_vec)& live_intervals,
                                        const std::set<Reg>& reg_pool) {
        // http://web.cs.ucla.edu/~palsberg/course/cs132/linearscan.pdf
        std::map<LiveInterval, Reg, by_ending> active;
        // uniquely defined register => reused reg (reduced subset enabled by reg by reusage)
        std::map<Reg, Reg> register_map;
        std::stack<Reg> bank;
        // regs are stored in ascending order in reg_pool, so walk in reverse to assign them the same way
        for (auto rit = reg_pool.crbegin(); rit != reg_pool.crend(); rit++)
            bank.push(*rit);

        LiveInterval interval, active_interval;
        Reg unique_reg, active_unique_reg;
        for (const auto& interval_reg : live_intervals) {
            std::tie(interval, unique_reg) = interval_reg;
            // check expired
            while (!active.empty()) {
                std::tie(active_interval, active_unique_reg) = *active.begin();
                // if end of active interval has not passed yet => stop removing actives since they are sorted by end
                if (active_interval.second >= interval.first) {
                    break;
                }
                active.erase(active_interval);
                bank.push(register_map[active_unique_reg]);
            }
            // allocate
            OPENVINO_ASSERT(active.size() != reg_pool.size(), "Can't allocate registers for a snippet: not enough registers");
            register_map[unique_reg] = bank.top();
            bank.pop();
            active.insert(interval_reg);
        }
        return register_map;
    };

    const auto& map_vec = linescan_assign_registers(live_intervals_vec, vec_pool);
    assigned_reg_map.insert(map_vec.begin(), map_vec.end());
    const auto& map_gpr = linescan_assign_registers(live_intervals_gpr, gpr_pool);
    assigned_reg_map.insert(map_gpr.begin(), map_gpr.end());

    for (const auto& expr : exprs) {
        std::set<Reg> mapped_live_regs;
        for (const auto& live_reg : m_reg_manager.get_live_regs(expr))
            mapped_live_regs.insert(assigned_reg_map[live_reg]);
        m_reg_manager.set_live_regs(expr, std::move(mapped_live_regs), true);
        for (const auto& in : expr->get_input_port_descriptors())
            in->set_reg(assigned_reg_map[in->get_reg()]);
        for (const auto& out : expr->get_output_port_descriptors())
            out->set_reg(assigned_reg_map[out->get_reg()]);
    }
    return false;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov

