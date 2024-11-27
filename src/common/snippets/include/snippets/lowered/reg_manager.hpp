// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "openvino/core/node.hpp"
#include "snippets/emitter.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/generator.hpp"

/**
 * @interface RegManager
 * @brief The class holds supplementary info about assigned registers and live ranges
 * @ingroup snippets
 */
namespace ov {
namespace snippets {
namespace lowered {

using RegTypeMapper = std::function<RegType(const ov::Output<Node>& out)>;
using LiveInterval = std::pair<double, double>;
class RegManager {
public:
    RegManager() = delete;
    RegManager(const std::shared_ptr<Generator>& generator) : m_generator(generator) {}
    inline RegType get_reg_type(const ov::Output<Node>& out) const { return m_generator->get_op_out_reg_type(out); }
    inline size_t get_gp_reg_count() const { return m_generator->get_target_machine()->get_gp_reg_count(); }
    inline size_t get_vec_reg_count() const { return m_generator->get_target_machine()->get_vec_reg_count(); }
//    inline bool need_abi_reg_spill() const {m_generator->}
    inline void set_live_regs(const ExpressionPtr& expr, std::set<Reg>&& live, bool force = false) {
        OPENVINO_ASSERT(force || m_live_reg.count(expr) == 0, "Live regs for this expression already registered");
        m_live_reg.insert({expr, live});
    }
    inline const std::set<Reg>& get_live_regs(const ExpressionPtr& expr) const {
        OPENVINO_ASSERT(m_live_reg.count(expr), "Live regs for this expression were not registered");
        return m_live_reg.at(expr);
    }

    inline void set_live_range(const Reg& reg, const LiveInterval& interval, bool force = false) {
        OPENVINO_ASSERT(force || m_reg_live_range.count(reg) == 0, "Live range for this reg is already set");
        m_reg_live_range[reg] = interval;
    }

    inline const LiveInterval& get_live_range(const Reg& reg) {
        OPENVINO_ASSERT(m_reg_live_range.count(reg), "Live range for this reg was not set");
        return m_reg_live_range[reg];
    }
    inline std::map<Reg, LiveInterval> get_live_range_map() const {
        return m_reg_live_range;
    }

private:
    // Maps Register to {Start, Stop} pairs
    std::map<Reg, LiveInterval> m_reg_live_range;
    // Regs that are live on input of the key expression
    std::unordered_map<ExpressionPtr , std::set<Reg>> m_live_reg;
    const std::shared_ptr<const Generator> m_generator;
};

} // namespace lowered
} // namespace snippets
} // namespace ov
