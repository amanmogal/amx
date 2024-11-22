// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_reg_spill_emitters.hpp"

#include "emitters/snippets/jit_snippets_call_args.hpp"
#include "emitters/snippets/x64/utils.hpp"
#include "snippets/utils/utils.hpp"
#include "emitters/plugin/x64/utils.hpp"

#include "snippets/op/reg_spill.hpp"

using namespace Xbyak;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;

namespace ov {
namespace intel_cpu {

/* ================== jit_reg_spill_begin_emitters ====================== */

jit_reg_spill_begin_emitter::jit_reg_spill_begin_emitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                                                         const ov::snippets::lowered::ExpressionPtr& expr)
                                                           : jit_emitter(h, isa), m_reg_spill_begin_expr(expr) {
    OV_CPU_JIT_EMITTER_ASSERT(ov::is_type<snippets::op::RegSpillBegin>(m_reg_spill_begin_expr->get_node()),
                             "expects RegSpillBegin expression");
    // todo: abstract registers were not mapped on physical onces at this point (Kernel emitter does that),
    //  so we have to declare reg_spiller mutable
    m_abi_reg_spiller = std::make_shared<EmitABIRegSpills>(h);
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
}

void jit_reg_spill_begin_emitter::validate_arguments(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    OV_CPU_JIT_EMITTER_ASSERT(out.empty(), "Out regs should be empty for reg_spill_begin emitter");
}

void jit_reg_spill_begin_emitter::emit_code(const std::vector<size_t> &in, const std::vector<size_t> &out,
                                            const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs) const {
    validate_arguments(in, out);
    emit_impl(in, out);
}

void jit_reg_spill_begin_emitter::emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    const auto& reg_info = m_reg_spill_begin_expr->get_reg_info();
    OPENVINO_ASSERT(reg_info.second.empty(), "Out regs should be empty for reg_spill_begin emitter");
    m_abi_reg_spiller->limit_to_live_regs({reg_info.first.begin(), reg_info.first.end()});
    m_abi_reg_spiller->preamble();
}

/* ============================================================== */

/* ================== jit_reg_spill_end_emitter ====================== */

jit_reg_spill_end_emitter::jit_reg_spill_end_emitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                                                     const ov::snippets::lowered::ExpressionPtr& expr)
    : jit_emitter(h, isa) {
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
    OV_CPU_JIT_EMITTER_ASSERT(ov::is_type<snippets::op::RegSpillEnd>(expr->get_node()) && expr->get_input_count() == 1,
                              "Invalid expression in RegSpillEnd emitter");
    const auto& parent_expr = expr->get_input_port_connector(0)->get_source().get_expr();
    const auto& reg_spill_begin_emitter = std::dynamic_pointer_cast<jit_reg_spill_begin_emitter>(parent_expr->get_emitter());
    OV_CPU_JIT_EMITTER_ASSERT(reg_spill_begin_emitter, "Failed to obtain reg_spill_begin emitter");
    m_abi_reg_spiller = reg_spill_begin_emitter->m_abi_reg_spiller;
}


void jit_reg_spill_end_emitter::validate_arguments(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    OV_CPU_JIT_EMITTER_ASSERT(out.empty(), "Out regs should be empty for reg_spill_end emitter");
}

void jit_reg_spill_end_emitter::emit_code(const std::vector<size_t> &in, const std::vector<size_t> &out,
                                     const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs) const {
    validate_arguments(in, out);
    emit_impl(in, out);
}

void jit_reg_spill_end_emitter::emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    m_abi_reg_spiller->postamble();
}

}   // namespace intel_cpu
}   // namespace ov
