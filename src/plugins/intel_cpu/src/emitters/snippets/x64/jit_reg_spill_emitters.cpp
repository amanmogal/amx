// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_reg_spill_emitters.hpp"

#include "emitters/snippets/jit_snippets_call_args.hpp"
#include "emitters/snippets/x64/utils.hpp"
#include "snippets/utils/utils.hpp"

#include "snippets/op/reg_spill.hpp"

using namespace Xbyak;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;

namespace ov {
namespace intel_cpu {

/* ================== jit_reg_spill_begin_emitters ====================== */

jit_reg_spill_begin_emitters::jit_reg_spill_begin_emitters(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                                                           const ov::snippets::lowered::ExpressionPtr& expr)
                                                           : jit_emitter(h, isa), m_reg_spill_begin_expr(expr) {
    OV_CPU_JIT_EMITTER_ASSERT(ov::is_type_ptr<snippets::op::RegSpillBegin>(m_reg_spill_begin_expr->get_node()),
                             "expects RegSpillBegin expression");
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
}

void jit_reg_spill_begin_emitters::validate_arguments(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
//    OV_CPU_JIT_EMITTER_ASSERT(in.empty(), "Invalid inputs size: expected 0 got " + std::to_string(in.size()));
    OV_CPU_JIT_EMITTER_ASSERT(out.empty(), "Out regs should be empty for reg_spill_begin emitter");
}

void jit_reg_spill_begin_emitters::emit_code(const std::vector<size_t> &in, const std::vector<size_t> &out,
                                       const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs) const {
    validate_arguments(in, out);
    emit_impl(in, out);
}

void jit_reg_spill_begin_emitters::emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    m_reg_spill_begin_expr->get_reg_info();
}

/* ============================================================== */

/* ================== jit_loop_end_emitter ====================== */

jit_loop_end_emitter::jit_loop_end_emitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                                           const ov::snippets::lowered::ExpressionPtr& expr)
    : jit_emitter(h, isa), loop_begin_label{nullptr}, loop_end_label{new Xbyak::Label()} {
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
    const auto loop_end = ov::as_type_ptr<snippets::op::LoopEnd>(expr->get_node());
    OV_CPU_JIT_EMITTER_ASSERT(loop_end != nullptr, "expected LoopEnd expr");
    num_inputs = loop_end->get_input_num();
    num_outputs = loop_end->get_output_num();
    work_amount = loop_end->get_work_amount();
    wa_increment = loop_end->get_increment();
    is_incremented = loop_end->get_is_incremented();
    ptr_increments = loop_end->get_ptr_increments();
    finalization_offsets = loop_end->get_finalization_offsets();
    data_sizes = loop_end->get_element_type_sizes();
    evaluate_once = loop_end->get_evaluate_once();
    loop_id = loop_end->get_id();

    are_ptr_increments_dynamic =
        std::any_of(ptr_increments.cbegin(), ptr_increments.cend(), ov::snippets::utils::is_dynamic_value<int64_t>);
    are_final_offsets_dynamic =
        std::any_of(finalization_offsets.cbegin(), finalization_offsets.cend(), ov::snippets::utils::is_dynamic_value<int64_t>);
    are_ptr_shifts_dynamic = are_ptr_increments_dynamic || are_final_offsets_dynamic;

    const auto begin_expr = get_loop_begin_expr(expr);
    const auto& loop_begin_emitter = std::dynamic_pointer_cast<jit_loop_begin_emitter>(begin_expr->get_emitter());
    OV_CPU_JIT_EMITTER_ASSERT(loop_begin_emitter, "LoopBegin expected jit_loop_begin_emitter");
    loop_begin_emitter->set_loop_end_label(loop_end_label);
    loop_begin_label = loop_begin_emitter->get_begin_label();
}

ov::snippets::lowered::ExpressionPtr jit_loop_end_emitter::get_loop_begin_expr(const ov::snippets::lowered::ExpressionPtr& expr) {
    const auto begin_expr = expr->get_input_port_connectors().back()->get_source().get_expr();
    OV_CPU_JIT_EMITTER_ASSERT(ov::is_type<snippets::op::LoopBegin>(begin_expr->get_node()),
                              "LoopEnd expression must have th last port connector to LoopBegin");
    return begin_expr;
}

void jit_loop_end_emitter::validate_arguments(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    const auto io_size = num_inputs + num_outputs;
    OV_CPU_JIT_EMITTER_ASSERT(out.size() == 0, "Invalid number of out arguments: expected ", 0, " got ", out.size());
    OV_CPU_JIT_EMITTER_ASSERT(in.size() == io_size + 1, "Invalid number of in arguments: expected ", io_size + 1, " got ", in.size());
    OV_CPU_JIT_EMITTER_ASSERT(is_incremented.size() == io_size, "Invalid is_incremented size: expected ", io_size, " got ", is_incremented.size());
    OV_CPU_JIT_EMITTER_ASSERT(ptr_increments.size() == io_size, "Invalid ptr_increments size: expected ", io_size, " got ", ptr_increments.size());
    OV_CPU_JIT_EMITTER_ASSERT(finalization_offsets.size() == io_size,
                              "Invalid finalization_offsets size: expected: ", io_size, " got ", finalization_offsets.size());
    OV_CPU_JIT_EMITTER_ASSERT(data_sizes.size() == io_size, "Invalid data_sizes size: expected: ", io_size, " got ", data_sizes.size());
    OV_CPU_JIT_EMITTER_ASSERT(loop_end_label != nullptr && loop_begin_label != nullptr, "has not inited labels!");
    OV_CPU_JIT_EMITTER_ASSERT(!snippets::utils::is_dynamic_value(wa_increment) || evaluate_once,
                              "loop increment might be dynamic only if loop evaluates once!");
}

void jit_loop_end_emitter::emit_code(const std::vector<size_t> &in, const std::vector<size_t> &out,
                                     const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs) const {
    validate_arguments(in, out);
    jit_emitter::emit_code(in, out, pool_vec_idxs, pool_gpr_idxs);
}

void jit_loop_end_emitter::emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    std::vector<size_t> data_ptr_reg_idxs;
    // the last input is actually a work_amount reg
    data_ptr_reg_idxs.reserve(num_inputs + num_outputs);
    std::copy(in.begin(), in.end() - 1, std::back_inserter(data_ptr_reg_idxs));

    auto apply_increments = [&](bool use_runtime_args, size_t field_offset, const std::vector<int64_t>& increments, size_t scale) {
        Reg64 reg_increments;
        auto add_increments = [&]() {
            for (size_t idx = 0; idx < data_ptr_reg_idxs.size(); idx++) {
                const auto& increment = increments[idx];
                if (is_incremented[idx] && increment != 0) {
                    if (ov::snippets::utils::is_dynamic_value(increment)) {
                        OV_CPU_JIT_EMITTER_ASSERT(use_runtime_args, "Loop argument structure cannot be pushed to aux GPR");
                        h->add(Reg64(static_cast<int>(data_ptr_reg_idxs[idx])), h->ptr[reg_increments + idx * sizeof(int64_t)]);
                    } else {
                        h->add(Reg64(static_cast<int>(data_ptr_reg_idxs[idx])), increment * scale * data_sizes[idx]);
                    }
                }
            }
        };

        const auto id_offset = loop_id * sizeof(jit_snippets_call_args::loop_args_t);
        if (use_runtime_args) {
            jit_aux_gpr_holder gpr_holder(h, aux_gpr_idxs, in); // loop_end has only input registers
            reg_increments = gpr_holder.get_reg();
            h->mov(reg_increments, h->ptr[abi_param1 + GET_OFF(loop_args)]);
            h->mov(reg_increments, h->ptr[reg_increments + id_offset + field_offset]);
            add_increments();
        } else {
            add_increments();
        }
    };

    if (!evaluate_once) {
        apply_increments(are_ptr_increments_dynamic, GET_OFF_LOOP_ARGS(m_ptr_increments), ptr_increments, wa_increment);

        Reg64 reg_work_amount = Reg64(in.back());
        h->sub(reg_work_amount, wa_increment);
        h->cmp(reg_work_amount, wa_increment);
        h->jge(*loop_begin_label, Xbyak::CodeGenerator::T_NEAR);
    }

    apply_increments(are_final_offsets_dynamic, GET_OFF_LOOP_ARGS(m_finalization_offsets), finalization_offsets, 1);

    h->L(*loop_end_label);
}

/* ============================================================== */

}   // namespace intel_cpu
}   // namespace ov
