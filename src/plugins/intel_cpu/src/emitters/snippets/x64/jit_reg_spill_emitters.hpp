// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "emitters/plugin/x64/jit_emitter.hpp"

namespace ov {
namespace intel_cpu {

/* ================== jit_reg_spill_begin_emitters ====================== */

class jit_reg_spill_begin_emitters: public jit_emitter {
public:
    jit_reg_spill_begin_emitters(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                                const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_num() const override { return 0; }

    void emit_code(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs,
                  const std::vector<size_t> &pool_vec_idxs = {}, const std::vector<size_t> &pool_gpr_idxs = {}) const override;

protected:
    void validate_arguments(const std::vector<size_t> &in, const std::vector<size_t> &out) const override;
    void emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;
    const ov::snippets::lowered::ExpressionPtr m_reg_spill_begin_expr;
};


/* ============================================================== */

/* ================== jit_loop_end_emitter ====================== */

class jit_loop_end_emitter: public jit_emitter {
public:
    jit_loop_end_emitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                           const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_num() const override { return 0; }

    void emit_code(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs,
                   const std::vector<size_t> &pool_vec_idxs = {}, const std::vector<size_t> &pool_gpr_idxs = {}) const override;

protected:
    void validate_arguments(const std::vector<size_t> &in, const std::vector<size_t> &out) const override;
    void emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;

    // `jit_loop_end_emitter` handles manually aux_gpr allocation using `jit_aux_gpr_holder`
    size_t aux_gprs_count() const override { return 0; }

    static ov::snippets::lowered::ExpressionPtr get_loop_begin_expr(const ov::snippets::lowered::ExpressionPtr& expr);

    std::shared_ptr<const Xbyak::Label> loop_begin_label = nullptr;
};

/* ============================================================== */

}   // namespace intel_cpu
}   // namespace ov
