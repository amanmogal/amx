// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu/x64/jit_generator.hpp"
#include "snippets/emitter.hpp"

namespace ov {
namespace intel_cpu {

// The class emit register spills for the possible call of external binary code
template<dnnl::impl::cpu::x64::cpu_isa_t isa>
class EmitABIRegSpills {
public:
    EmitABIRegSpills(dnnl::impl::cpu::x64::jit_generator* h, const std::set<snippets::Reg>& live_regs = {});
    ~EmitABIRegSpills();

    // push (save) all registers on the stack
    void preamble();
    // pop (take) all registers from the stack
    void postamble();

    // align stack on 16-byte and allocate shadow space as ABI reqiures
    // callee is responsible to save and restore `rbx`. `rbx` must not be changed after call callee.
    void rsp_align();
    void rsp_restore();

private:
    EmitABIRegSpills() = default;

    dnnl::impl::cpu::x64::jit_generator* h {nullptr};

    std::vector<Xbyak::Reg> m_regs_to_spill;
    uint32_t m_bytes_to_spill = 0;
    using Vmm = typename dnnl::impl::cpu::x64::cpu_isa_traits<isa>::Vmm;

    bool spill_status = true;
    bool rsp_status = true;
};

get_EmitABIRegSpills() {
    if (dnnl::impl::cpu::x64::mayiuse(avx512_core)) return avx512_core;
    if (mayiuse(avx2)) return avx2;
    if (mayiuse(sse41)) return sse41;
    OV_CPU_JIT_EMITTER_THROW("unsupported isa");
}

}   // namespace intel_cpu
}   // namespace ov
