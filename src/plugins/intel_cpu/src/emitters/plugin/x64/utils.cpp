// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.hpp"

#include "emitters/utils.hpp"
#include "snippets/utils/utils.hpp"

namespace ov {
namespace intel_cpu {

using namespace Xbyak;
using namespace dnnl::impl::cpu::x64;

inline snippets::Reg Xbyak2SnippetsReg(const Xbyak::Reg& xb_reg) {
    auto get_reg_type = [](const Xbyak::Reg& xb_reg) {
        switch (xb_reg.getKind()) {
            case Xbyak::Reg::REG:
                return snippets::RegType::gpr;
            case Xbyak::Reg::XMM:
            case Xbyak::Reg::YMM:
            case Xbyak::Reg::ZMM:
                return snippets::RegType::vec;
            case Xbyak::Reg::OPMASK:
                return snippets::RegType::mask;
            default:
                OPENVINO_THROW("Unhandled Xbyak reg type in conversion to snippets reg type");
        }
    };
    return {get_reg_type(xb_reg), static_cast<size_t>(xb_reg.getIdx())};
}
template<dnnl::impl::cpu::x64::cpu_isa_t isa>
EmitABIRegSpills<isa>::EmitABIRegSpills(jit_generator* h_arg, const std::set<snippets::Reg>& live_regs) : h(h_arg) {
    // all regs to spill according to ABI
    m_regs_to_spill = {h->r8, h->r9, h->r10, h->r11, h->r12, h->r13, h->r14, h->r15,
                      h->rax, h->rbx, h->rcx, h->rdx, h->rdi, h->rsi, h->rbp};
    for (int i = 0; i < cpu_isa_traits<isa>::n_vregs; ++i)
        m_regs_to_spill.push_back(Vmm(i));
    if (isa == cpu_isa_t::avx512_core) {
        for (size_t i = 0; i < 8; ++i)
            m_regs_to_spill.push_back(Xbyak::Opmask(static_cast<int>(i)));
    }

    if (!live_regs.empty()) {
        auto not_live = [&live_regs](const Xbyak::Reg& reg) -> bool {
            return live_regs.count(Xbyak2SnippetsReg(reg)) == 0;
        };
        m_regs_to_spill.erase(std::remove_if(m_regs_to_spill.begin(), m_regs_to_spill.end(), not_live), m_regs_to_spill.end());
    }
    for (const auto& reg : m_regs_to_spill) {
        const auto reg_bit_size = reg.getBit();
        OPENVINO_ASSERT(reg_bit_size % 8 == 0, "Unexpected reg bit size");
        m_bytes_to_spill += reg_bit_size / 8;
    }
}

template<dnnl::impl::cpu::x64::cpu_isa_t isa>
EmitABIRegSpills<isa>::~EmitABIRegSpills() {
    OPENVINO_ASSERT(spill_status, "postamble or preamble is missed");
    OPENVINO_ASSERT(rsp_status, "rsp_align or rsp_restore is missed");
}

template<dnnl::impl::cpu::x64::cpu_isa_t isa>
void EmitABIRegSpills<isa>::preamble() {
    h->sub(h->rsp, m_bytes_to_spill);
    uint32_t byte_stack_offset = 0;
    for (const auto& reg : m_regs_to_spill) {
        Xbyak::Address addr = h->ptr[h->rsp + byte_stack_offset];
        byte_stack_offset += reg.getBit() / 8;
        switch (reg.getKind()) {
            case Xbyak::Reg::REG:
                h->mov(addr, reg);
                break;
            case Xbyak::Reg::XMM:
            case Xbyak::Reg::YMM:
            case Xbyak::Reg::ZMM:
                h->uni_vmovups(addr, Vmm(reg));
                break;
            case Xbyak::Reg::OPMASK:
                h->kmovq(addr, Xbyak::Opmask(reg.getIdx()));
                break;
            default:
                OPENVINO_THROW("Unhandled Xbyak reg type in conversion to snippets reg type");
        }
    }
    // Update the status
    spill_status = false;
}
template<dnnl::impl::cpu::x64::cpu_isa_t isa>
void EmitABIRegSpills<isa>::postamble() {
    uint32_t byte_stack_offset = m_bytes_to_spill;
    for (size_t i = m_regs_to_spill.size(); i > 0; i--) {
        const auto& reg = m_regs_to_spill[i - 1];
        byte_stack_offset -= reg.getBit() / 8;
        Xbyak::Address addr =  h->ptr[h->rsp + byte_stack_offset];
        switch (reg.getKind()) {
            case Xbyak::Reg::REG:
                h->mov(reg, addr);
                break;
            case Xbyak::Reg::XMM:
            case Xbyak::Reg::YMM:
            case Xbyak::Reg::ZMM:
                h->uni_vmovups(Vmm(reg), addr);
                break;
            case Xbyak::Reg::OPMASK:
                h->kmovq(Xbyak::Opmask(reg), addr);
                break;
            default:
                OPENVINO_THROW("Unhandled Xbyak reg type in conversion to snippets reg type");
        }
    }
    h->add(h->rsp, m_bytes_to_spill);
    // Update the status
    spill_status = true;
}
template<dnnl::impl::cpu::x64::cpu_isa_t isa>
void EmitABIRegSpills<isa>::rsp_align() {
    h->mov(h->rbx, h->rsp);
    h->and_(h->rbx, 0xf);
    h->sub(h->rsp, h->rbx);
#ifdef _WIN32
    // Allocate shadow space (home space) according to ABI
    h->sub(h->rsp, 32);
#endif

    // Update the status
    rsp_status = false;
}
template<dnnl::impl::cpu::x64::cpu_isa_t isa>
void EmitABIRegSpills<isa>::rsp_restore() {
#ifdef _WIN32
    // Release shadow space (home space)
    h->add(h->rsp, 32);
#endif
    h->add(h->rsp, h->rbx);

    // Update the status
    rsp_status = true;
}

cpu_isa_t EmitABIRegSpills::get_isa() {
    // need preserve based on cpu capability, instead of host isa.
    // in case there are possibilty that different isa emitters exist in one kernel from perf standpoint in the future.
    // e.g. other emitters isa is avx512, while this emitter isa is avx2, and internal call is used. Internal call may use avx512 and spoil k-reg, ZMM.
    // do not care about platform w/ avx512_common but w/o avx512_core(knight landing), which is obsoleted.
    if (mayiuse(avx512_core)) return avx512_core;
    if (mayiuse(avx2)) return avx2;
    if (mayiuse(sse41)) return sse41;
    OV_CPU_JIT_EMITTER_THROW("unsupported isa");
}

}   // namespace intel_cpu
}   // namespace ov
