// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "node.h"

#include "emitters/snippets/cpu_runtime_configurator.hpp"
#include "emitters/snippets/x64/cpu_generator.hpp"
#include "snippets/op/subgraph.hpp"

#include <array>

namespace ov {
namespace intel_cpu {
namespace node {

class Subgraph : public Node {
    class SubgraphCodeGenerator;
    class SubgraphExecutor;
    class SubgraphJitExecutor;
    class SubgraphJitStaticExecutor;
    class SubgraphJitShapeAgnosticExecutor;
    class SubgraphJitDynamicSpecializedExecutor;
public:
    Subgraph(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);
    ~Subgraph() override = default;

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void selectOptimalPrimitiveDescriptor() override;
    ov::element::Type getRuntimePrecision() const override;

    void createPrimitive() override;
    void prepareParams() override;

    bool canBeInPlace() const override;
    bool created() const override;

    // if generator is set, it would execute generated code otherwise it would fallback to nGraph reference
    void execute(dnnl::stream strm) override;
    void executeDynamicImpl(dnnl::stream strm) override;

    struct SubgraphAttrs {
        // Local copy of subgraph node for canonization & code generation
        std::shared_ptr<snippets::op::Subgraph> snippet;
        uint64_t bodyHash;
        std::vector<VectorDims> inMemOrders;
        std::vector<VectorDims> outMemOrders;
        std::vector<ov::element::Type> inMemPrecs;
        std::vector<ov::element::Type> outMemPrecs;
    };

protected:
    IShapeInfer::Result shapeInfer() const override;

private:
    void init_memory_ptrs();
    void init_attrs();
    void init_start_offsets();
    void init_plugin_blocked_shapes() const;
    void init_snippets_blocked_shapes(snippets::op::Subgraph::BlockedShapeVector& in_blocked_shapes) const;
    void init_precisions(std::vector<ov::element::Type>& input_types, std::vector<ov::element::Type>& output_types) const;
    void lower();

    static uint64_t get_body_hash(const std::shared_ptr<snippets::op::Subgraph>& snippet);

    uint8_t get_broadcasting_mask(const std::vector<VectorDims>& input_shapes) const;

    using DataFlowPasses = std::vector<ov::snippets::pass::Manager::PositionedPassBase>;
    using ControlFlowPasses = std::vector<ov::snippets::lowered::pass::PassPipeline::PositionedPassLowered>;
    using ControlFlowConfig = std::shared_ptr<ov::snippets::lowered::pass::PassConfig>;

    DataFlowPasses get_data_flow_passes() const;
    std::pair<ControlFlowConfig, ControlFlowPasses> get_control_flow_passes() const;

    // Holds ISA version used is codeGeneration target
    dnnl::impl::cpu::x64::cpu_isa_t host_isa;
    std::shared_ptr<SubgraphAttrs> snippetAttrs;

    size_t input_num = 0;
    size_t output_num = 0;

    std::vector<MemoryPtr> srcMemPtrs = {};
    std::vector<MemoryPtr> dstMemPtrs = {};

    std::vector<ptrdiff_t> start_offset_in = {};
    std::vector<ptrdiff_t> start_offset_out = {};

    bool is_dynamic = false;
    // Input shapes that are used in PrepareParams and ShapeInfer to avoid frequent memory allocation
    mutable std::vector<VectorDims> in_shapes;

    std::shared_ptr<SubgraphExecutor> execPtr = nullptr;
};

// Class for snippet compilation
class Subgraph::SubgraphCodeGenerator {
public:
    SubgraphCodeGenerator(const std::shared_ptr<SubgraphAttrs>& snippet_attrs, const std::shared_ptr<CPURuntimeConfig>& config);

    const std::shared_ptr<snippets::Schedule>& get() const { return schedule; }

private:
    std::shared_ptr<snippets::Schedule> schedule;
};

// Base class for all Executors
class Subgraph::SubgraphExecutor {
public:
    SubgraphExecutor() = default;
    virtual ~SubgraphExecutor() = default;

    virtual void exec(const std::vector<MemoryPtr>& inMemPtrs, const std::vector<MemoryPtr>& outMemPtrs) = 0;
};

// Base class for Jit Executors
class Subgraph::SubgraphJitExecutor : public Subgraph::SubgraphExecutor {
public:
    SubgraphJitExecutor(const std::shared_ptr<SubgraphAttrs>& snippet_attrs,
                        const std::shared_ptr<SubgraphCodeGenerator>& snippet,
                        const std::vector<ptrdiff_t>& start_offset_in,
                        const std::vector<ptrdiff_t>& start_offset_out);
    virtual ~SubgraphJitExecutor() = default;

protected:
    void parallel_for6d(const std::function<void(jit_snippets_call_args&)>& initializer,
                        const std::function<void(jit_snippets_call_args&, const size_t*)>& caller);
    void parallel_forNd(const std::function<void(jit_snippets_call_args&)>& initializer,
                        const std::function<void(jit_snippets_call_args&, const size_t*)>& caller);

    virtual void init_runtime_params(const std::shared_ptr<CPURuntimeConfig>& cpu_config);

    std::shared_ptr<snippets::Schedule> m_schedule;
    // Holds index of output used as in execution domain
    // it should be compatible with a schedule's work size
    std::vector<size_t> m_parallel_exec_domain = {};
    size_t m_harness_work_amount = 0;

    // Buffer scratchpad
    std::vector<uint8_t> m_buffer_scratchpad = {};
    size_t m_buffer_scratchpad_size = 0;

    const size_t rank6D = 6;

    // Count of threads for parallel_nt
    int m_nthreads = 0;

    std::vector<ptrdiff_t> m_start_offset_in = {};
    std::vector<ptrdiff_t> m_start_offset_out = {};

#ifdef SNIPPETS_DEBUG_CAPS
    bool enabled_segfault_detector = false;
    inline void segfault_detector();
#endif
};

// Class for Subgraphs with static shapes
class Subgraph::SubgraphJitStaticExecutor : public Subgraph::SubgraphJitExecutor {
public:
    SubgraphJitStaticExecutor(const std::shared_ptr<SubgraphAttrs>& snippet_attrs,
                              const std::shared_ptr<SubgraphCodeGenerator>& snippet,
                              const std::vector<ptrdiff_t>& start_offset_in,
                              const std::vector<ptrdiff_t>& start_offset_out,
                              const std::shared_ptr<CPURuntimeConfig>& config);

    void exec(const std::vector<MemoryPtr>& inMemPtrs, const std::vector<MemoryPtr>& outMemPtrs) override;

protected:
    typedef void (*kernel)(const void*, const void*);

    inline void init_call_args(jit_snippets_call_args& call_args, const std::vector<MemoryPtr>& srcMemPtrs, const std::vector<MemoryPtr>& dstMemPtrs);
};

// Specialized dynamic executor based on shape agnostic kernel for the specific input shapes
class Subgraph::SubgraphJitDynamicSpecializedExecutor : public Subgraph::SubgraphJitExecutor {
public:
    SubgraphJitDynamicSpecializedExecutor(const std::shared_ptr<SubgraphAttrs>& snippet_attrs,
                                          const std::shared_ptr<SubgraphCodeGenerator>& snippet,
                                          const std::vector<ptrdiff_t>& start_offset_in,
                                          const std::vector<ptrdiff_t>& start_offset_out,
                                          const std::shared_ptr<CPURuntimeConfig>& config);

    void exec(const std::vector<MemoryPtr>& inMemPtrs, const std::vector<MemoryPtr>& outMemPtrs) override;

protected:
    typedef void (*dynamic_kernel)(const void *);

    inline void init_original_ptrs(const std::vector<MemoryPtr>& srcMemPtrs, const std::vector<MemoryPtr>& dstMemPtrs,
                                   std::vector<const uint8_t*>& src_ptrs, std::vector<uint8_t*>& dst_ptrs);
    inline void init_call_args(jit_snippets_call_args& call_args);
    inline void update_ptrs(jit_snippets_call_args& call_args, const std::vector<const uint8_t*>& src_ptrs,
                            const std::vector<uint8_t*>& dst_ptrs, const size_t* indexes) const;
    void init_runtime_params(const std::shared_ptr<CPURuntimeConfig>& cpu_config) override;

    std::vector<std::vector<size_t>> data_offsets = {};
    std::vector<jit_snippets_call_args::loop_args_t> loop_args = {};
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
