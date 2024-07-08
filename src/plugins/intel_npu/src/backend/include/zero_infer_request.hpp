// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ze_api.h>
#include <ze_graph_ext.h>

#include <mutex>

#include "intel_npu/utils/logger/logger.hpp"
#include "npu.hpp"
#include "zero_executor.hpp"
#include "zero_pipeline.hpp"
#include "zero_profiling.hpp"
#include "zero_utils.hpp"
#include "zero_wrappers.hpp"

namespace {
constexpr std::size_t DEFAULT_BATCH_SIZE = 1;
}  // namespace

namespace intel_npu {

class ZeroInferRequest final : public SyncInferRequest {
public:
    explicit ZeroInferRequest(const std::shared_ptr<ZeroInitStructsHolder>& backendPtr,
                              const std::shared_ptr<const ICompiledModel>& compiledModel,
                              const std::shared_ptr<const IExecutor>& executor,
                              const Config& config);

    void infer() override;
    void infer_async() override;

    void get_result() override;

private:
    std::vector<ov::ProfilingInfo> get_profiling_info() const override;
    std::vector<uint8_t> get_raw_profiling_data() const;

    void check_network_precision(const ov::element::Type_t precision) override;

    /**
     * @brief Determines if batching can be addressed inside the plugin. In the positive case, the batch size used by
     * the model will also be deduced and returned.
     * @details Batching can be handled by the plugin only if:
     *  - The batch axis is the first axis.
     *  - The batch size received by the compiler takes the default value of 1.
     *  - The batch size found in the IR model matches for all inputs/outputs and takes a value different than the
     * default one.
     *
     * If any of the previous conditions is not fulfilled, the functon will return the default batch size, thus no
     * custom algorithm will be applied inside the plugin in order to address batching.
     *
     * @param metadata Metadata containing the shape values as seen by both the compiler and IR model. These will
     * ultimately be used for determining the batch size.
     * @returns The batch size deduced by the algorithm or the default value of 1 if batching cannot be performed inside
     * the plugin.
     */
    size_t getBatchSize(const NetworkMetadata& metadata);

    const std::shared_ptr<const IExecutor> _executorPtr;
    const ZeroExecutor* _executor;
    const Config _config;
    Logger _logger;

    zeroProfiling::ProfilingPool _profilingPool;
    zeroProfiling::ProfilingQuery _profilingQuery;
    std::shared_ptr<zeroProfiling::NpuInferProfiling> _npuProfiling;
    std::unique_ptr<Pipeline> _pipeline;

    // If batching is handled on the compiler side then batching on the plugin shall be set to 1, we don't do any
    // specific operations on the plugin in this case.
    size_t _batchSize = DEFAULT_BATCH_SIZE;
};

}  //  namespace intel_npu
