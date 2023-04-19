// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../eltwise.hpp"
#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "acl_utils.hpp"

namespace ov {
namespace intel_cpu {

class AclEltwiseExecutor : public EltwiseExecutor {
public:
    AclEltwiseExecutor(const ExecutorContext::CPtr context);

    bool init(const EltwiseAttrs& eltwiseAttrs,
              const std::vector<MemoryDescPtr>& srcDescs,
              const std::vector<MemoryDescPtr>& dstDescs,
              const std::vector<EltwisePostOp>& postOps) override;

    void exec(const std::vector<MemoryCPtr>& src,
              const std::vector<MemoryPtr>& dst,
              const void *post_ops_data_) override;

    impl_desc_type getImplType() const override {
        return implType;
    }
private:
    EltwiseAttrs aclEltwiseAttrs{};
    impl_desc_type implType = impl_desc_type::acl;
    std::vector<arm_compute::Tensor> srcTensors, dstTensors;
    std::function<void()> exec_func;
};

class AclEltwiseExecutorBuilder : public EltwiseExecutorBuilder {
public:
    bool isSupported(const EltwiseAttrs& eltwiseAttrs,
                     const std::vector<MemoryDescPtr>& srcDescs,
                     const std::vector<MemoryDescPtr>& dstDescs) const override {
        switch (eltwiseAttrs.algorithm) {
            case Algorithm::EltwiseAdd:
            case Algorithm::EltwiseMultiply:
            case Algorithm::EltwiseSubtract:
            case Algorithm::EltwiseDivide:
            case Algorithm::EltwiseMaximum:
            case Algorithm::EltwiseMinimum:
            case Algorithm::EltwiseSquaredDifference:
            case Algorithm::EltwisePowerDynamic:
            case Algorithm::EltwiseEqual:
            case Algorithm::EltwiseNotEqual:
            case Algorithm::EltwiseGreater:
            case Algorithm::EltwiseGreaterEqual:
            case Algorithm::EltwiseLess:
            case Algorithm::EltwiseLessEqual:
            case Algorithm::EltwiseRelu:
            case Algorithm::EltwiseGeluErf:
            case Algorithm::EltwiseElu:
            case Algorithm::EltwiseTanh:
            case Algorithm::EltwiseSigmoid:
            case Algorithm::EltwiseAbs:
            case Algorithm::EltwiseSqrt:
            case Algorithm::EltwiseSoftRelu:
            case Algorithm::EltwiseExp:
            case Algorithm::EltwiseClamp:
            case Algorithm::EltwiseSwish:
            case Algorithm::EltwisePrelu:
            case Algorithm::EltwiseHswish:
            case Algorithm::EltwiseLog:
                break;
            default:
                return false;
        }

        // ACL supports only U8 precision on output for comparison operations
        if (one_of(eltwiseAttrs.algorithm, Algorithm::EltwiseEqual, Algorithm::EltwiseNotEqual, Algorithm::EltwiseGreater,
                                           Algorithm::EltwiseGreaterEqual, Algorithm::EltwiseLess, Algorithm::EltwiseLessEqual)) {
            if (dstDescs[0]->getPrecision() != InferenceEngine::Precision::U8) {
                return false;
            }
        }
        for (const auto &srcD : srcDescs) {
            for (const auto &dstD : dstDescs) {
                if ((srcD->getPrecision() != InferenceEngine::Precision::FP32 &&
                     srcD->getPrecision() != InferenceEngine::Precision::FP16) ||
                     srcD->getPrecision() != dstD->getPrecision())
                    return false;
            }
        }

        for (int i = 0; i < srcDescs.size(); i++) {
            if (getAclDataLayoutByMemoryDesc(srcDescs[i]) == arm_compute::DataLayout::UNKNOWN)
                 return false;
        }
        for (int i = 0; i < dstDescs.size(); i++) {
            if (getAclDataLayoutByMemoryDesc(dstDescs[i]) == arm_compute::DataLayout::UNKNOWN)
                return false;
        }

        return true;
    }

    EltwiseExecutorPtr makeExecutor(const ExecutorContext::CPtr context) const override {
        return std::make_shared<AclEltwiseExecutor>(context);
    }
};

}   // namespace intel_cpu
}   // namespace ov