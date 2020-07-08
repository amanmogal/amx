// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>
#include <cpp_interfaces/impl/ie_memory_state_internal.hpp>
#include "gna_plugin.hpp"

namespace  GNAPluginNS {
namespace memory {
class GNAMemoryState : public InferenceEngine::IMemoryStateInternal {
 public:
    GNAMemoryState(std::string name, GNAMemoryLayer state)
        : name(name), state(state) { }

    void Reset() override;
    void SetState(InferenceEngine::Blob::Ptr newState) override;
    InferenceEngine::Blob::CPtr GetLastState() const override;
    std::string GetName() const override;

private:
    GNAMemoryLayer state;
    std::string name;
};
}  // namespace memory
}  // namespace GNAPluginNS
