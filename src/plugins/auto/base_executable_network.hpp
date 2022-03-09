// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <map>
#include <vector>
#include <string>

#include "cpp_interfaces/impl/ie_executable_network_thread_safe_default.hpp"
#include "base_schedule.hpp"

#ifdef  MULTIUNITTEST
#define MOCKTESTMACRO virtual
#define MultiDevicePlugin MockMultiDevicePlugin
#else
#define MOCKTESTMACRO
#endif

namespace MultiDevicePlugin {
class BaseExecutableNetwork : public
    InferenceEngine::ExecutableNetworkThreadSafeDefault {
public:
    using Ptr = std::shared_ptr<BaseExecutableNetwork>;
    BaseExecutableNetwork(const Schedule::Ptr& schedule,
        const Context::Ptr& context);
    void SetConfig(const std::map<std::string, InferenceEngine::Parameter>& config)
    override;
    InferenceEngine::Parameter GetConfig(const std::string& name) const override;
    InferenceEngine::Parameter GetMetric(const std::string& name) const override;
    IInferPtr CreateInferRequest() override;
    IInferPtr CreateInferRequestImpl(InferenceEngine::InputsDataMap networkInputs,
        InferenceEngine::OutputsDataMap networkOutputs) override;
    IInferPtr CreateInferRequestImpl(const
        std::vector<std::shared_ptr<const ov::Node>>& inputs,
        const std::vector<std::shared_ptr<const ov::Node>>& outputs) override;
    std::shared_ptr<InferenceEngine::RemoteContext> GetContext() const override;
    ~BaseExecutableNetwork() override;

protected:
    Schedule::Ptr _schedule;
    Context::Ptr _context;
    InferenceEngine::SoExecutableNetworkInternal _executableNetwork;

private:
    void SetExeNetworkForContext();

private:
    std::once_flag _oc;
};

}  // namespace MultiDevicePlugin
