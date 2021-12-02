// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/core_visibility.hpp"
#include "openvino/core/preprocess/input_network_info.hpp"
#include "openvino/core/preprocess/input_tensor_info.hpp"
#include "openvino/core/preprocess/preprocess_steps.hpp"

namespace ov {
namespace preprocess {

/// \brief Class holding preprocessing information for one input
/// From preprocessing pipeline perspective, each input can be represented as:
///    - User's input parameter info (InputInfo::tensor)
///    - Preprocessing steps applied to user's input (InputInfo::preprocess)
///    - Network's input info, which is a final info after preprocessing (InputInfo::network)
///
class OPENVINO_API InputInfo final {
    class InputInfoImpl;
    std::unique_ptr<InputInfoImpl> m_impl;
    friend class PrePostProcessor;

    /// \brief Empty constructor for internal usage
    InputInfo();

public:
    /// \brief Move constructor
    InputInfo(InputInfo&& other) noexcept;

    /// \brief Move assignment operator
    InputInfo& operator=(InputInfo&& other) noexcept;

    /// \brief Default destructor
    ~InputInfo();

    /// \brief Get current input tensor information with ability to change specific data
    ///
    /// \return Reference to current input tensor structure
    InputTensorInfo& tensor();

    /// \brief Get current input preprocess information with ability to add more preprocessing steps
    ///
    /// \return Reference to current preprocess steps structure
    PreProcessSteps& preprocess();

    /// \brief Get current input network/model information with ability to change original network's input data
    ///
    /// \return Reference to current network's input information structure
    InputNetworkInfo& network();
};

}  // namespace preprocess
}  // namespace ov
