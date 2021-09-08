// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "openvino/pass/pass.hpp"
#include "ngraph/opsets/opset8.hpp"

namespace ov {
    namespace pass {
        /**
         * @brief The transformation replaces the provided pairs Parameter and Result with ngraph Memory layers
         * ReadValue and Assign
         */
        class OPENVINO_API ReplaceInputsOutputsWithMemory : public FunctionPass {
        public:
            OPENVINO_RTTI_DECLARATION;

            using InOutPairs = std::vector<std::pair<std::shared_ptr<ngraph::opset8::Parameter>,
                                                     std::shared_ptr<ngraph::opset8::Result>>>;

            static InOutPairs findInputsOutputsByName(const std::shared_ptr<ngraph::Function>& func, const
                                               std::vector<std::pair<std::string, std::string>>& param_res_names);
            explicit ReplaceInputsOutputsWithMemory(const InOutPairs& pairs_to_replace) : m_pairs_to_replace(pairs_to_replace) {}
            bool run_on_function(std::shared_ptr<ngraph::Function> f) override;
        private:
            InOutPairs m_pairs_to_replace;
        };
    }  // namespace pass
}  // namespace ov
