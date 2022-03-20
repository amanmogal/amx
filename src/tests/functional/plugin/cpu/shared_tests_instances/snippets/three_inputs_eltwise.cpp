// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/three_inputs_eltwise.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

    const std::vector<InferenceEngine::Precision> netPrecisions = {
            InferenceEngine::Precision::FP32
    };
    INSTANTIATE_TEST_SUITE_P(smoke_Snippets_Eltwise, ThreeInputsEltwise,
                         ::testing::Combine(
                                 ::testing::ValuesIn(netPrecisions),
                                 ::testing::Values(InferenceEngine::SizeVector({1, 64, 10, 10})),
                                 ::testing::Values(InferenceEngine::SizeVector({1, 64, 10,  1})),
                                 ::testing::Values(InferenceEngine::SizeVector({1, 1, 1,  10})),
                                 ::testing::Values(2), // eltwises fuse only for non-broadcasted shapes
                                 ::testing::Values(0), // SnippetsMarkSkipped disables tokenization for eltwise chains after inputs
                                 ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                             ThreeInputsEltwise::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_Snippets_Eltwise, ThreeInputsEltwiseConvert,
            ::testing::Combine(
            ::testing::ValuesIn(netPrecisions),
            ::testing::Values(InferenceEngine::SizeVector({1, 64, 10, 10})),
            ::testing::Values(InferenceEngine::SizeVector({1, 64, 10,  1})),
            ::testing::Values(InferenceEngine::SizeVector({1, 1, 1,  10})),
            ::testing::Values(4), // Subgraph + 3 converts after inputs
            ::testing::Values(1), // Subgraph is created, since the inputs are followed by converts
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                             ThreeInputsEltwise::getTestCaseName);

}  // namespace