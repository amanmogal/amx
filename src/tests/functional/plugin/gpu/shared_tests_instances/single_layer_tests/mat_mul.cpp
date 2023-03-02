// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/mat_mul.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> inputPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
};

const std::vector<ShapeRelatedParams> shapeRelatedParams = {
        { { {2, 1, 1, 5, 6}, false }, { {1, 1, 6, 4}, false } },
        { { {2, 2, 4, 16}, true }, { {1, 1, 1, 4}, true } },
        { { {2, 1, 2, 3, 5, 6}, false }, { {1, 1, 6, 4}, false } },
        { { {1, 4, 5, 6}, false }, { {1, 4, 6, 4}, false } },
        { { {1, 16, 128}, false }, { {1, 64, 128}, true } },
        { { {4, 5, 6}, false }, { {6, 3}, false } },
        { { {9, 9, 9}, false }, { {9, 9}, false } },
        { { {1, 2, 3}, false }, { {1, 1, 3, 2}, false } },
        { { {1, 3, 2, 4}, false }, { {2, 1, 4, 2}, false } },
        { { {2, 1, 2, 4}, false }, { {1, 3, 4, 2}, false } },
        { { {3, 2, 4}, false }, { {2, 1, 4, 2}, false } },
        { { {2, 1, 4, 2}, false }, { {3, 2, 4}, false } },
        { { {2, 1, 2, 3}, true }, { {3, 2, 4}, false } },
        { { {2, 1, 3, 2}, false }, { {3, 4, 2}, true } },
        { { {2, 1, 2, 3}, true }, { {3, 4, 2}, true } },
        { { {1, 64, 80}, false }, { {1, 77, 80}, true } },
        { { {3}, false }, { {2, 2, 3, 1}, false } },
        { { {2, 2, 1, 3}, false }, { {3}, false } },
        { { {65, 100}, false }, { {73, 100}, true } },
        { { {100, 65}, true }, { {100, 73}, false } },
        { { {100, 65}, true }, { {73, 100}, true } },
        { { {1, 5}, false }, { {5, 1}, false } },
        { { {5, 1}, true }, { {5, 1}, false } },
        { { {1, 5}, false }, { {1, 5}, true } },
        { { {1, 5}, false }, { {5}, false } },
        { { {5}, false }, { {5, 1}, false } },
        { { {5}, false }, { {5}, false } },
        { { {5}, true }, { {5}, true } }
};

std::vector<ngraph::helpers::InputLayerType> secondaryInputTypes = {
        ngraph::helpers::InputLayerType::CONSTANT,
        ngraph::helpers::InputLayerType::PARAMETER,
};

std::map<std::string, std::string> additional_config = {};

INSTANTIATE_TEST_SUITE_P(smoke_MatMul, MatMulTest,
        ::testing::Combine(
                ::testing::ValuesIn(shapeRelatedParams),
                ::testing::ValuesIn(inputPrecisions),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::ValuesIn(secondaryInputTypes),
                ::testing::Values(CommonTestUtils::DEVICE_GPU),
                ::testing::Values(additional_config)),
        MatMulTest::getTestCaseName);

const size_t max_batch_value = 256;

std::vector<ShapeRelatedParams> generateInputShapes() {
        std::vector<ShapeRelatedParams> inputShapes;
        for (size_t i = 2; i < max_batch_value; i *= 2) {
                inputShapes.push_back({ { {i, 4, 5, 6}, false }, { {1, 4, 6, 4}, false } });
                inputShapes.push_back({ { {1, 4, 5, 6}, false }, { {i, 4, 6, 4}, false } });
                inputShapes.push_back({ { {i, 4, 5, 6}, false }, { {i, 4, 6, 4}, false } });

                inputShapes.push_back({ { {i, 3, 2, 4}, false }, { {1, 1, 4, 2}, false } });
                inputShapes.push_back({ { {1, 3, 2, 4}, false }, { {i, 1, 4, 2}, false } });
                inputShapes.push_back({ { {i, 3, 2, 4}, false }, { {i, 1, 4, 2}, false } });

                inputShapes.push_back({ { {i, 1, 2, 4}, false }, { {1, 3, 4, 2}, false } });
                inputShapes.push_back({ { {1, 1, 2, 4}, false }, { {i, 3, 4, 2}, false } });
                inputShapes.push_back({ { {i, 1, 2, 4}, false }, { {i, 3, 4, 2}, false } });

                inputShapes.push_back({ { {3, 2, 4}, false }, { {i, 1, 4, 2}, false } });
                inputShapes.push_back({ { {i, 1, 4, 2}, false }, { {3, 2, 4}, false } });
                inputShapes.push_back({ { {i, 1, 2, 3}, true }, { {3, 2, 4}, false } });
                inputShapes.push_back({ { {i, 1, 3, 2}, false }, { {3, 4, 2}, true } });
                inputShapes.push_back({ { {i, 1, 2, 3}, true }, { {3, 4, 2}, true } });
                inputShapes.push_back({ { {i, 2, 1, 3}, false }, { {3}, false } });
        }
        return inputShapes;
}

const std::vector<ShapeRelatedParams> nightlyInputShapes(generateInputShapes());

INSTANTIATE_TEST_SUITE_P(nightly_MatMul, MatMulTest,
        ::testing::Combine(
                ::testing::ValuesIn(nightlyInputShapes),
                ::testing::ValuesIn(inputPrecisions),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::ValuesIn(secondaryInputTypes),
                ::testing::Values(CommonTestUtils::DEVICE_GPU),
                ::testing::Values(additional_config)),
        MatMulTest::getTestCaseName);

} // namespace
