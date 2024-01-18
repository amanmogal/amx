// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/matrix_nms.hpp"

namespace {
using ov::test::MatrixNmsLayerTest;

const std::vector<std::vector<ov::Shape>> inStaticShapeParams = {{{3, 100, 4}, {3, 1, 100}},
                                                                 {{1, 10, 4}, {1, 100, 10}}};

const std::vector<ov::op::v8::MatrixNms::SortResultType> sortResultType = {ov::op::v8::MatrixNms::SortResultType::CLASSID,
                                                                           ov::op::v8::MatrixNms::SortResultType::SCORE,
                                                                           ov::op::v8::MatrixNms::SortResultType::NONE};
const std::vector<ov::element::Type> outType = {ov::element::i32, ov::element::i64};
const std::vector<ov::test::TopKParams> topKParams = {ov::test::TopKParams{-1, 5},
                                                      ov::test::TopKParams{100, -1}};
const std::vector<ov::test::ThresholdParams> thresholdParams = {
                                    ov::test::ThresholdParams{0.0f, 2.0f, 0.0f},
                                    ov::test::ThresholdParams{0.1f, 1.5f, 0.2f}};
const std::vector<int> backgroudClass = {-1, 1};
const std::vector<bool> normalized = {true, false};
const std::vector<ov::op::v8::MatrixNms::DecayFunction> decayFunction = {ov::op::v8::MatrixNms::DecayFunction::GAUSSIAN,
                                                                         ov::op::v8::MatrixNms::DecayFunction::LINEAR};

const auto nmsParamsStatic =
    ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inStaticShapeParams)),
                       ::testing::Values(ov::element::f32),
                       ::testing::ValuesIn(sortResultType),
                       ::testing::ValuesIn(outType),
                       ::testing::ValuesIn(topKParams),
                       ::testing::ValuesIn(thresholdParams),
                       ::testing::ValuesIn(backgroudClass),
                       ::testing::ValuesIn(normalized),
                       ::testing::ValuesIn(decayFunction),
                       ::testing::Values(ov::test::utils::DEVICE_GPU));

INSTANTIATE_TEST_SUITE_P(smoke_MatrixNmsLayerTest_static,
                         MatrixNmsLayerTest,
                         nmsParamsStatic,
                         MatrixNmsLayerTest::getTestCaseName);

} // namespace
