// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/multinomial.hpp"

#include <openvino/core/type/element_type.hpp>
#include <openvino/runtime/tensor.hpp>

namespace {

using ov::test::MultinomialLayerTest;

std::vector<std::pair<uint64_t, uint64_t>> global_op_seed = {{1ul, 2ul}, {0ul, 0ul}};

std::vector<float> probs_4x4_f32 = {0.00001f,
                                    0.001f,
                                    0.1f,
                                    10.0f,
                                    0.001f,
                                    0.00001f,
                                    10.0f,
                                    0.1f,
                                    0.1f,
                                    10.0f,
                                    0.00001f,
                                    0.001f,
                                    10.0f,
                                    0.1f,
                                    0.001f,
                                    0.00001f};

std::vector<ov::float16> probs_2x3_f16 = {ov::float16(0.001f),
                                          ov::float16(0.1f),
                                          ov::float16(10.0f),
                                          ov::float16(10.0f),
                                          ov::float16(0.001f),
                                          ov::float16(0.1f)};

std::vector<ov::bfloat16> probs_1x3_bf16 = {ov::bfloat16(0.1f), ov::bfloat16(1.0f), ov::bfloat16(10.0f)};

std::vector<float> probs_4x4_f32_log =
    {3.0f, 6.0f, 10.0f, 0.0f, 3.0f, 0.0f, 10.0f, 6.0f, 6.0f, 10.0f, 0.0f, 3.0f, 10.0f, 6.0f, 3.0f, 0.0f};

std::vector<ov::float16> probs_2x3_f16_log = {ov::float16(3.0f),
                                              ov::float16(6.0f),
                                              ov::float16(10.0f),
                                              ov::float16(10.0f),
                                              ov::float16(3.0f),
                                              ov::float16(6.0f)};

std::vector<ov::bfloat16> probs_1x3_bf16_log = {ov::bfloat16(3.0f), ov::bfloat16(6.0f), ov::bfloat16(10.0f)};

std::vector<int> num_samples_scalar_i32 = {1};
std::vector<int64_t> num_samples_1x1_i64 = {2};
std::vector<int64_t> num_samples_scalar_i64 = {3};

const std::vector<ov::Tensor> probs = {ov::Tensor(ov::element::f32, {4, 4}, probs_4x4_f32.data()),
                                       ov::Tensor(ov::element::f16, {2, 3}, probs_2x3_f16.data()),
                                       ov::Tensor(ov::element::bf16, {1, 3}, probs_1x3_bf16.data())};

const std::vector<ov::Tensor> probs_log = {ov::Tensor(ov::element::f32, {4, 4}, probs_4x4_f32_log.data()),
                                           ov::Tensor(ov::element::f16, {2, 3}, probs_2x3_f16_log.data()),
                                           ov::Tensor(ov::element::bf16, {1, 3}, probs_1x3_bf16_log.data())};

const std::vector<ov::Tensor> num_samples = {ov::Tensor(ov::element::i32, {}, num_samples_scalar_i32.data()),
                                             ov::Tensor(ov::element::i64, {1}, num_samples_1x1_i64.data()),
                                             ov::Tensor(ov::element::i64, {}, num_samples_scalar_i64.data())};

const std::vector<ov::test::ElementType> convert_type = {ov::test::ElementType::i32};

const std::vector<bool> with_replacement = {
    // true,
    false};

const auto params_static = ::testing::Combine(::testing::Values("static"),
                                              ::testing::ValuesIn(probs),
                                              ::testing::ValuesIn(num_samples),
                                              ::testing::ValuesIn(convert_type),
                                              ::testing::ValuesIn(with_replacement),
                                              ::testing::Values(false),  // log_probs
                                              ::testing::ValuesIn(global_op_seed),
                                              ::testing::Values(ov::test::utils::DEVICE_CPU));

const auto params_static_log = ::testing::Combine(::testing::Values("static"),
                                                  ::testing::ValuesIn(probs_log),
                                                  ::testing::ValuesIn(num_samples),
                                                  ::testing::ValuesIn(convert_type),
                                                  ::testing::ValuesIn(with_replacement),
                                                  ::testing::Values(true),  // log_probs
                                                  ::testing::ValuesIn(global_op_seed),
                                                  ::testing::Values(ov::test::utils::DEVICE_CPU));

const auto params_dynamic = ::testing::Combine(::testing::Values("dynamic"),
                                               ::testing::ValuesIn(probs),
                                               ::testing::ValuesIn(num_samples),
                                               ::testing::ValuesIn(convert_type),
                                               ::testing::ValuesIn(with_replacement),
                                               ::testing::Values(false),  // log_probs
                                               ::testing::ValuesIn(global_op_seed),
                                               ::testing::Values(ov::test::utils::DEVICE_CPU));

const auto params_dynamic_log = ::testing::Combine(::testing::Values("dynamic"),
                                                   ::testing::ValuesIn(probs_log),
                                                   ::testing::ValuesIn(num_samples),
                                                   ::testing::ValuesIn(convert_type),
                                                   ::testing::ValuesIn(with_replacement),
                                                   ::testing::Values(true),  // log_probs
                                                   ::testing::ValuesIn(global_op_seed),
                                                   ::testing::Values(ov::test::utils::DEVICE_CPU));

INSTANTIATE_TEST_SUITE_P(smoke_MultinomialStatic,
                         MultinomialLayerTest,
                         params_static,
                         MultinomialLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MultinomialStaticLog,
                         MultinomialLayerTest,
                         params_static_log,
                         MultinomialLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MultinomialDynamic,
                         MultinomialLayerTest,
                         params_dynamic,
                         MultinomialLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MultinomialDynamicLog,
                         MultinomialLayerTest,
                         params_dynamic_log,
                         MultinomialLayerTest::getTestCaseName);
}  // namespace
