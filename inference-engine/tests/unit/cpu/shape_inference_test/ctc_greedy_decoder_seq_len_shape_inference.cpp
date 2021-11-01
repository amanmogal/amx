// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils/shape_inference/static_shape.hpp"
#include <ctc_greedy_decoder_seq_len_shape_inference.hpp>
#include <gtest/gtest.h>
#include <openvino/core/coordinate_diff.hpp>
#include <openvino/op/ops.hpp>
#include <openvino/op/parameter.hpp>
#include "utils/shape_inference/static_shape.hpp"

using namespace ov;

TEST(StaticShapeInferenceTest, CtcGreedyDecoderSeqLenTest) {
    auto P = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1});
    auto I = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{-1});
    auto G = std::make_shared<op::v6::CTCGreedyDecoderSeqLen>(P, I);
    std::vector<PartialShape> input_shapes = {PartialShape{3, 100, 1200}, PartialShape{3}},
                              output_shapes = {PartialShape{}, PartialShape{}};
    shape_infer(G.get(), input_shapes, output_shapes);
    ASSERT_EQ(output_shapes[0], PartialShape({3, 100}));
    ASSERT_EQ(output_shapes[1], PartialShape({3}));

    std::vector<StaticShape> static_input_shapes = {StaticShape{3, 100, 1200}, StaticShape{3}},
                             static_output_shapes = {StaticShape{}, StaticShape{}};
    shape_infer(G.get(), static_input_shapes, static_output_shapes);
    ASSERT_EQ(static_output_shapes[0], StaticShape({3, 100}));
    ASSERT_EQ(static_output_shapes[1], StaticShape({3}));
}