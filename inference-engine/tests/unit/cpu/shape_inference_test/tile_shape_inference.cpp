// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <tile_shape_inference.hpp>
#include <openvino/core/coordinate_diff.hpp>
#include <openvino/op/ops.hpp>
#include <openvino/op/parameter.hpp>

#include "utils/shape_inference/static_shape.hpp"

using namespace ov;

TEST(StaticShapeInferenceTest, TileTest) {
    auto param0 = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1});
    auto param1 = std::make_shared<ov::op::v0::Constant>(element::i64, ov::Shape{3}, std::vector<int>{3, 4, 1});
    auto tile = std::make_shared<op::v0::Tile>(param0, param1);
    //Test Partial Shape
    std::vector<PartialShape> input_shapes = {PartialShape{6, 8, 10}, PartialShape{3}},
                              output_shapes = {PartialShape{}};
    shape_infer(tile.get(), input_shapes, output_shapes);
    ASSERT_EQ(output_shapes[0], PartialShape({18, 32, 10}));
    //Test Static Shape
    std::vector<StaticShape> static_input_shapes = {StaticShape{6, 8, 10}, StaticShape{3}},
                             static_output_shapes = {StaticShape{}};
    shape_infer(tile.get(), static_input_shapes, static_output_shapes);
    ASSERT_EQ(static_output_shapes[0], StaticShape({18, 32, 10}));
    //Test Wrong Static Shape
    std::vector<StaticShape> wrong_static_input_shapes = {StaticShape{6, 8, 10}, StaticShape{}},
                             wrong_static_output_shapes = {StaticShape{}};

    ASSERT_THROW(shape_infer(tile.get(),
                             wrong_static_input_shapes,
                             wrong_static_output_shapes),
                 ov::AssertFailure);
}