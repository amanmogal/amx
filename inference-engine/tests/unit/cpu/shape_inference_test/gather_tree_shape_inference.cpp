// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <gather_tree_shape_inference.hpp>
#include <openvino/core/coordinate_diff.hpp>
#include <openvino/op/gather_tree.hpp>
#include <openvino/op/ops.hpp>
#include <openvino/op/parameter.hpp>

#include "utils/shape_inference/static_shape.hpp"

using namespace ov;

TEST(StaticShapeInferenceTest, GatherTreeTest) {
    auto step_ids = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1});
    auto parent_idx = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1});
    auto max_seq_len = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1});
    auto end_token = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{Shape{}});
    auto gather_tree = std::make_shared<op::v1::GatherTree>(step_ids, parent_idx, max_seq_len, end_token);
    std::vector<PartialShape> input_shapes = {PartialShape{1, 2, 3},
                                              PartialShape{1, 2, 3},
                                              PartialShape{2},
                                              PartialShape{Shape{}}},
                              output_shapes = {PartialShape{}};
    shape_infer(gather_tree.get(), input_shapes, output_shapes);
}