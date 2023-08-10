// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <openvino/op/parameter.hpp>
#include <openvino/op/variadic_split.hpp>
#include <ov_ops/rotary_positional_embeddings.hpp>
#include <transformations/common_optimizations/fuse_rotary_positional_embeddings.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace std;
using namespace ov;
using namespace ov::op;

void name_parameter(const shared_ptr<v0::Parameter>& parameter, std::string name) {
    parameter->set_friendly_name(name);
    parameter->output(0).set_names({name});
}

TEST_F(TransformationTestsF, FuseRPE) {
    {
        auto data = make_shared<v0::Parameter>(element::f32, PartialShape::dynamic());
        name_parameter(data, "source");
        auto sin = make_shared<v0::Parameter>(element::f32, PartialShape::dynamic());
        name_parameter(sin, "sin");
        auto cos = make_shared<v0::Parameter>(element::f32, PartialShape::dynamic());
        name_parameter(cos, "cos");
        auto axis = v0::Constant::create(element::i64, {}, {-1});
        auto split_lengths = v0::Constant::create(element::i64, {2}, {10, 10});
        auto split = make_shared<v1::VariadicSplit>(data, axis, split_lengths);

        auto minus_one = v0::Constant::create(element::f32, {}, {-1});
        auto negate = make_shared<v1::Multiply>(split->output(1), minus_one);

        auto concat = make_shared<v0::Concat>(OutputVector{negate, split->output(0)}, -1);

        auto mul_sin = make_shared<op::v1::Multiply>(concat, sin);
        auto mul_cos = make_shared<op::v1::Multiply>(data, cos);
        auto add = make_shared<op::v1::Add>(mul_cos, mul_sin);

        model = std::make_shared<Model>(NodeVector{add}, ParameterVector{data, sin, cos});

        manager.register_pass<ov::pass::RPE_Fusion>();
    }
    {
        auto data = make_shared<v0::Parameter>(element::f32, PartialShape::dynamic());
        auto sin = make_shared<v0::Parameter>(element::f32, PartialShape::dynamic());
        auto cos = make_shared<v0::Parameter>(element::f32, PartialShape::dynamic());
        auto rpe = make_shared<ov::op::internal::RPE>(data, sin, cos, -1);
        model_ref = std::make_shared<Model>(NodeVector{rpe}, ParameterVector{data, sin, cos});
    }
}
// FIXME: this test should pass. Right now I get the following error during matching process:
// Process finished with exit code 139 (interrupted by signal 11: SIGSEGV)
//
//TEST_F(TransformationTestsF, FuseRPESorcesAreMultiOutputed) {
//    {
//        auto data_ = make_shared<v0::Parameter>(element::f32, PartialShape::dynamic());
//        auto sin_ = make_shared<v0::Parameter>(element::f32, PartialShape::dynamic());
//        auto cos_ = make_shared<v0::Parameter>(element::f32, PartialShape::dynamic());
//
//        auto data = make_shared<v1::Split>(data_, v0::Constant::create(element::i64, {}, {-1}), 2);
//        auto sin = make_shared<v1::Split>(sin_, v0::Constant::create(element::i64, {}, {-1}), 2)->output(1);
//        auto cos = make_shared<v1::Split>(cos_, v0::Constant::create(element::i64, {}, {-1}), 2)->output(1);
//
//        auto axis = v0::Constant::create(element::i64, {}, {-1});
//        auto split_lengths = v0::Constant::create(element::i64, {2}, {10, 10});
//        auto split = make_shared<v1::VariadicSplit>(data->output(0), axis, split_lengths);
//
//        auto minus_one = v0::Constant::create(element::f32, {}, {-1});
//        auto negate = make_shared<v1::Multiply>(split->output(1), minus_one);
//
//        auto concat = make_shared<v0::Concat>(OutputVector{negate, split->output(0)}, -1);
//
//        auto mul_sin = make_shared<op::v1::Multiply>(concat, sin);
//        auto mul_cos = make_shared<op::v1::Multiply>(data->output(1), cos);
//        auto add = make_shared<op::v1::Add>(mul_cos, mul_sin);
//
//        model = std::make_shared<Model>(NodeVector{add}, ParameterVector{data_, sin_, cos_});
//
//        manager.register_pass<ov::pass::RPE_Fusion>();
//    }
//}