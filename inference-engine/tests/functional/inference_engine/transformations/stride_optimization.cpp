// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ngraph/opsets/opset7.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/common_optimizations/stride_optimization.hpp>
#include <transformations/init_node_info.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

// Tests are based on model-optimizer/mo/middle/passes/fusing/resnet_optimization_test.py
// In description of unit tests below will be used next syntax: Operation(NxM,XxY), where NxM - kernel size, XxY - stride

// Pl->Conv(1x1,1x1)->Conv(1x1,2x2) => Pl->Conv(1x1,2x2)->Conv(1x1,1x1)
TEST(TransformationTests, StrideOptimization1) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 224, 224});
        auto weights_1 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 1, 1}, {128});
        auto conv_1 = std::make_shared<ngraph::opset7::Convolution>(data, weights_1, ngraph::Strides{},
                ngraph::CoordinateDiff{}, ngraph::CoordinateDiff{}, ngraph::Strides{});
        auto weights_2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 1, 1}, {128});
        auto conv_2 = std::make_shared<ngraph::opset7::Convolution>(conv_1, weights_2, ngraph::Strides{2, 2},
                ngraph::CoordinateDiff{}, ngraph::CoordinateDiff{}, ngraph::Strides{});

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{conv_2}, ngraph::ParameterVector{data});
        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<ngraph::pass::StrideOptimization>();
        m.run_passes(f);
    }
    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 224, 224});
        auto weights_1 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 1, 1}, {128});
        auto conv_1 = std::make_shared<ngraph::opset7::Convolution>(data, weights_1, ngraph::Strides{2, 2},
                ngraph::CoordinateDiff{}, ngraph::CoordinateDiff{}, ngraph::Strides{});
        auto weights_2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 1, 1}, {128});
        auto conv_2 = std::make_shared<ngraph::opset7::Convolution>(conv_1, weights_2, ngraph::Strides{},
                ngraph::CoordinateDiff{}, ngraph::CoordinateDiff{}, ngraph::Strides{});

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{conv_2}, ngraph::ParameterVector{data});
    }

    auto res = compare_functions(f, f_ref, true);
    ASSERT_TRUE(res.first) << res.second;
}

// Pl->Conv(3x3,2x2)->Conv(1x1,2x2) => Pl->Conv(3x3,4x4)->Conv(1x1,1x1)
TEST(TransformationTests, StrideOptimization2) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 224, 224});
        auto weights_1 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 3, 3}, {128});
        auto conv_1 = std::make_shared<ngraph::opset7::Convolution>(data, weights_1, ngraph::Strides{2, 2},
                ngraph::CoordinateDiff{}, ngraph::CoordinateDiff{}, ngraph::Strides{});
        auto weights_2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 1, 1}, {128});
        auto conv_2 = std::make_shared<ngraph::opset7::Convolution>(conv_1, weights_2, ngraph::Strides{2, 2},
                ngraph::CoordinateDiff{}, ngraph::CoordinateDiff{}, ngraph::Strides{});

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{conv_2}, ngraph::ParameterVector{data});
        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<ngraph::pass::StrideOptimization>();
        m.run_passes(f);
    }
    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 224, 224});
        auto weights_1 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 3, 3}, {128});
        auto conv_1 = std::make_shared<ngraph::opset7::Convolution>(data, weights_1, ngraph::Strides{4, 4},
                ngraph::CoordinateDiff{}, ngraph::CoordinateDiff{}, ngraph::Strides{});
        auto weights_2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 1, 1}, {128});
        auto conv_2 = std::make_shared<ngraph::opset7::Convolution>(conv_1, weights_2, ngraph::Strides{},
                ngraph::CoordinateDiff{}, ngraph::CoordinateDiff{}, ngraph::Strides{});

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{conv_2}, ngraph::ParameterVector{data});
    }

    auto res = compare_functions(f, f_ref, true);
    ASSERT_TRUE(res.first) << res.second;
}

// Pl->Conv(3x3,2x2)->Conv(3x3,2x2) => Same
TEST(TransformationTests, StrideOptimization3) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 224, 224});
        auto weights_1 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 3, 3}, {128});
        auto conv_1 = std::make_shared<ngraph::opset7::Convolution>(data, weights_1, ngraph::Strides{2, 2},
                ngraph::CoordinateDiff{}, ngraph::CoordinateDiff{}, ngraph::Strides{});
        auto weights_2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 3, 3}, {128});
        auto conv_2 = std::make_shared<ngraph::opset7::Convolution>(conv_1, weights_2, ngraph::Strides{2, 2},
                ngraph::CoordinateDiff{}, ngraph::CoordinateDiff{}, ngraph::Strides{});

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{conv_2}, ngraph::ParameterVector{data});
        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<ngraph::pass::StrideOptimization>();
        m.run_passes(f);
    }
    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 224, 224});
        auto weights_1 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 3, 3}, {128});
        auto conv_1 = std::make_shared<ngraph::opset7::Convolution>(data, weights_1, ngraph::Strides{2, 2},
                ngraph::CoordinateDiff{}, ngraph::CoordinateDiff{}, ngraph::Strides{});
        auto weights_2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 3, 3}, {128});
        auto conv_2 = std::make_shared<ngraph::opset7::Convolution>(conv_1, weights_2, ngraph::Strides{2, 2},
                ngraph::CoordinateDiff{}, ngraph::CoordinateDiff{}, ngraph::Strides{});

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{conv_2}, ngraph::ParameterVector{data});
    }

    auto res = compare_functions(f, f_ref, true);
    ASSERT_TRUE(res.first) << res.second;
}

// Pl--->Conv(3x3,2x2)->ReLU--->Eltwise-->Conv(1x1,2x2) => Pl--->Conv(3x3,4x4)->ReLU--->Eltwise-->Conv(1x1,1x1)
//   `-->Conv(3x3,2x2)->ReLU---`                             `-->Conv(3x3,4x4)->ReLU---`
TEST(TransformationTests, StrideOptimization4) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 224, 224});
        auto weights_1 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 3, 3}, {128});
        auto conv_1 = std::make_shared<ngraph::opset7::Convolution>(data, weights_1, ngraph::Strides{2, 2},
                ngraph::CoordinateDiff{}, ngraph::CoordinateDiff{}, ngraph::Strides{});
        auto relu_1 = std::make_shared<ngraph::opset7::Relu>(conv_1);
        auto weights_2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 3, 3}, {128});
        auto conv_2 = std::make_shared<ngraph::opset7::Convolution>(data, weights_2, ngraph::Strides{2, 2},
                ngraph::CoordinateDiff{}, ngraph::CoordinateDiff{}, ngraph::Strides{});
        auto relu_2 = std::make_shared<ngraph::opset7::Relu>(conv_2);
        auto add = std::make_shared<ngraph::opset7::Add>(relu_1, relu_2);
        auto weights_3 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 1, 1}, {128});
        auto conv_3 = std::make_shared<ngraph::opset7::Convolution>(add, weights_3, ngraph::Strides{2, 2},
                ngraph::CoordinateDiff{}, ngraph::CoordinateDiff{}, ngraph::Strides{});

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{conv_3}, ngraph::ParameterVector{data});
        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<ngraph::pass::StrideOptimization>();
        m.run_passes(f);
    }
    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 224, 224});
        auto weights_1 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 3, 3}, {128});
        auto conv_1 = std::make_shared<ngraph::opset7::Convolution>(data, weights_1, ngraph::Strides{4, 4},
                ngraph::CoordinateDiff{}, ngraph::CoordinateDiff{}, ngraph::Strides{});
        auto relu_1 = std::make_shared<ngraph::opset7::Relu>(conv_1);
        auto weights_2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 3, 3}, {128});
        auto conv_2 = std::make_shared<ngraph::opset7::Convolution>(data, weights_2, ngraph::Strides{4, 4},
                ngraph::CoordinateDiff{}, ngraph::CoordinateDiff{}, ngraph::Strides{});
        auto relu_2 = std::make_shared<ngraph::opset7::Relu>(conv_2);
        auto add = std::make_shared<ngraph::opset7::Add>(relu_1, relu_2);
        auto weights_3 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 1, 1}, {128});
        auto conv_3 = std::make_shared<ngraph::opset7::Convolution>(add, weights_3, ngraph::Strides{1, 1},
                ngraph::CoordinateDiff{}, ngraph::CoordinateDiff{}, ngraph::Strides{});

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{conv_3}, ngraph::ParameterVector{data});
    }

    auto res = compare_functions(f, f_ref, true);
    ASSERT_TRUE(res.first) << res.second;
}

// Pl--->Conv(1x1,1x1)->ReLU--->Eltwise-->Conv(1x1,2x2) => Pl--->Conv(1x1,2x2)->ReLU--->Eltwise-->Conv(1x1,1x1)
//   `----------------->ReLU---`                             `-->Pool(1x1,2x2)->ReLU---`
TEST(TransformationTests, StrideOptimization5) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 224, 224});
        auto weights_1 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 1, 1}, {128});
        auto conv_1 = std::make_shared<ngraph::opset7::Convolution>(data, weights_1, ngraph::Strides{1, 1},
                ngraph::CoordinateDiff{0, 0}, ngraph::CoordinateDiff{0, 0}, ngraph::Strides{});
        auto relu_1 = std::make_shared<ngraph::opset7::Relu>(conv_1);
        auto relu_2 = std::make_shared<ngraph::opset7::Relu>(data);
        auto add = std::make_shared<ngraph::opset7::Add>(relu_1, relu_2);
        auto weights_2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 1, 1}, {128});
        auto conv_2 = std::make_shared<ngraph::opset7::Convolution>(add, weights_2, ngraph::Strides{2, 2},
                ngraph::CoordinateDiff{}, ngraph::CoordinateDiff{}, ngraph::Strides{});

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{conv_2}, ngraph::ParameterVector{data});
        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<ngraph::pass::StrideOptimization>();
        m.run_passes(f);
    }
    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 224, 224});
        auto weights_1 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 1, 1}, {128});
        auto conv_1 = std::make_shared<ngraph::opset7::Convolution>(data, weights_1, ngraph::Strides{2, 2},
                ngraph::CoordinateDiff{}, ngraph::CoordinateDiff{}, ngraph::Strides{});
        auto relu_1 = std::make_shared<ngraph::opset7::Relu>(conv_1);
        auto pool = std::make_shared<ngraph::opset7::MaxPool>(data, ngraph::Strides{2, 2}, ngraph::Shape{0, 0}, ngraph::Shape{0, 0}, ngraph::Shape{1, 1});
        auto relu_2 = std::make_shared<ngraph::opset7::Relu>(pool);
        auto add = std::make_shared<ngraph::opset7::Add>(relu_1, relu_2);
        auto weights_2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 1, 1}, {128});
        auto conv_2 = std::make_shared<ngraph::opset7::Convolution>(add, weights_2, ngraph::Strides{1, 1},
                ngraph::CoordinateDiff{}, ngraph::CoordinateDiff{}, ngraph::Strides{});

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{conv_2}, ngraph::ParameterVector{data});
    }

    auto res = compare_functions(f, f_ref, true);
    ASSERT_TRUE(res.first) << res.second;
}

// Pl->Conv(1x1,1x1)->Conv(1x1,2x2)->Conv(3x3,1x1)->Conv(1x1,2x2)
//       =>
// Pl->Conv(1x1,2x2)->Conv(1x1,1x1)->Conv(3x3,2x2)->Conv(1x1,1x1)
TEST(TransformationTests, StrideOptimization6) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 224, 224});
        auto weights_1 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 1, 1}, {128});
        auto conv_1 = std::make_shared<ngraph::opset7::Convolution>(data, weights_1, ngraph::Strides{1, 1},
                ngraph::CoordinateDiff{}, ngraph::CoordinateDiff{}, ngraph::Strides{});
        auto weights_2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 1, 1}, {128});
        auto conv_2 = std::make_shared<ngraph::opset7::Convolution>(conv_1, weights_2, ngraph::Strides{2, 2},
                ngraph::CoordinateDiff{}, ngraph::CoordinateDiff{}, ngraph::Strides{});
        auto weights_3 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 3, 3}, {128});
        auto conv_3 = std::make_shared<ngraph::opset7::Convolution>(conv_2, weights_3, ngraph::Strides{1, 1},
                ngraph::CoordinateDiff{}, ngraph::CoordinateDiff{}, ngraph::Strides{});
        auto weights_4 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 1, 1}, {128});
        auto conv_4 = std::make_shared<ngraph::opset7::Convolution>(conv_3, weights_4, ngraph::Strides{2, 2},
                ngraph::CoordinateDiff{}, ngraph::CoordinateDiff{}, ngraph::Strides{});

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{conv_4}, ngraph::ParameterVector{data});
        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<ngraph::pass::StrideOptimization>();
        m.run_passes(f);
    }
    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 224, 224});
        auto weights_1 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 1, 1}, {128});
        auto conv_1 = std::make_shared<ngraph::opset7::Convolution>(data, weights_1, ngraph::Strides{2, 2},
                ngraph::CoordinateDiff{}, ngraph::CoordinateDiff{}, ngraph::Strides{});
        auto weights_2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 1, 1}, {128});
        auto conv_2 = std::make_shared<ngraph::opset7::Convolution>(conv_1, weights_2, ngraph::Strides{1, 1},
                ngraph::CoordinateDiff{}, ngraph::CoordinateDiff{}, ngraph::Strides{});
        auto weights_3 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 3, 3}, {128});
        auto conv_3 = std::make_shared<ngraph::opset7::Convolution>(conv_2, weights_3, ngraph::Strides{2, 2},
                ngraph::CoordinateDiff{}, ngraph::CoordinateDiff{}, ngraph::Strides{});
        auto weights_4 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 1, 1}, {128});
        auto conv_4 = std::make_shared<ngraph::opset7::Convolution>(conv_3, weights_4, ngraph::Strides{1, 1},
                ngraph::CoordinateDiff{}, ngraph::CoordinateDiff{}, ngraph::Strides{});

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{conv_4}, ngraph::ParameterVector{data});
    }

    auto res = compare_functions(f, f_ref, true);
    ASSERT_TRUE(res.first) << res.second;
}
