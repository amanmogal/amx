// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/frontend/exception.hpp>
#include <openvino/frontend/extension.hpp>
#include <openvino/frontend/manager.hpp>
#include <openvino/opsets/opset10.hpp>
#include <transformations/common_optimizations/moc_transformations.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "gtest/gtest.h"
#include "test_common.hpp"
#include "tf_utils.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov;
using namespace ov::element;
using namespace ov::opset10;
using namespace ov::frontend;

namespace {
shared_ptr<Model> convert_model(const string& model_path, const ConversionExtension::Ptr& conv_ext = nullptr) {
    FrontEndManager fem;
    auto front_end = fem.load_by_framework(TF_FE);
    if (!front_end) {
        throw "TensorFlow Frontend is not initialized";
    }
    if (conv_ext) {
        front_end->add_extension(conv_ext);
    }
    auto model_filename = FrontEndTestUtils::make_model_path(string(TEST_TENSORFLOW_MODELS_DIRNAME) + model_path);
    auto input_model = front_end->load(model_filename);
    if (!input_model) {
        throw "Input model is not read";
    }
    auto model = front_end->convert(input_model);
    if (!model) {
        throw "Model is not converted";
    }

    return model;
}

ov::OutputVector fake_translator_ragged_tensor_to_sparse(const ov::frontend::NodeContext& node) {
    // NOTE: pay attention that this is a fake translator for RaggedTensorToSparse
    // only serves for testing purposes
    FRONT_END_GENERAL_CHECK(node.get_input_size() > 1, "RaggedTensorToSparse expects at least two inputs.");
    auto node_name = node.get_name();
    auto row_splits = node.get_input(0);
    auto strings = node.get_input(1);

    // Override type of input tensor if this is a Parameter
    if (auto parameter = as_type_ptr<ov::opset10::Parameter>(strings.get_node_shared_ptr())) {
        parameter->set_partial_shape(ov::PartialShape{Dimension()});
        parameter->set_element_type(ov::element::u8);
        parameter->validate_and_infer_types();
    }

    row_splits = make_shared<ConvertLike>(row_splits, strings);
    auto const_one = make_shared<Constant>(row_splits.get_element_type(), Shape{}, 1);
    Output<Node> mul = make_shared<Multiply>(row_splits, const_one);
    auto const_two = make_shared<Constant>(ov::element::u8, Shape{}, 2);
    Output<Node> add = make_shared<Add>(strings, const_two);
    auto const_three = make_shared<Constant>(ov::element::u8, Shape{}, 3);
    Output<Node> sub = make_shared<Subtract>(strings, const_three);

    mul.get_tensor().add_names({node_name + ":0"});
    add.get_tensor().add_names({node_name + ":1"});
    sub.get_tensor().add_names({node_name + ":2"});

    return {mul, add, sub};
}
}  // namespace

TEST(FrontEndConvertTrickyModels, undefined_input_shape) {
    shared_ptr<Model> model;
    try {
        model = convert_model("undefined_input_shape/undefined_input_shape.pb");
    } catch (std::exception& ex) {
        ASSERT_TRUE(false) << ex.what();
    }

    for (auto& node : model->get_ordered_ops()) {
        if (node->get_friendly_name() == "x") {
            ASSERT_TRUE(node->get_output_partial_shape(0).same_scheme(ov::PartialShape::dynamic()));
        } else if (node->get_friendly_name() == "y") {
            ASSERT_TRUE(node->get_output_partial_shape(0).same_scheme(ov::PartialShape{2, 3}));
        } else if (node->get_friendly_name() == "z") {
            ASSERT_TRUE(node->get_output_partial_shape(0).same_scheme(ov::PartialShape::dynamic()));
        }
    }
}

TEST(FrontEndConvertTrickyModels, simple_wide_and_deep) {
    shared_ptr<Model> model;
    try {
        model = convert_model("simple_wide_and_deep/simple_wide_and_deep.pb");
    } catch (std::exception& ex) {
        ASSERT_TRUE(false) << ex.what();
    }

    int num_emb_segment_sum = 0;
    for (auto& node : model->get_ordered_ops()) {
        if (std::dynamic_pointer_cast<EmbeddingSegmentsSum>(node)) {
            ++num_emb_segment_sum;
        }
    }

    ASSERT_EQ(num_emb_segment_sum, 1) << "The number of EmbeddingSegmentsSum nodes must be 1";
}

TEST(FrontEndConvertTrickyModels, model_with_output_shapes) {
    shared_ptr<Model> model;
    try {
        model = convert_model("model_with_output_shapes_attr/model_with_output_shapes_attr.pb");
    } catch (std::exception& ex) {
        ASSERT_TRUE(false) << ex.what();
    }

    for (auto& node : model->get_ordered_ops()) {
        if (node->get_friendly_name() == "x") {
            ASSERT_TRUE(node->get_output_partial_shape(0).same_scheme(ov::PartialShape{2, 3}));
        } else if (node->get_friendly_name() == "relu") {
            ASSERT_TRUE(node->get_output_partial_shape(0).same_scheme(ov::PartialShape{2, 3}));
        }
    }
}

TEST_F(TransformationTestsF, AssertAndStringTensors) {
    {
        model = convert_model("string_tensors_model/string_tensors_model.pb");
        // TODO: investigate - why we have redundant nodes after the conversion
        manager.register_pass<pass::MOCTransformations>(false);
    }
    {
        auto x = make_shared<Parameter>(f32, Shape{2, 3});
        auto y = make_shared<Parameter>(f32, Shape{2, 3});
        auto cond = make_shared<Constant>(boolean, Shape{1, 1}, std::vector<bool>{true});
        auto select = make_shared<Select>(cond, x, y);

        model_ref = make_shared<Model>(OutputVector{select}, ParameterVector{x, y});
    }
}

TEST_F(TransformationTestsF, UnsortedNodes) {
    { model = convert_model("forward_edge_model_unsorted/forward_edge_model_unsorted.pb"); }
    { model_ref = convert_model("forward_edge_model/forward_edge_model.pb"); }
}

TEST_F(TransformationTestsF, ModelWithSwishF32BodyGraph) {
    {
        model = convert_model("swish_f32/swish_f32.pb");
        // need to call shape inference since body graphs can be injected with undefined shapes
        model->validate_nodes_and_infer_types();
    }
    {
        auto x = make_shared<Parameter>(f32, Shape{1, 112, 112, 32});
        auto const_add = make_shared<Constant>(f32, Shape{}, std::vector<float>{2});
        auto add = make_shared<Add>(x, const_add);
        auto sigmoid = make_shared<Sigmoid>(add);
        auto mul = make_shared<Multiply>(add, sigmoid);
        auto sigmoid2 = make_shared<Sigmoid>(mul);

        model_ref = make_shared<Model>(OutputVector{sigmoid2}, ParameterVector{x});
    }
}

TEST_F(TransformationTestsF, PartitionedCall) {
    {
        model = convert_model("partitioned_call/partitioned_call.pb");
        // need to call shape inference since body graphs can be injected with undefined shapes
        model->validate_nodes_and_infer_types();
    }
    {
        auto x = make_shared<Parameter>(i32, Shape{2});
        auto y = make_shared<Parameter>(i32, Shape{1});
        auto sub = make_shared<Subtract>(x, y);
        auto const_pow = make_shared<Constant>(i32, Shape{}, 2);
        auto pow = make_shared<Power>(sub, const_pow);

        model_ref = make_shared<Model>(OutputVector{pow}, ParameterVector{x, y});
    }
}

TEST_F(TransformationTestsF, ModelWithIf) {
    { model = convert_model("model_with_if/model_with_if.pb"); }
    {
        // create then branch body graph
        auto then_x = make_shared<Parameter>(i32, Shape{2});
        auto then_y = make_shared<Parameter>(i32, Shape{1});
        auto add = make_shared<Add>(then_x, then_y);
        auto then_result = make_shared<Result>(add);
        auto then_model = make_shared<Model>(OutputVector{then_result}, ParameterVector{then_x, then_y});

        // create else branch body graph
        auto else_x = make_shared<Parameter>(i32, Shape{2});
        auto else_y = make_shared<Parameter>(i32, Shape{1});
        auto sub = make_shared<Subtract>(else_x, else_y);
        auto else_result = make_shared<Result>(sub);
        auto else_model = make_shared<Model>(OutputVector{else_result}, ParameterVector{else_x, else_y});

        // create the main graph
        auto x = make_shared<Parameter>(i32, Shape{2});
        auto y = make_shared<Parameter>(i32, Shape{1});
        auto cond_const = make_shared<Constant>(i32, Shape{}, 10);
        auto cond = make_shared<Greater>(x, cond_const);
        auto if_op = make_shared<If>(cond);
        if_op->set_then_body(then_model);
        if_op->set_else_body(else_model);
        if_op->set_input(x, then_x, else_x);
        if_op->set_input(y, then_y, else_y);
        if_op->set_output(then_result, else_result);

        model_ref = make_shared<Model>(OutputVector{if_op}, ParameterVector{x, y});
    }
}

TEST_F(TransformationTestsF, InjectedBodyAndIf) {
    {
        model = convert_model("injected_body_and_if/injected_body_and_if.pb");
        // need to call shape inference since body graphs can be injected with undefined shapes
        model->validate_nodes_and_infer_types();
    }
    {
        // create then branch body graph
        auto then_x = make_shared<Parameter>(i32, Shape{2});
        auto then_y = make_shared<Parameter>(i32, Shape{1});
        auto add = make_shared<Add>(then_x, then_y);
        auto then_result = make_shared<Result>(add);
        auto then_model = make_shared<Model>(OutputVector{then_result}, ParameterVector{then_x, then_y});

        // create else branch body graph
        auto else_x = make_shared<Parameter>(i32, Shape{2});
        auto else_y = make_shared<Parameter>(i32, Shape{1});
        auto sub = make_shared<Subtract>(else_x, else_y);
        auto pow_const = make_shared<Constant>(i32, Shape{}, 2);
        auto pow = make_shared<Power>(sub, pow_const);
        auto else_result = make_shared<Result>(pow);
        auto else_model = make_shared<Model>(OutputVector{else_result}, ParameterVector{else_x, else_y});

        // create the main graph
        auto x = make_shared<Parameter>(i32, Shape{2});
        auto y = make_shared<Parameter>(i32, Shape{1});
        auto cond_const = make_shared<Constant>(i32, Shape{}, 10);
        auto cond = make_shared<Greater>(x, cond_const);
        auto if_op = make_shared<If>(cond);
        if_op->set_then_body(then_model);
        if_op->set_else_body(else_model);
        if_op->set_input(x, then_x, else_x);
        if_op->set_input(y, then_y, else_y);
        if_op->set_output(then_result, else_result);

        model_ref = make_shared<Model>(OutputVector{if_op}, ParameterVector{x, y});
    }
}

TEST_F(TransformationTestsF, ModelWithDilatedGroupConvolution) {
    {
        model = convert_model("dilated_gconv_model/dilated_gconv_model.pb");
        // need to call MOC to fuse BatchToSpace/SpaceToBatch with GroupConvolution
        manager.register_pass<pass::MOCTransformations>(false);
    }
    {
        auto x = make_shared<Parameter>(f32, Shape{1, 129, 257, 384});
        auto transpose_before_const = make_shared<Constant>(i64, Shape{4}, std::vector<int64_t>{0, 3, 1, 2});
        auto transpose_before = make_shared<Transpose>(x, transpose_before_const);
        auto const_filter = make_shared<Constant>(f32, Shape{384, 1, 1, 3, 3}, std::vector<float>(384 * 3 * 3, 0));
        Strides dilations{2, 2};
        CoordinateDiff pads_begin{2, 2};
        CoordinateDiff pads_end{2, 2};
        Strides strides{1, 1};
        auto gconv =
            make_shared<GroupConvolution>(transpose_before, const_filter, strides, pads_begin, pads_end, dilations);
        auto transpose_after_const = make_shared<Constant>(i64, Shape{4}, std::vector<int64_t>{0, 2, 3, 1});
        auto transpose_after = make_shared<Transpose>(gconv, transpose_after_const);

        model_ref = make_shared<Model>(OutputVector{transpose_after}, ParameterVector{x});
    }
}

TEST_F(TransformationTestsF, ModelWithSaveV2) {
    {
        model = convert_model("model_savev2/model_savev2.pb");
        // need to call shape inference since body graphs can be injected with undefined shapes
        model->validate_nodes_and_infer_types();
    }
    {
        // create a reference graph
        auto x = make_shared<Parameter>(element::f32, Shape{2});
        auto const_2 = make_shared<Constant>(element::f32, Shape{2}, vector<float>{1, 2});
        auto add = make_shared<Add>(x, const_2);

        model_ref = make_shared<Model>(OutputVector{add}, ParameterVector{x});
    }
}

TEST_F(TransformationTestsF, ModelWithConstResultSubgraphs) {
    { model = convert_model("model_with_const_result/model_with_const_result.pb"); }
    {
        // create a reference graph
        auto x = make_shared<Parameter>(element::f32, PartialShape{Dimension::dynamic(), 60, 60, 1});
        auto perm_order = make_shared<Constant>(element::i64, Shape{4}, vector<int64_t>{0, 3, 1, 2});
        auto transpose_to_nchw = make_shared<Transpose>(x, perm_order);
        auto max_pool = make_shared<MaxPool>(transpose_to_nchw,
                                             Strides{2, 2},
                                             Strides{1, 1},
                                             Shape{0, 0},
                                             Shape{0, 0},
                                             Shape{2, 2},
                                             ov::op::RoundingType::FLOOR,
                                             ov::op::PadType::VALID,
                                             element::i64);
        auto inverse_order = make_shared<Constant>(element::i64, Shape{4}, vector<int64_t>{0, 2, 3, 1});
        auto transpose_to_nhwc = make_shared<Transpose>(max_pool, inverse_order);

        model_ref = make_shared<Model>(OutputVector{transpose_to_nhwc}, ParameterVector{x});
    }
}

TEST_F(TransformationTestsF, ModelWithIteratorGetNext) {
    { model = convert_model("model_with_iterator_get_next/model_with_iterator_get_next.pb"); }
    {
        // create a reference graph
        auto x = make_shared<Parameter>(element::f32, Shape{2, 3});
        auto y = make_shared<Parameter>(element::f32, Shape{2, 3});
        auto sub = make_shared<Subtract>(x, y);

        model_ref = make_shared<Model>(OutputVector{sub}, ParameterVector{x, y});
    }
}

TEST_F(TransformationTestsF, ModelWithQueueOperations) {
    { model = convert_model("model_with_queue_ops/model_with_queue_ops.pb"); }
    {
        // create a reference graph
        auto x = make_shared<Parameter>(element::f32, PartialShape{Dimension::dynamic(), 160, 160, 3});
        auto y = make_shared<Parameter>(element::f32, PartialShape{Dimension::dynamic(), 160, 160, 3});
        auto sub = make_shared<Subtract>(x, y);

        model_ref = make_shared<Model>(OutputVector{sub}, ParameterVector{x, y});
    }
}

TEST_F(TransformationTestsF, ModelWithQueueOperations2) {
    { model = convert_model("model_with_queue_ops2/model_with_queue_ops2.pb"); }
    {
        // create a reference graph
        auto x = make_shared<Parameter>(element::f32, PartialShape{1, Dimension::dynamic(), Dimension::dynamic(), 3});
        auto y = make_shared<Constant>(element::f32,
                                       Shape{1, 1, 1, 3},
                                       vector<float>{123.68000030517578, 116.77899932861328, 103.93900299072266});
        auto sub = make_shared<Subtract>(x, y);

        model_ref = make_shared<Model>(OutputVector{sub}, ParameterVector{x});
    }
}

TEST_F(TransformationTestsF, ModelWithLookupTableOperations) {
    { model = convert_model("model_with_lookup_table/model_with_lookup_table.pb"); }
    {
        // create a reference graph
        auto x = make_shared<Parameter>(element::f32, Shape{2});
        auto const_2 = make_shared<Constant>(element::f32, Shape{2}, vector<float>{1, 2});
        auto add = make_shared<Add>(x, const_2);

        model_ref = make_shared<Model>(OutputVector{add}, ParameterVector{x});
    }
}

TEST_F(TransformationTestsF, ModelWithIteratorGetNextAndUnsupportedOp) {
    { model = convert_model("unsupported_op_itergetnext/unsupported_op_itergetnext.pb"); }
    {
        // create then branch body graph
        auto x = make_shared<Parameter>(f32, Shape{2, 3});
        auto y = make_shared<Parameter>(f32, Shape{3});
        auto add = make_shared<Add>(x, y);

        model_ref = make_shared<Model>(OutputVector{add}, ParameterVector{x, y});
    }
}

TEST_F(TransformationTestsF, ModelWithMultioutputBodyGraphNode) {
    { model = convert_model("partitioned_call2/partitioned_call2.pb"); }
    {
        auto x = make_shared<Parameter>(i32, Shape{5});
        auto y = make_shared<Parameter>(i32, Shape{5});
        auto sub = make_shared<Subtract>(x, y);
        auto const_three = make_shared<Constant>(i32, Shape{}, 3);
        auto const_ten = make_shared<Constant>(i32, Shape{}, 10);
        auto topk =
            make_shared<TopK>(sub, const_three, -1, op::v1::TopK::Mode::MAX, op::v1::TopK::SortType::SORT_VALUES, i32);
        auto add = make_shared<Add>(topk->output(1), const_ten);
        model_ref = make_shared<Model>(OutputVector{add}, ParameterVector{x, y});
    }
}

TEST_F(TransformationTestsF, ModelWithEmptyTensorListAndPushBack) {
    { model = convert_model("empty_tensor_list/empty_tensor_list.pb"); }
    {
        auto x = make_shared<Parameter>(f32, Shape{2, 3, 5});
        auto minus_one_const = make_shared<Constant>(i32, Shape{1}, -1);
        auto x_flatten = make_shared<Reshape>(x, minus_one_const, false);
        auto zero_const = make_shared<Constant>(i32, Shape{1}, 0);
        auto x_unsqueeze_flatten = make_shared<Unsqueeze>(x_flatten, zero_const);
        auto empty_const = make_shared<Constant>(f32, Shape{0, 30}, vector<float>{});
        auto list_push_back = make_shared<Concat>(OutputVector{empty_const, x_unsqueeze_flatten}, 0);
        auto recover_item_shape = make_shared<Constant>(i32, Shape{4}, vector<int32_t>{1, 2, 3, 5});
        auto recover_item = make_shared<Reshape>(list_push_back, recover_item_shape, false);
        model_ref = make_shared<Model>(OutputVector{recover_item}, ParameterVector{x});
    }
}

TEST_F(TransformationTestsF, ModelWithAssertNode) {
    { model = convert_model("model_with_assert/model_with_assert.pb"); }
    {
        auto x = make_shared<Parameter>(i32, PartialShape{Dimension::dynamic()});
        auto y = make_shared<Parameter>(i32, PartialShape{Dimension::dynamic()});
        auto add = make_shared<Add>(x, y);
        model_ref = make_shared<Model>(OutputVector{add}, ParameterVector{x, y});
    }
}

TEST_F(TransformationTestsF, PartitionedCallWithUnique) {
    // This test aims to test named output ports for Unique operation
    { model = convert_model("partitioned_call_with_unique/partitioned_call_with_unique.pb"); }
    {
        auto x = make_shared<Parameter>(f32, Shape{5});
        auto relu = make_shared<Relu>(x);
        auto unique = make_shared<Unique>(relu, false, i32);
        auto const_one = make_shared<Constant>(i32, Shape{}, 1);
        auto add = make_shared<Add>(unique->output(2), const_one);
        auto sigmoid = make_shared<Sigmoid>(unique->output(0));
        model_ref = make_shared<Model>(OutputVector{sigmoid, add}, ParameterVector{x});
    }
}

TEST_F(TransformationTestsF, RaggedTensorToSparse) {
    // This test aims to test named output ports for RaggedTensorToSparse operation
    // also, it tests propagation of custom type (specified in the extension) to Parameter node in the parent graph
    {
        // create FAKE conversion extension for RaggedTensorToSparse
        auto conv_ext = std::make_shared<ov::frontend::ConversionExtension>("RaggedTensorToSparse",
                                                                            fake_translator_ragged_tensor_to_sparse);
        model = convert_model("ragged_tensor_to_sparse/ragged_tensor_to_sparse.pb", conv_ext);
    }
    {
        auto strings = make_shared<Parameter>(u8, PartialShape{3});
        auto row_splits = make_shared<Parameter>(i32, PartialShape{5});
        auto convert_like = make_shared<ConvertLike>(row_splits, strings);

        auto const_one = make_shared<Constant>(u8, Shape{}, 1);
        Output<Node> mul = make_shared<Multiply>(convert_like, const_one);
        auto const_three = make_shared<Constant>(u8, Shape{}, 3);
        Output<Node> sub = make_shared<Subtract>(strings, const_three);

        auto target_shape1 = make_shared<Constant>(i32, Shape{1}, -1);
        auto reshape1 = make_shared<Reshape>(mul, target_shape1, false);
        auto target_shape2 = make_shared<Constant>(i32, Shape{1}, -1);
        auto reshape2 = make_shared<Reshape>(sub, target_shape2, false);

        auto concat = make_shared<Concat>(OutputVector{reshape1, reshape2}, 0);

        model_ref = make_shared<Model>(OutputVector{concat}, ParameterVector{row_splits, strings});
    }
}
