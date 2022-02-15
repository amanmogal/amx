// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <single_layer_tests/op_impl_check/op_impl_check.hpp>
#include <single_layer_tests/op_impl_check/single_op_graph.hpp>

namespace ov {
namespace test {
namespace subgraph {

namespace {
std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::Op> &node) {
    return nullptr;
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v5::GRUSequence> &node) {
    const auto params = ngraph::builder::makeDynamicParams(ov::element::f32, {{2, 5, 3}, {2, 1, 3}});
    const auto params_seqLength = ngraph::builder::makeDynamicParams(ov::element::i64, {{2}});
    const auto W = ngraph::builder::makeConstant<float>(ov::element::f32, {1, 9, 3}, {}, true);
    const auto R = ngraph::builder::makeConstant<float>(ov::element::f32, {1, 9, 3}, {}, true);
    const auto B = ngraph::builder::makeConstant<float>(ov::element::f32, {1, 9}, {}, true);
    auto RNNCellBaseNode = std::make_shared<ov::op::v5::GRUSequence>(params[0], params[1], params_seqLength[0],
                                                                     W, R, B, 3, ov::op::RecurrentSequenceDirection::FORWARD);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(RNNCellBaseNode->output(0)),
                                 std::make_shared<ngraph::opset1::Result>(RNNCellBaseNode->output(1))};
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{params[0], params[1], params_seqLength[0]}, "RNNCellBaseGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v0::LSTMSequence> &node) {
    const auto params = ngraph::builder::makeDynamicParams(ov::element::f32, {{5, 10, 10}, {5, 1, 10}, {5, 1, 10}});
    const auto params_seqLength = ngraph::builder::makeDynamicParams(ov::element::i64, {{5}});
    const auto W = ngraph::builder::makeConstant<float>(ov::element::f32, {1, 40, 10}, {}, true);
    const auto R = ngraph::builder::makeConstant<float>(ov::element::f32, {1, 40, 10}, {}, true);
    const auto B = ngraph::builder::makeConstant<float>(ov::element::f32, {1, 40}, {}, true);
    const auto P = ngraph::builder::makeConstant<float>(ov::element::f32, {1, 30}, {}, true);
    auto RNNCellBaseNode = std::make_shared<ov::op::v0::LSTMSequence>(params[0], params[1], params[2], params_seqLength[0],
                                                                      W, R, B, P, 10, ov::op::RecurrentSequenceDirection::FORWARD);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(RNNCellBaseNode->output(0)),
                                 std::make_shared<ngraph::opset1::Result>(RNNCellBaseNode->output(1)),
                                 std::make_shared<ngraph::opset1::Result>(RNNCellBaseNode->output(2))};
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{params[0], params[1], params[2], params_seqLength[0]}, "RNNCellBaseGraph");
}

std::shared_ptr<ov::Model> generateBinaryEltwise(const std::shared_ptr<ov::op::Op> &node) {
    const auto params = ngraph::builder::makeDynamicParams(ov::element::f32, {{1, 2},
                                                                              {1, 2}});
    std::shared_ptr<ov::Node> eltwiseNode;
    if (ov::is_type<ov::op::v0::SquaredDifference>(node)) {
        eltwiseNode = std::make_shared<ov::op::v0::SquaredDifference>(params.front(), params.back());
    } else if (ov::is_type<ov::op::v1::Add>(node)) {
        eltwiseNode = std::make_shared<ov::op::v1::Add>(params.front(), params.back());
    } else if (ov::is_type<ov::op::v1::Divide>(node)) {
        eltwiseNode = std::make_shared<ov::op::v1::Divide>(params.front(), params.back());
    } else if (ov::is_type<ov::op::v1::FloorMod>(node)) {
        eltwiseNode = std::make_shared<ov::op::v1::FloorMod>(params.front(), params.back());
    } else if (ov::is_type<ov::op::v1::Maximum>(node)) {
        eltwiseNode = std::make_shared<ov::op::v1::Maximum>(params.front(), params.back());
    } else if (ov::is_type<ov::op::v1::Minimum>(node)) {
        eltwiseNode = std::make_shared<ov::op::v1::Minimum>(params.front(), params.back());
    } else if (ov::is_type<ov::op::v1::Multiply>(node)) {
        eltwiseNode = std::make_shared<ov::op::v1::Multiply>(params.front(), params.back());
    } else if (ov::is_type<ov::op::v1::Power>(node)) {
        eltwiseNode = std::make_shared<ov::op::v1::Power>(params.front(), params.back());
    } else if (ov::is_type<ov::op::v1::Subtract>(node)) {
        eltwiseNode = std::make_shared<ov::op::v1::Subtract>(params.front(), params.back());
    } else {
        return nullptr;
    }

    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(eltwiseNode)};
    return std::make_shared<ngraph::Function>(results, params, "BinaryEltwiseGraph");
}

std::shared_ptr<ov::Model> generateDeformableConvolutionBase(const std::shared_ptr<ov::op::Op> &node) {
    const auto params = ngraph::builder::makeDynamicParams(ov::element::f32, {{1, 2, 4, 4},
                                                                              {1, 18, 2, 2},
                                                                              {1, 2, 3, 3}});
    std::shared_ptr<ov::Node> deformableConvolutionNode;
    if (ov::is_type<ov::op::v1::DeformableConvolution>(node)) {
        deformableConvolutionNode = std::make_shared<ov::op::v1::DeformableConvolution>(params[0], params[1], params[2],
                                                                                        ov::Strides {1, 1},
                                                                                        ov::CoordinateDiff {0, 0},
                                                                                        ov::CoordinateDiff {0, 0},
                                                                                        ov::Strides {1, 1});
    } else if (ov::is_type<ov::op::v8::DeformableConvolution>(node)) {
        deformableConvolutionNode = std::make_shared<ov::op::v8::DeformableConvolution>(params[0], params[1], params[2],
                                                                                        ov::Strides {1, 1},
                                                                                        ov::CoordinateDiff {0, 0},
                                                                                        ov::CoordinateDiff {0, 0},
                                                                                        ov::Strides {1, 1});
    } else {
        return nullptr;
    }

    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(deformableConvolutionNode)};
    return std::make_shared<ngraph::Function>(results, params, "DeformableConvolutionBaseGraph");
}

std::shared_ptr<ov::Model> generateDetectionOutputBase(const std::shared_ptr<ov::op::Op> &node) {
    const auto params = ngraph::builder::makeDynamicParams(ov::element::f32, {{2, 8},
                                                                              {2, 6},
                                                                              {2, 1, 8}});
    ov::op::v0::DetectionOutput::Attributes attrs;
    ov::op::v8::DetectionOutput::Attributes attrs_v8;
    attrs.num_classes = 3;
    attrs_v8.background_label_id = attrs.background_label_id = -1;
    attrs_v8.top_k = attrs.top_k = -1;
    attrs_v8.variance_encoded_in_target = attrs.variance_encoded_in_target = true;
    attrs_v8.keep_top_k = attrs.keep_top_k = {2};
    attrs_v8.code_type = attrs.code_type = "caffe.PriorBoxParameter.CORNER";
    attrs_v8.share_location = attrs.share_location = true;
    attrs_v8.nms_threshold = attrs.nms_threshold = 0.5;
    attrs_v8.confidence_threshold = attrs.confidence_threshold = 0.3;
    attrs_v8.clip_after_nms = attrs.clip_after_nms = false;
    attrs_v8.clip_before_nms = attrs.clip_before_nms = true;
    attrs_v8.decrease_label_id = attrs.decrease_label_id = false;
    attrs_v8.normalized = attrs.normalized = true;
    attrs_v8.input_height = attrs.input_height = 0;
    attrs_v8.input_width = attrs.input_width = 0;
    attrs_v8.objectness_score = attrs.objectness_score = 0;

    std::shared_ptr<ov::Node> DetectionOutputNode;
    if (ov::is_type<ov::op::v0::DetectionOutput>(node)) {
        DetectionOutputNode = std::make_shared<ov::op::v0::DetectionOutput>(params[0], params[1], params[2], attrs);
    } else if (ov::is_type<ov::op::v8::DetectionOutput>(node)) {
        DetectionOutputNode = std::make_shared<ov::op::v8::DetectionOutput>(params[0], params[1], params[2], attrs_v8);
    } else {
        return nullptr;
    }

    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(DetectionOutputNode)};
    return std::make_shared<ngraph::Function>(results, params, "DetectionOutputBaseGraph");
}

std::shared_ptr<ov::Model> generateEmbeddingBagOffsetsBase(const std::shared_ptr<ov::op::Op> &node) {
    const auto params = ngraph::builder::makeDynamicParams(ov::element::f32, {{5, 2}});
    const auto indices = ngraph::builder::makeConstant<int32_t>(ov::element::i32, {4}, {}, true);
    const auto offsets = ngraph::builder::makeConstant<int32_t>(ov::element::i32, {3}, {}, true);
    const auto default_index = ngraph::builder::makeConstant<int32_t>(ov::element::i32, ov::Shape(), std::vector<int32_t>{0});

    std::shared_ptr<ov::Node> EmbeddingBagOffsetsSumNode;
    if (ov::is_type<ov::op::v3::EmbeddingBagOffsetsSum>(node)) {
        EmbeddingBagOffsetsSumNode = std::make_shared<ov::op::v3::EmbeddingBagOffsetsSum>(params[0], indices, offsets, default_index);
    } else {
        return nullptr;
    }

    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(EmbeddingBagOffsetsSumNode)};
    return std::make_shared<ngraph::Function>(results, params, "EmbeddingBagOffsetsBaseGraph");
}

std::shared_ptr<ov::Model> generateEmbeddingBagPackedBase(const std::shared_ptr<ov::op::Op> &node) {
    const auto params = ngraph::builder::makeDynamicParams(ov::element::f32, {{5, 2}});
    const auto indices = ngraph::builder::makeConstant<int32_t>(ov::element::i32, {2, 3}, {}, true);

    std::shared_ptr<ov::Node> EmbeddingBagPackedSumNode;
    if (ov::is_type<ov::op::v3::EmbeddingBagPackedSum>(node)) {
        EmbeddingBagPackedSumNode = std::make_shared<ov::op::v3::EmbeddingBagPackedSum>(params[0], indices);
    } else {
        return nullptr;
    }

    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(EmbeddingBagPackedSumNode)};
    return std::make_shared<ngraph::Function>(results, params, "EmbeddingBagPackedBaseGraph");
}

std::shared_ptr<ov::Model> generateFFTBase(const std::shared_ptr<ov::op::Op> &node) {
    const auto params = ngraph::builder::makeDynamicParams(ov::element::f32, {{2, 10, 10, 2}});
    const auto axes = ngraph::builder::makeConstant<int32_t>(ov::element::i32, {1}, {2});

    std::shared_ptr<ov::Node> FFTBaseNode;
    if (ov::is_type<ov::op::v7::DFT>(node)) {
        FFTBaseNode = std::make_shared<ov::op::v7::DFT>(params[0], axes);
    } else if (ov::is_type<ov::op::v7::IDFT>(node)) {
        FFTBaseNode = std::make_shared<ov::op::v7::IDFT>(params[0], axes);
    } else {
        return nullptr;
    }

    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(FFTBaseNode)};
    return std::make_shared<ngraph::Function>(results, params, "FFTBaseGraph");
}

std::shared_ptr<ov::Model> generateGatherBase(const std::shared_ptr<ov::op::Op> &node) {
    const auto params = ngraph::builder::makeDynamicParams(ov::element::i32, {{2, 2, 3, 3}, {2}});
    const auto axis = ngraph::builder::makeConstant<int64_t>(ov::element::i64, ov::Shape(), std::vector<int64_t>{2});

    std::shared_ptr<ov::Node> GatherBaseNode;
    if (ov::is_type<ov::op::v1::Gather>(node)) {
        GatherBaseNode = std::make_shared<ov::op::v1::Gather>(params[0], params[1], axis);
    } else if (ov::is_type<ov::op::v7::Gather>(node)) {
        GatherBaseNode = std::make_shared<ov::op::v7::Gather>(params[0], params[1], axis);
    } else if (ov::is_type<ov::op::v8::Gather>(node)) {
        GatherBaseNode = std::make_shared<ov::op::v8::Gather>(params[0], params[1], axis);
    } else {
        return nullptr;
    }

    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(GatherBaseNode)};
    return std::make_shared<ngraph::Function>(results, params, "GatherBaseGraph");
}

std::shared_ptr<ov::Model> generateGatherNDBase(const std::shared_ptr<ov::op::Op> &node) {
    const auto params = ngraph::builder::makeDynamicParams(ov::element::i32, {{2, 3, 4, 2}, {2, 3, 3, 2}});

    std::shared_ptr<ov::Node> GatherNDBaseNode;
    if (ov::is_type<ov::op::v5::GatherND>(node)) {
        GatherNDBaseNode = std::make_shared<ov::op::v5::GatherND>(params[0], params[1]);
    } else if (ov::is_type<ov::op::v8::GatherND>(node)) {
        GatherNDBaseNode = std::make_shared<ov::op::v8::GatherND>(params[0], params[1]);
    } else {
        return nullptr;
    }

    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(GatherNDBaseNode)};
    return std::make_shared<ngraph::Function>(results, params, "GatherNDBaseGraph");
}

std::shared_ptr<ov::Model> generateRNNCellBase(const std::shared_ptr<ov::op::Op> &node) {
    std::shared_ptr<ov::Node> RNNCellBaseNode;
    if (ov::is_type<ov::op::v3::GRUCell>(node)) {
        const auto params = ngraph::builder::makeDynamicParams(ov::element::f32, {{2, 3}, {2, 3}});
        const auto W = ngraph::builder::makeConstant<float>(ov::element::f32, {9, 3}, {}, true);
        const auto R = ngraph::builder::makeConstant<float>(ov::element::f32, {9, 3}, {}, true);
        const auto B = ngraph::builder::makeConstant<float>(ov::element::f32, {9}, {}, true);
        RNNCellBaseNode = std::make_shared<ov::op::v3::GRUCell>(params[0], params[1],
                                                                W, R, B, 3);
        ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(RNNCellBaseNode)};
        return std::make_shared<ngraph::Function>(results, params, "RNNCellBaseGraph");
    } else if (ov::is_type<ov::op::v0::LSTMCell>(node)) {
        const auto params = ngraph::builder::makeDynamicParams(ov::element::f32, {{2, 3}, {2, 3}, {2, 3}});
        const auto W = ngraph::builder::makeConstant<float>(ov::element::f32, {12, 3}, {}, true);
        const auto R = ngraph::builder::makeConstant<float>(ov::element::f32, {12, 3}, {}, true);
        const auto B = ngraph::builder::makeConstant<float>(ov::element::f32, {12}, {}, true);
        const auto P = ngraph::builder::makeConstant<float>(ov::element::f32, {9}, {}, true);
        RNNCellBaseNode = std::make_shared<ov::op::v0::LSTMCell>(params[0], params[1], params[2],
                                                                 W, R, B, P, 3);
        ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(RNNCellBaseNode->output(0)),
                                     std::make_shared<ngraph::opset1::Result>(RNNCellBaseNode->output(1))};
        return std::make_shared<ngraph::Function>(results, params, "RNNCellBaseGraph");
    } else if (ov::is_type<ov::op::v4::LSTMCell>(node)) {
        const auto params = ngraph::builder::makeDynamicParams(ov::element::f32, {{2, 3}, {2, 3}, {2, 3}});
        const auto W = ngraph::builder::makeConstant<float>(ov::element::f32, {12, 3}, {}, true);
        const auto R = ngraph::builder::makeConstant<float>(ov::element::f32, {12, 3}, {}, true);
        const auto B = ngraph::builder::makeConstant<float>(ov::element::f32, {12}, {}, true);
        RNNCellBaseNode = std::make_shared<ov::op::v4::LSTMCell>(params[0], params[1], params[2],
                                                                 W, R, B, 3);
        ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(RNNCellBaseNode->output(0)),
                                     std::make_shared<ngraph::opset1::Result>(RNNCellBaseNode->output(1))};;
        return std::make_shared<ngraph::Function>(results, params, "RNNCellBaseGraph");
    } else if (ov::is_type<ov::op::v5::LSTMSequence>(node)) {
        const auto params = ngraph::builder::makeDynamicParams(ov::element::f32, {{5, 10, 10}, {5, 1, 10}, {5, 1, 10}});
        const auto params_seqLength = ngraph::builder::makeDynamicParams(ov::element::i64, {{5}});
        const auto W = ngraph::builder::makeConstant<float>(ov::element::f32, {1, 40, 10}, {}, true);
        const auto R = ngraph::builder::makeConstant<float>(ov::element::f32, {1, 40, 10}, {}, true);
        const auto B = ngraph::builder::makeConstant<float>(ov::element::f32, {1, 40}, {}, true);
        RNNCellBaseNode = std::make_shared<ov::op::v5::LSTMSequence>(params[0], params[1], params[2], params_seqLength[0],
                                                                     W, R, B, 10, ov::op::RecurrentSequenceDirection::FORWARD);
        ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(RNNCellBaseNode->output(0)),
                                     std::make_shared<ngraph::opset1::Result>(RNNCellBaseNode->output(1)),
                                     std::make_shared<ngraph::opset1::Result>(RNNCellBaseNode->output(2))};
        return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{params[0], params[1], params[2], params_seqLength[0]}, "RNNCellBaseGraph");
    } else if (ov::is_type<ov::op::v0::RNNCell>(node)) {
        const auto params = ngraph::builder::makeDynamicParams(ov::element::f32, {{2, 3}, {2, 3}});
        const auto W = ngraph::builder::makeConstant<float>(ov::element::f32, {3, 3}, {}, true);
        const auto R = ngraph::builder::makeConstant<float>(ov::element::f32, {3, 3}, {}, true);
        const auto B = ngraph::builder::makeConstant<float>(ov::element::f32, {3}, {}, true);
        RNNCellBaseNode = std::make_shared<ov::op::v0::RNNCell>(params[0], params[1],
                                                                W, R, B, 3);
        ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(RNNCellBaseNode)};
        return std::make_shared<ngraph::Function>(results, params, "RNNCellBaseGraph");
    } else if (ov::is_type<ov::op::v5::RNNSequence>(node)) {
        const auto params = ngraph::builder::makeDynamicParams(ov::element::f32, {{2, 5, 3}, {2, 1, 3}});
        const auto params_seqLength = ngraph::builder::makeDynamicParams(ov::element::i64, {{2}});
        const auto W = ngraph::builder::makeConstant<float>(ov::element::f32, {1, 3, 3}, {}, true);
        const auto R = ngraph::builder::makeConstant<float>(ov::element::f32, {1, 3, 3}, {}, true);
        const auto B = ngraph::builder::makeConstant<float>(ov::element::f32, {1, 3}, {}, true);
        RNNCellBaseNode = std::make_shared<ov::op::v5::RNNSequence>(params[0], params[1], params_seqLength[0],
                                                                    W, R, B, 3, ov::op::RecurrentSequenceDirection::FORWARD);
        ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(RNNCellBaseNode->output(0)),
                                     std::make_shared<ngraph::opset1::Result>(RNNCellBaseNode->output(1))};
        return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{params[0], params[1], params_seqLength[0]}, "RNNCellBaseGraph");
    } else {
        return nullptr;
    }
}
} // namespace

template <typename T>
std::shared_ptr<ov::Model> generateGraph() {
        std::shared_ptr<T> node = std::shared_ptr<T>(new T);
    if (ov::is_type<ov::op::util::BinaryElementwiseArithmetic>(node)) {
        return generateBinaryEltwise(node);
    } else if (ov::is_type<ov::op::util::DeformableConvolutionBase>(node)) {
        return generateDeformableConvolutionBase(node);
    } else if (ov::is_type<ov::op::util::DetectionOutputBase>(node)) {
        return generateDetectionOutputBase(node);
    } else if (ov::is_type<ov::op::util::EmbeddingBagOffsetsBase>(node)) {
        return generateEmbeddingBagOffsetsBase(node);
    } else if (ov::is_type<ov::op::util::EmbeddingBagPackedBase>(node)) {
        return generateEmbeddingBagPackedBase(node);
    } else if (ov::is_type<ov::op::util::FFTBase>(node)) {
        return generateFFTBase(node);
    } else if (ov::is_type<ov::op::util::GatherBase>(node)) {
        return generateGatherBase(node);
    } else if (ov::is_type<ov::op::util::GatherNDBase>(node)) {
        return generateGatherNDBase(node);
    } else if (ov::is_type<ov::op::util::RNNCellBase>(node)) {
        return generateRNNCellBase(node);
    }
    return generate(node);
}

OpGenerator getOpGeneratorMap() {
    static OpGenerator opGeneratorMap{
#define _OPENVINO_OP_REG(NAME, NAMESPACE) {NAMESPACE::NAME::get_type_info_static(), generateGraph<NAMESPACE::NAME>},
#include "openvino/opsets/opset1_tbl.hpp"
#include "openvino/opsets/opset2_tbl.hpp"
#include "openvino/opsets/opset3_tbl.hpp"
#include "openvino/opsets/opset4_tbl.hpp"
#include "openvino/opsets/opset5_tbl.hpp"
#include "openvino/opsets/opset6_tbl.hpp"
#include "openvino/opsets/opset7_tbl.hpp"
#include "openvino/opsets/opset8_tbl.hpp"
#undef _OPENVINO_OP_REG
    };
    return opGeneratorMap;
}

}  // namespace subgraph
}  // namespace test
}  // namespace ov
