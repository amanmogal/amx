// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <ngraph/pass/constant_folding.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/common_optimizations/add_fake_quantize_fusion.hpp>
#include <transformations/common_optimizations/align_eltwise_input_ranks.hpp>
#include <transformations/common_optimizations/batch_to_space_fusion.hpp>
#include <transformations/common_optimizations/binarize_weights.hpp>
#include <transformations/common_optimizations/broadcast_elementwise_fusion.hpp>
#include <transformations/common_optimizations/clamp_fusion.hpp>
#include <transformations/common_optimizations/conv_mul_fusion.hpp>
#include <transformations/common_optimizations/conv_to_binary_conv.hpp>
#include <transformations/common_optimizations/convert_nms_gather_path_to_unsigned.hpp>
#include <transformations/common_optimizations/convert_quantize_dequantize.hpp>
#include <transformations/common_optimizations/depth_to_space_fusion.hpp>
#include <transformations/common_optimizations/dilated_convolution_converter.hpp>
#include <transformations/common_optimizations/disable_random_uniform_constant_folding.hpp>
#include <transformations/common_optimizations/disable_shapeof_constant_folding.hpp>
#include <transformations/common_optimizations/divide_fusion.hpp>
#include <transformations/common_optimizations/eliminate_unsqueeze_gather.hpp>
#include <transformations/common_optimizations/fold_subgraph_empty_inputs.hpp>
#include <transformations/common_optimizations/fq_mul_fusion.hpp>
#include <transformations/common_optimizations/fq_reshape_fusion.hpp>
#include <transformations/common_optimizations/gelu_fusion.hpp>
#include <transformations/common_optimizations/gru_cell_fusion.hpp>
#include <transformations/common_optimizations/hsigmoid_fusion.hpp>
#include <transformations/common_optimizations/hswish_fusion.hpp>
#include <transformations/common_optimizations/leaky_relu_fusion.hpp>
#include <transformations/common_optimizations/lin_op_sequence_fusion.hpp>
#include <transformations/common_optimizations/matmul_const_transposes_extraction.hpp>
#include <transformations/common_optimizations/matmul_multiply_fusion.hpp>
#include <transformations/common_optimizations/moc_transformations.hpp>
#include <transformations/common_optimizations/mul_conv_fusion.hpp>
#include <transformations/common_optimizations/mul_fake_quantize_fusion.hpp>
#include <transformations/common_optimizations/mvn_fusion.hpp>
#include <transformations/common_optimizations/nearest_neighbor_upsampling_fusion.hpp>
#include <transformations/common_optimizations/nop_elimination.hpp>
#include <transformations/common_optimizations/normalize_l2_fusion.hpp>
#include <transformations/common_optimizations/optimize_strided_slice.hpp>
#include <transformations/common_optimizations/pad_fusion.hpp>
#include <transformations/common_optimizations/prelu_fusion.hpp>
#include <transformations/common_optimizations/pull_transpose_through_fq.hpp>
#include <transformations/common_optimizations/random_uniform_fusion.hpp>
#include <transformations/common_optimizations/relu_fake_quantize_fusion.hpp>
#include <transformations/common_optimizations/remove_concat_zero_dim_input.hpp>
#include <transformations/common_optimizations/remove_filtering_boxes_by_size.hpp>
#include <transformations/common_optimizations/remove_multi_subgraph_op_dangling_params.hpp>
#include <transformations/common_optimizations/reshape_sequence_fusion.hpp>
#include <transformations/common_optimizations/ric_fusion.hpp>
#include <transformations/common_optimizations/shuffle_channels_fusion.hpp>
#include <transformations/common_optimizations/simplify_shape_of_sub_graph.hpp>
#include <transformations/common_optimizations/softmax_fusion.hpp>
#include <transformations/common_optimizations/softplus_fusion.hpp>
#include <transformations/common_optimizations/softplus_to_mish_fusion.hpp>
#include <transformations/common_optimizations/space_to_batch_fusion.hpp>
#include <transformations/common_optimizations/split_concat_pair_to_interpolate_fusion.hpp>
#include <transformations/common_optimizations/split_squeeze_concat_fusion.hpp>
#include <transformations/common_optimizations/subtract_fusion.hpp>
#include <transformations/common_optimizations/swish_fusion.hpp>
#include <transformations/common_optimizations/transpose_sinking.hpp>
#include <transformations/common_optimizations/transpose_to_reshape.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/low_precision/disable_convert_constant_folding_on_const_path.hpp>
#include <transformations/op_conversions/batch_norm_decomposition.hpp>
#include <transformations/op_conversions/convert_divide.hpp>
#include <transformations/op_conversions/convert_negative.hpp>
#include <transformations/op_conversions/convert_scatter_elements_to_scatter.hpp>
#include <transformations/smart_reshape/lstm_states_broadcast.hpp>
#include <transformations/smart_reshape/reshape_sinking.hpp>

#include "itt.hpp"

bool ngraph::pass::MOCTransformations::run_on_model(const std::shared_ptr<ngraph::Function>& f) {
    RUN_ON_FUNCTION_SCOPE(MOCTransformations);
    // To avoid issues with dynamism we make nGraph Function dynamic and after we apply all
    // transformations we restore original shapes to the nGraph Function back
    std::unordered_map<ngraph::op::Parameter*, PartialShape> input_shapes;
    if (!m_use_shapes) {
        for (auto&& param : f->get_parameters()) {
            input_shapes[param.get()] = param->get_partial_shape();
            param->set_partial_shape(PartialShape::dynamic(param->get_partial_shape().rank()));
        }
        f->validate_nodes_and_infer_types();
    }

    ngraph::pass::Manager manager(get_pass_config());
    manager.set_per_pass_validation(false);
    REGISTER_PASS_MODEL_SCOPE(manager, ngraph::pass, InitNodeInfo)
    if (m_low_precision_enabled) {
        manager.register_pass<ngraph::pass::DisableConvertConstantFoldingOnConstPath>(
            element::TypeVector{ngraph::element::i8, ngraph::element::u8, ngraph::element::i4, ngraph::element::u4});
    }
    if (!m_use_shapes) {
        manager.register_pass<ngraph::pass::DisableShapeOfConstantFolding>();
    }
    // RemoveConcatZeroDimInput and RemoveMultiSubGraphOpDanglingParams
    // should be performed before first ConstantFolding call.
    // The passes can deteach graph branches where zero dimesion is calculated.
    // Zero dimensions in shape causes creation empty tensors, which are incorrect during CF.
    // In particular, if zero dim tensor is consumed in body of MultiSubGraphOp
    // RemoveConcatZeroDimInput and RemoveMultiSubGraphOpDanglingParams should be called together.
    REGISTER_PASS_SCOPE(manager, ov::pass, RemoveConcatZeroDimInput)
    REGISTER_PASS_SCOPE(manager, ov::pass, Validate)
    REGISTER_PASS_SCOPE(manager, ov::pass, RemoveMultiSubGraphOpDanglingParams)
    REGISTER_PASS_SCOPE(manager, ov::pass, FoldSubgraphEmptyInputs)

    manager.register_pass<ngraph::pass::DisableRandomUniformConstantFolding>();
    REGISTER_PASS_SCOPE(manager, ngraph::pass, ConstantFolding)
    REGISTER_PASS_SCOPE(manager, ngraph::pass, Validate)

    // FusedFilteringBoxesBySize transformation has the complex pattern
    // which can be affected by further transformations. So we have to
    // execute it at the beginning of the pipeline. Also, this pass resolves
    // dynamism, so we have to execute type/shape propagation after.
    manager.register_pass<ngraph::pass::FuseFilteringBoxesBySize>();
    REGISTER_PASS_MODEL_SCOPE(manager, ngraph::pass, Validate)

    if (!m_use_shapes) {  // Approved Smart Reshape
        REGISTER_PASS_FUNCTION_SCOPE(manager, ov::pass, LSTMStatesBroadcast)
        REGISTER_PASS_MODEL_SCOPE(manager, ov::pass, Validate)
        REGISTER_PASS_SCOPE(manager, ov::pass, ReshapeSinkingMatMul)
        REGISTER_PASS_MODEL_SCOPE(manager, ov::pass, Validate)
    }
    REGISTER_PASS_SCOPE(manager, ngraph::pass, ConvertQuantizeDequantize)
    REGISTER_PASS_FUNCTION_SCOPE(manager, ngraph::pass, SimplifyShapeOfSubGraph)

    if (!m_use_shapes) {
        manager.register_pass<ngraph::pass::DisableShapeOfConstantFolding>();
    }
    // workaround until dynamism in NMS is not supported
    manager.register_pass<ngraph::pass::ConvertNmsGatherPathToUnsigned>();
    REGISTER_PASS_FUNCTION_SCOPE(manager, ngraph::pass, StridedSliceOptimization, m_use_shapes)
    REGISTER_PASS_SCOPE(manager, ngraph::pass, BroadcastElementwiseFusion)
    REGISTER_PASS_MODEL_SCOPE_IF(GraphRewrite) {
        auto transpose_sinking = manager.register_pass<ngraph::pass::GraphRewrite>();
        ADD_MATCHER_SCOPE_WITH_OBJ(transpose_sinking, ngraph::pass, TransposeSinking)

        // SplitSqueezeConcatFusion should work in same GraphRewrite as TransposesSinking,
        // because it replaces pattern that may contain Transposes which must be optimized before
        // the transformation and it also inserts Transpose that can be optimized by TransposeSinking
        ADD_MATCHER_SCOPE_WITH_OBJ(transpose_sinking, ngraph::pass, SplitSqueezeConcatFusion)
    }
    REGISTER_PASS_MODEL_SCOPE_IF(GraphRewrite) {
        auto eliminations = manager.register_pass<ngraph::pass::GraphRewrite>();
        ADD_MATCHER_SCOPE_WITH_OBJ(eliminations, ngraph::pass, EliminateUnsqueezeGather)
        ADD_MATCHER_SCOPE_WITH_OBJ(eliminations, ngraph::pass, NopElimination, m_use_shapes)
        eliminations->set_name("ngraph::pass::CommonEliminations");
    }
    manager.register_pass<ngraph::pass::ConstantFolding>();

    REGISTER_PASS_MODEL_SCOPE_IF(GraphRewrite) {
        auto common_fusions = manager.register_pass<ngraph::pass::GraphRewrite>();
        ADD_MATCHER_SCOPE_WITH_OBJ(common_fusions, ngraph::pass, ConvertScatterElementsToScatter)
        ADD_MATCHER_SCOPE_WITH_OBJ(common_fusions, ngraph::pass, SoftPlusFusion)
        ADD_MATCHER_SCOPE_WITH_OBJ(common_fusions, ngraph::pass, SoftPlusToMishFusion)
        ADD_MATCHER_SCOPE_WITH_OBJ(common_fusions, ngraph::pass, SwishFusion)
        ADD_MATCHER_SCOPE_WITH_OBJ(common_fusions, ngraph::pass, HSwishFusion)
        ADD_MATCHER_SCOPE_WITH_OBJ(common_fusions, ngraph::pass, HSigmoidFusion)
        ADD_MATCHER_SCOPE_WITH_OBJ(common_fusions, ngraph::pass, NormalizeL2Fusion)
        ADD_MATCHER_SCOPE_WITH_OBJ(common_fusions, ngraph::pass, ClampFusion)
        ADD_MATCHER_SCOPE_WITH_OBJ(common_fusions, ngraph::pass, PadFusion)
        ADD_MATCHER_SCOPE_WITH_OBJ(common_fusions, ngraph::pass, SoftmaxFusion)
        ADD_MATCHER_SCOPE_WITH_OBJ(common_fusions, ngraph::pass, MVNFusion)
        ADD_MATCHER_SCOPE_WITH_OBJ(common_fusions, ngraph::pass, DilatedConvolutionConverter)
        ADD_MATCHER_SCOPE_WITH_OBJ(common_fusions, ngraph::pass, GeluFusion)
        ADD_MATCHER_SCOPE_WITH_OBJ(common_fusions, ngraph::pass, LeakyReluFusion)
        ADD_MATCHER_SCOPE_WITH_OBJ(common_fusions, ngraph::pass, RandomUniformFusion)
        ADD_MATCHER_SCOPE_WITH_OBJ(common_fusions, ngraph::pass, SplitConcatPairToInterpolateFusion)
        ADD_MATCHER_SCOPE_WITH_OBJ(common_fusions, ngraph::pass, SplitConcatPairToInterpolateFusion, m_use_shapes)
        if (m_use_shapes) {
            ADD_MATCHER_SCOPE_WITH_OBJ(common_fusions, ngraph::pass, NearestNeighborUpsamplingFusion)
        }
        ADD_MATCHER_SCOPE_WITH_OBJ(common_fusions, ngraph::pass, DivideFusion)
        ADD_MATCHER_SCOPE_WITH_OBJ(common_fusions, ngraph::pass, SubtractFusion);

        ADD_MATCHER_SCOPE_WITH_OBJ(common_fusions, ngraph::pass, TransposeToReshape)
        ADD_MATCHER_SCOPE_WITH_OBJ(common_fusions, ngraph::pass, ReshapeSequenceFusion, m_use_shapes)
        ADD_MATCHER_SCOPE_WITH_OBJ(common_fusions, ngraph::pass, MatMulConstTransposesExtraction)
        ADD_MATCHER_SCOPE_WITH_OBJ(common_fusions, ngraph::pass, PReluFusion)
        ADD_MATCHER_SCOPE_WITH_OBJ(common_fusions, ngraph::pass, DepthToSpaceFusion)
        ADD_MATCHER_SCOPE_WITH_OBJ(common_fusions, ngraph::pass, ShuffleChannelsFusion, !m_use_shapes)
        common_fusions->set_name("ngraph::pass::CommonFusions");
    }

    REGISTER_PASS_SCOPE(manager, ngraph::pass, BinarizeWeights)
    REGISTER_PASS_SCOPE(manager, ngraph::pass, ConvToBinaryConv)

    REGISTER_PASS_MODEL_SCOPE_IF(GraphRewrite) {
        auto decomp = manager.register_pass<ngraph::pass::GraphRewrite>();
        ADD_MATCHER_SCOPE_WITH_OBJ(decomp, ngraph::pass, BatchNormDecomposition)
        ADD_MATCHER_SCOPE_WITH_OBJ(decomp, ngraph::pass, ConvertDivideWithConstant)
        ADD_MATCHER_SCOPE_WITH_OBJ(decomp, ngraph::pass, ConvertNegative)
    }
    manager.register_pass<ngraph::pass::LinOpSequenceFusion>();
    REGISTER_PASS_MODEL_SCOPE_IF(GraphRewrite) {
        auto multiply_fusions = manager.register_pass<ngraph::pass::GraphRewrite>();
        ADD_MATCHER_SCOPE_WITH_OBJ(multiply_fusions, ngraph::pass, ConvolutionMultiplyFusion)
        ADD_MATCHER_SCOPE_WITH_OBJ(multiply_fusions, ngraph::pass, GroupConvolutionMultiplyFusion)
        ADD_MATCHER_SCOPE_WITH_OBJ(multiply_fusions, ngraph::pass, ConvolutionBackpropDataMultiplyFusion)
        ADD_MATCHER_SCOPE_WITH_OBJ(multiply_fusions, ngraph::pass, GroupConvolutionBackpropDataMultiplyFusion)
        ADD_MATCHER_SCOPE_WITH_OBJ(multiply_fusions, ngraph::pass, MultiplyConvolutionFusion)
        ADD_MATCHER_SCOPE_WITH_OBJ(multiply_fusions, ngraph::pass, MultiplyGroupConvolutionFusion)
        ADD_MATCHER_SCOPE_WITH_OBJ(multiply_fusions, ngraph::pass, MultiplyConvolutionBackpropDataFusion)
        ADD_MATCHER_SCOPE_WITH_OBJ(multiply_fusions, ngraph::pass, MultiplyGroupConvolutionBackpropDataFusion)
        ADD_MATCHER_SCOPE_WITH_OBJ(multiply_fusions, ngraph::pass, MatMulMultiplyFusion)
        multiply_fusions->set_name("ngraph::pass::MultiplyFusions");
    }

    REGISTER_PASS_SCOPE(manager, ngraph::pass, ConstantFolding)

    REGISTER_PASS_MODEL_SCOPE_IF(GraphRewrite) {
        auto fq_fusions = manager.register_pass<ngraph::pass::GraphRewrite>();
        ADD_MATCHER_SCOPE_WITH_OBJ(fq_fusions, ngraph::pass, FakeQuantizeMulFusion)
        ADD_MATCHER_SCOPE_WITH_OBJ(fq_fusions, ngraph::pass, FakeQuantizeReshapeFusion)
        ADD_MATCHER_SCOPE_WITH_OBJ(fq_fusions, ngraph::pass, PullTransposeThroughFQUp)
        ADD_MATCHER_SCOPE_WITH_OBJ(fq_fusions, ngraph::pass, ReluFakeQuantizeFusion)
        ADD_MATCHER_SCOPE_WITH_OBJ(fq_fusions, ngraph::pass, AddFakeQuantizeFusion)
        ADD_MATCHER_SCOPE_WITH_OBJ(fq_fusions, ngraph::pass, MulFakeQuantizeFusion)
        fq_fusions->set_name("ngraph::pass::FakeQuantizeFusions");
    }
    manager.register_pass<ngraph::pass::ReverseInputChannelsFusion>();

    manager.register_pass<ngraph::pass::AlignEltwiseInputRanks>();
    REGISTER_PASS_MODEL_SCOPE(manager, ngraph::pass, ConstantFolding)

    manager.run_passes(f);

    if (!m_use_shapes) {
        // Restore original shapes to the nGraph Function
        for (auto&& param : f->get_parameters()) {
            param->set_partial_shape(input_shapes.at(param.get()));
        }
        f->validate_nodes_and_infer_types();
    }

    return false;
}
