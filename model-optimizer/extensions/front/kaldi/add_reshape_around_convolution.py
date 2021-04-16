# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from extensions.ops.Cast import Cast
from extensions.ops.elementwise import Div
from extensions.ops.transpose import Transpose
from mo.front.common.partial_infer.utils import int64_array, float32_array
from mo.front.common.replacement import FrontReplacementPattern
from mo.front.tf.graph_utils import create_op_with_const_inputs, create_op_node_with_second_input
from mo.graph.graph import Graph
from mo.ops.concat import Concat
from mo.ops.reshape import Reshape
from mo.ops.shape import Shape
from mo.utils.shape import node_to_get_shape_value_of_indices


class ReplaceConvolutionReshape(FrontReplacementPattern):
    """
       This pass adds Reshapes around a Convolution layer for reshaping from NH to NCHW
       For example:
           Let's suppose we have next graph:

           Prev_Layer [N, H] -> Convolution [N, C, H, W] -> Next_Layer [N, H]

           In this case Convolution takes only [N, H] from input tensor in 3rd dim
           So this pass will convert this graph to the next one:

           Prev_Layer [N, H] -> Reshape -> Convolution [N, C, H, W] -> Reshape -> Next_Layer [N, H]

   """
    enabled = True
    graph_condition = [lambda graph: graph.graph['fw'] == "kaldi"]

    def run_before(self):
        from extensions.front.kaldi.add_permute_after_convolution import ReplaceConvolutionTranspose
        return [ReplaceConvolutionTranspose]

    def pattern(self):
        return dict(nodes=[('conv', dict(op='Convolution'))],
                    edges=[])

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        node = match['conv']
        node_name = node.soft_get('name', node.id)

        dst_dtype = np.float32  # even if data_type=FP16 use float32 for shape values

        # create Reshape before convolution
        # shape = [in_shape[0], t, patch_stride, in_shape[1]/(patch_stride*t)]
        i_shape = Shape(graph, {'name': node_name + '/Shape'}).create_node()
        shape = Cast(graph, {'name': node_name + '/to_float',
                             'dst_type': dst_dtype}).create_node()
        i_shape.in_port(0).connect(node.in_port(0).get_source())
        shape.in_port(0).connect(i_shape.out_port(0))

        N, H = node_to_get_shape_value_of_indices(shape, [0]), node_to_get_shape_value_of_indices(shape, [1])

        div = create_op_with_const_inputs(
            graph, Div, {1: float32_array([node.patch_stride * node.kernel[2]])}, {'name': node_name + '/div_stride_h'})
        div.in_port(0).connect(H.out_port(0))

        # added if to avoid changes in old networks where channel dimension is 1 and transpose is not needed
        # (GNA doesn't support)
        if node.kernel[1] == 1:
            channel_dim = 1
            sp_dim_1 = 2
            sp_dim_2 = 3
        else:
            channel_dim = 3
            sp_dim_1 = 1
            sp_dim_2 = 2

        concat = create_op_with_const_inputs(graph, Concat, {sp_dim_2: float32_array([node.patch_stride]),
                                                             sp_dim_1: float32_array([node.kernel[2]])},
                                             {'name': node_name + '/concat_all_dims', 'in_ports_count': 4, 'axis': 0})
        concat.in_port(0).connect(N.out_port(0))
        concat.in_port(channel_dim).connect(div.out_port(0))

        reshape_pattern = Cast(graph, {'name': node_name + '/to_int', 'dst_type': np.int64}).create_node()
        concat.out_port(0).connect(reshape_pattern.in_port(0))

        reshape_in = Reshape(graph, {'name': node_name + '/reshape_in'}).create_node()
        reshape_in.in_port(1).connect(reshape_pattern.out_port(0))

        # change layout from NHWC to NCHW
        # should be replaced by common Permute logic in future
        transpose = None
        if node.kernel[1] != 1:
            transpose = create_op_node_with_second_input(graph, Transpose, int64_array([0, 3, 1, 2]),
                                                         {'name': node.name + '/Transpose'}, reshape_in)

        # create Reshape after Convolution
        reshape_out = create_op_node_with_second_input(graph, Reshape, int64_array([0, -1]),
                                                       {'name': node_name + '/reshape_out'})

        # connect input_reshape_node
        source = node.in_port(0).get_source()
        node.in_port(0).get_connection().set_source(transpose.out_port(0) if transpose else reshape_in.out_port(0))
        reshape_in.in_port(0).connect(source)
        # connect output_reshape_node
        node.out_port(0).get_connection().set_source(reshape_out.out_port(0))
        node.out_port(0).connect(reshape_out.in_port(0))
