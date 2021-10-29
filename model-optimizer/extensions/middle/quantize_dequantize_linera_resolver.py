# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from extensions.ops.Cast import Cast
from extensions.ops.elementwise import Mul
from extensions.ops.fakequantize import FakeQuantize
from mo.front.common.partial_infer.utils import float_array, int64_array
from mo.front.tf.graph_utils import create_op_with_const_inputs
from mo.graph.graph import Graph, rename_nodes
from mo.middle.replacement import MiddleReplacementPattern
from extensions.middle.quantize_linear_resolver import QuantizeLinearResolver
from mo.ops.const import Const
from mo.ops.reshape import Reshape
from mo.utils.error import Error


class QuantizeDequantizeLinearResolver(MiddleReplacementPattern):
    """
    Replaces QuantizeLinear with FakeQuantize
    Transformation result depends on the axis value.
    If the axis is not set or x_scale input is scalar or 1D tensor with one element then QuantizeLinear is
    replaced with the sub-graph which can be expressed with the following formula:
        QuantizeLinear -> FakeQuantize(input
                                       Mul(y_scale, Const(low_value))
                                       Mul(y_scale, Const(high_value))
                                       Const(low_value)
                                       Const(high_value))
        low_value and high_value depend on from y_zero_point type
    In other cases y_scale and y_zero_point can be transform with addition reshape.
    Target shape for y_scale and y_zero_point depend on axis value.
    """
    enabled = True
    graph_condition = [lambda graph: graph.graph['layout'] == 'NCHW']
    force_clean_up = True

    def run_after(self):
        from extensions.middle.quantize_fuses import MarkNodesToFuseUpToFakeQuantize
        return [MarkNodesToFuseUpToFakeQuantize]

    def find_and_replace_pattern(self, graph: Graph):
        for dequantize_node in graph.get_op_nodes(op='DequantizeLinear'):
            if dequantize_node.is_in_port_connected(0):
                quantize_node = dequantize_node.in_port(0).get_source().node
                if quantize_node.soft_get('op') != 'QuantizeLinear':
                    continue
                scale_zerop_is_exist = quantize_node.is_in_port_connected(1) and \
                                    quantize_node.is_in_port_connected(2) and \
                                    dequantize_node.is_in_port_connected(1) and \
                                    dequantize_node.is_in_port_connected(2)
                if not scale_zerop_is_exist:
                    continue
                q_scale = quantize_node.in_port(1).get_source().node
                q_zerop = quantize_node.in_port(2).get_source().node
                dq_scale = dequantize_node.in_port(1).get_source().node
                dq_zerop = dequantize_node.in_port(2).get_source().node
                scales_and_zerop_is_const = q_scale.soft_get('type') == 'Const' and \
                    dq_scale.soft_get('type') == 'Const' and q_zerop.soft_get('type') == 'Const' and \
                    dq_zerop.soft_get('type') == 'Const'
                scales_and_zerop_equals = np.array_equal(q_scale.value, dq_scale.value) and \
                    np.array_equal(q_zerop.value, dq_zerop.value)
                # only constant as for zero_point/scale supported
                # only patterns with same scale/zero_point values for Q and DQ are supported
                if not (scales_and_zerop_is_const or scales_and_zerop_equals):
                    continue
                fq_node_with_cast = QuantizeLinearResolver.quantize_to_fakequantize(graph, quantize_node)
                fq_node_with_cast['stop_value_propagation'] = True
                dequantize_node.out_port(0).get_connection().set_source(fq_node_with_cast.out_port(0))
                dq_name = dequantize_node.soft_get('name', dequantize_node.id)
                rename_nodes([(dequantize_node, dq_name + '/to_be_removed'), (fq_node_with_cast, dq_name)])



