# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from mo.front.common.partial_infer.utils import is_fully_defined, shape_array
from mo.graph.graph import Node, Graph
from mo.ops.op import Op
from mo.utils.broadcasting import bi_directional_shape_broadcasting


class Select(Op):
    op = 'Select'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'op': self.op,
            'type': self.op,
            'version': 'opset1',
            'in_ports_count': 3,
            'out_ports_count': 1,
            'infer': self.infer,
            'type_infer': self.type_infer,
            'auto_broadcast': 'numpy'
        }
        super().__init__(graph, mandatory_props, attrs)

    def backend_attrs(self):
        return ['auto_broadcast']

    @staticmethod
    def infer(node: Node):
        assert len([port for port in node.in_ports().values() if not port.disconnected()]) == 3, \
            "Select operation must have 3 inputs: 'condition', 'then' and 'else' tensors"

        condition_value = node.in_port(0).data.get_value()
        resulting_tensors = [node.in_port(1).data.get_value(), node.in_port(2).data.get_value()]

        a_shape = node.in_port(1).data.get_shape()
        b_shape = node.in_port(2).data.get_shape()
        node.out_port(0).data.set_shape(bi_directional_shape_broadcasting(a_shape, b_shape))
        # Case with unknown condition
        if condition_value is not None and is_fully_defined(condition_value):
            fully_defined_values = is_fully_defined(resulting_tensors[0]) and is_fully_defined(resulting_tensors[1])
            output_value = np.where(condition_value, resulting_tensors[0], resulting_tensors[1])
            if condition_value.size != 1:
                if np.any(output_value is None):
                    # If any element of output value is None that means that we use the value from the 'then' or the
                    # 'else' tensor which is not defined, this means that we cannot perform value propagation.
                    output_value = None
            else:
                output_value = np.array(output_value,
                                        dtype=resulting_tensors[not np.bool(condition_value.item(0))].dtype)

            if output_value is not None and not fully_defined_values:
                output_value = shape_array(output_value)
            if output_value is not None:
                node.out_port(0).data.set_value(output_value)

    @staticmethod
    def type_infer(node: Node):
        assert node.in_port(1).get_source().get_data_type() == node.in_port(2).get_source().get_data_type(), \
            'The data type of the second and the third inputs must be equal for the node {}'.format(node.name)
        node.out_port(0).set_data_type(node.in_port(1).get_source().get_data_type())
