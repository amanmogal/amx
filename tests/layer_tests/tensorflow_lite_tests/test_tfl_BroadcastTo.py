import itertools

import pytest
import tensorflow as tf

from common.tflite_layer_test_class import TFLiteLayerTest
from common.utils.tflite_utils import data_generators

test_ops = [
    {'op_name': 'BROADCAST_TO', 'op_func': tf.broadcast_to},
]

test_params = [
    {'shape': [1, 1, 1, 1, 40], 'broadcast_shape': [1, 1, 56, 56, 40]},
    {'shape': [1, 1, 1, 1, 10], 'broadcast_shape': [1, 1, 10, 10, 10]}
]


test_data = list(itertools.product(test_ops, test_params))
for i, (parameters, shapes) in enumerate(test_data):
    parameters.update(shapes)
    test_data[i] = parameters.copy()


class TestTFLiteBroadcastToLayerTest(TFLiteLayerTest):
    inputs = ["Input"]
    outputs = ["BroadcastToOP"]

    def _prepare_input(self, inputs_dict, generator=None):
        if generator is None:
            return super()._prepare_input(inputs_dict)
        return data_generators[generator](inputs_dict)

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'op_name', 'op_func', 'shape', 'broadcast_shape'})) == 4, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        self.allowed_ops = [params['op_name']]
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            place_holder = tf.compat.v1.placeholder(params.get('dtype', tf.float32), params['shape'],
                                                    name=TestTFLiteBroadcastToLayerTest.inputs[0])
            shape_of = tf.reshape(place_holder, params['shape'])
            params['op_func'](input=shape_of, shape=params['broadcast_shape'],
                              name=TestTFLiteBroadcastToLayerTest.outputs[0])
            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_broadcast_to(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, params)
