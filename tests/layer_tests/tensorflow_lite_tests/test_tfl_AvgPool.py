import itertools

import pytest
import tensorflow as tf

from common.tflite_layer_test_class import TFLiteLayerTest
from common.utils.tflite_utils import data_generators, additional_test_params

test_ops = [
    {'op_name': 'AVERAGE_POOL_2D', 'op_func': tf.nn.avg_pool2d},
]

test_params = [
    # TFLite doesn't support avgpool with 'NCHW' format
    {'shape': [1, 22, 22, 8], 'ksize': [3, 3], 'strides': 2, 'padding': 'SAME', 'data_format': 'NHWC'},
    {'shape': [1, 22, 22, 8], 'ksize': [3, 3], 'strides': (2, 1), 'padding': 'SAME', 'data_format': 'NHWC'},
    {'shape': [1, 22, 22, 8], 'ksize': [3, 3], 'strides': 2, 'padding': 'VALID', 'data_format': 'NHWC'},
    {'shape': [1, 22, 22, 8], 'ksize': [3, 3], 'strides': (3, 4), 'padding': 'VALID', 'data_format': 'NHWC'},
]


test_data = list(itertools.product(test_ops, test_params))
for i, (parameters, shapes) in enumerate(test_data):
    parameters.update(shapes)
    test_data[i] = parameters.copy()


class TestTFLiteAvgPoolLayerTest(TFLiteLayerTest):
    inputs = ["Input"]
    outputs = ["AvgPool"]

    def _prepare_input(self, inputs_dict, generator=None):
        if generator is None:
            return super()._prepare_input(inputs_dict)
        return data_generators[generator](inputs_dict)

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'op_name', 'op_func', 'shape', 'ksize', 'strides',
                                                    'padding', 'data_format'})) == 7, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        self.allowed_ops = [params['op_name']]
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            place_holder = tf.compat.v1.placeholder(params.get('dtype', tf.float32), params['shape'],
                                                    name=TestTFLiteAvgPoolLayerTest.inputs[0])
            params['op_func'](place_holder, params['ksize'], params['strides'],
                              params['padding'], params['data_format'], name=TestTFLiteAvgPoolLayerTest.outputs[0])
            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_avg_pool(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, params)
