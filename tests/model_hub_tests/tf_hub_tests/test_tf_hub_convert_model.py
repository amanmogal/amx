# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np
import pytest
import tensorflow as tf
import tensorflow_hub as hub
from models_hub_common.test_convert_model import TestConvertModel
from models_hub_common.utils import get_models_list


class TestTFHubConvertModel(TestConvertModel):
    def load_model(self, model_name, model_link):
        load = hub.load(model_link)
        if 'default' in list(load.signatures.keys()):
            concrete_func = load.signatures['default']
        else:
            signature_keys = sorted(list(load.signatures.keys()))
            assert len(signature_keys) > 0, "No signatures for a model {}, url {}".format(model_name, model_link)
            concrete_func = load.signatures[signature_keys[0]]
        return concrete_func

    def get_inputs_info(self, model_obj):
        inputs_info = []
        for input_info in model_obj.inputs:
            input_shape = []
            for dim in input_info.shape.as_list():
                if dim is None:
                    input_shape.append(1)
                else:
                    input_shape.append(dim)
            type_map = {
                tf.float64: np.float64,
                tf.float32: np.float32,
                tf.int8: np.int8,
                tf.int16: np.int16,
                tf.int32: np.int32,
                tf.int64: np.int64,
                tf.uint8: np.uint8,
                tf.uint16: np.uint16,
                tf.string: str,
                tf.bool: bool,
            }
            assert input_info.dtype in type_map, "Unsupported input type: {}".format(input_info.dtype)
            inputs_info.append((input_shape, type_map[input_info.dtype]))

        return inputs_info

    def infer_fw_model(self, model_obj, inputs):
        tf_inputs = []
        for input_data in inputs:
            tf_inputs.append(tf.constant(input_data))
        output_dict = {}
        for out_name, out_value in model_obj(*tf_inputs).items():
            output_dict[out_name] = out_value.numpy()

        # map external tensor names to internal names
        # TODO: remove this workaround
        fw_outputs = {}
        for out_name, out_value in output_dict.items():
            mapped_name = out_name
            if out_name in model_obj.structured_outputs:
                mapped_name = model_obj.structured_outputs[out_name].name
            fw_outputs[mapped_name] = out_value
        return fw_outputs

    @pytest.mark.parametrize("model_name,model_link",
                             get_models_list(os.path.join(os.path.dirname(__file__), "precommit_models")))
    @pytest.mark.precommit
    def test_convert_model(self, model_name, model_link, ie_device):
        # we do not perform transpose in the test in case of new frontend
        self.run(model_name, model_link, ie_device)
