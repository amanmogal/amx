# -*- coding: utf-8 -*-
# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import openvino
from openvino.runtime import Model, Core, Shape, Type
from openvino.runtime.op import Parameter
import openvino.runtime.opset8 as ops
from typing import Tuple, Union, List
import numpy as np


def get_test_model():
    element_type = Type.f32
    param = Parameter(element_type, Shape([1, 3, 22, 22]))
    relu = ops.relu(param)
    model = Model([relu], [param], "test")
    assert model is not None
    return model


def test_compare_models():
    try:
        from openvino.test_utils import compare_models
        model = get_test_model()
        status, _ = compare_models(model, model)
        assert status
    except RuntimeError:
        print("openvino.test_utils.compare_models is not available")


def generate_image(shape: Tuple = (1, 3, 32, 32), dtype: Union[str, np.dtype] = "float32") -> np.array:
    np.random.seed(42)
    return np.random.rand(*shape).astype(dtype)


def generate_relu_model(input_shape: List[int]) -> openvino.runtime.ie_api.CompiledModel:
    param = ops.parameter(input_shape, np.float32, name="parameter")
    relu = ops.relu(param, name="relu")
    model = Model([relu], [param], "test")
    model.get_ordered_ops()[2].friendly_name = "friendly"

    core = Core()
    return core.compile_model(model, "CPU", {})


def generate_add_model() -> openvino.pyopenvino.Model:
    param1 = ops.parameter(Shape([2, 1]), dtype=np.float32, name="data1")
    param2 = ops.parameter(Shape([2, 1]), dtype=np.float32, name="data2")
    add = ops.add(param1, param2)
    return Model(add, [param1, param2], "TestFunction")
