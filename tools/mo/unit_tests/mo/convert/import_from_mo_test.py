# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
import unittest

from generator import generator, generate
from openvino.runtime import serialize
from utils import create_onnx_model, save_to_onnx
from openvino.tools.mo import InputCutInfo, LayoutMap
from unit_tests.mo.unit_test_with_mocked_telemetry import UnitTestWithMockedTelemetry



@generator
class ConvertImportMOTest(UnitTestWithMockedTelemetry):
    # Checks convert import from openvino.tools.mo
    test_directory = os.path.dirname(os.path.realpath(__file__))

    @generate(*[
        ({}),
        ({'input': InputCutInfo(name='LeakyRelu_out', shape=None, type=None, value=None)}),
        ({'layout': {'input': LayoutMap(source_layout='NCHW', target_layout='NHWC')}}),
    ])
    def test_import(self, params):
        from openvino.tools.mo import convert

        with tempfile.TemporaryDirectory(dir=self.test_directory) as tmpdir:
            model = create_onnx_model()
            model_path = save_to_onnx(model, tmpdir)
            out_xml = os.path.join(tmpdir, "model.xml")

            ov_model = convert(input_model=model_path, **params)
            serialize(ov_model, out_xml.encode('utf-8'), out_xml.replace('.xml', '.bin').encode('utf-8'))
            assert os.path.exists(out_xml)
