# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from pytorch_layer_test_class import PytorchLayerTest


class TestRelu(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.random.randn(1, 3, 224, 224).astype(np.float32),)

    def create_model(self):

        import torch
        import torch.nn.functional as F

        class aten_relu(torch.nn.Module):
            def __init__(self):
                super(aten_relu, self).__init__()

            def forward(self, x):
                return F.relu(x)

        ref_net = None

        return aten_relu(), ref_net, "aten::relu"

    @pytest.mark.nightly
    def test_relu(self, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version)
