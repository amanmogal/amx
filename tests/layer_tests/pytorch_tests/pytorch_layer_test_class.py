# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import itertools
import warnings

import numpy as np
from common.constants import test_device, test_precision

from openvino.frontend import FrontEndManager
from openvino.frontend.pytorch import TorchScriptPythonDecoder
from openvino.runtime import Core, Type, PartialShape


class PytorchLayerTest:
    _type_map = {
        "float64": Type.f64,
        "float32": Type.f32,
        "int32": Type.i32
    }

    def _test(self, model, ref_net, ie_device, precision, ir_version, infer_timeout=60, **kwargs):
        """
        :param enabled_transforms/disabled_transforms: string with idxs of transforms that should be enabled/disabled.
                                                       Example: "transform_1,transform_2"
        """
        import torch
        with torch.no_grad():
            model.eval()
            model = torch.jit.freeze(torch.jit.script(model))

            fe_manager = FrontEndManager()
            fe = fe_manager.load_by_framework('pytorch')

            decoder = TorchScriptPythonDecoder(model.inlined_graph)

            im = fe.load(decoder)
            om = fe.convert(im)

        if 'kwargs_to_prepare_input' in kwargs and kwargs['kwargs_to_prepare_input']:
            inputs = self._prepare_input(kwargs['kwargs_to_prepare_input'])
        else:
            inputs = self._prepare_input()

        params = om.get_parameters()
        # todo: support lists and dicts
        for i in range(len(inputs)):
            inp = inputs[i]
            assert inp.dtype.name in self._type_map, f"Unknown type {inp.dtype}."
            params[i].set_element_type(self._type_map[inp.dtype.name])
            dyn_shape = [-1] * len(inp.shape)
            params[i].set_partial_shape(PartialShape(dyn_shape))
        om.validate_nodes_and_infer_types()

        # OV infer:
        core = Core()
        compiled = core.compile_model(om, ie_device)
        infer_res = compiled(inputs)

        if hasattr(self, 'skip_framework') and self.skip_framework:
            warnings.warn('Framework is skipped')
            return

        # Framework infer:
        torch_inps = [torch.from_numpy(inp) for inp in inputs]
        fw_res = model(*torch_inps)

        # check if results dtypes match
        assert torch.tensor(np.array(list(infer_res.values()))).dtype == fw_res.dtype

        if not isinstance(fw_res, tuple):
            fw_res = (fw_res,)

        if 'custom_eps' in kwargs and kwargs['custom_eps'] is not None:
            custom_eps = kwargs['custom_eps']
        else:
            custom_eps = 1e-4

        # Compare Ie results with Framework results
        fw_eps = custom_eps if precision == 'FP32' else 5e-2
        for i in range(len(infer_res)):
            cur_fw_res = fw_res[i].numpy()
            cur_ov_res = infer_res[compiled.output(i)]
            print(f"fw_re: {cur_fw_res}; ov_res: {cur_ov_res}")
            if not np.allclose(cur_ov_res, cur_fw_res,
                            atol=fw_eps,
                            rtol=fw_eps):
                is_ok = False
                print("Max diff is {}".format(
                    np.array(
                        abs(cur_ov_res - cur_fw_res)).max()))
            else:
                print("Accuracy validation successful!\n")
                print("absolute eps: {}, relative eps: {}".format(fw_eps, fw_eps))

    # Each model should specify inputs
    def _prepare_input(self):
        raise RuntimeError("Please provide inputs generation function")


def get_params(ie_device=None, precision=None):
    """
    :param ie_device: list of devices
    :param precision: list of precisions
    """

    ie_device_params = ie_device if ie_device else test_device
    precision_params = precision if precision else test_precision

    test_args = []
    for element in itertools.product(ie_device_params, precision_params):
        if element[0] == 'CPU' and element[1] == 'FP16':
            continue
        test_args.append(element)
    return test_args
