import logging as log
import os
import sys
from pathlib import Path

from utils.e2e.ref_collector.provider import ClassProvider


os.environ["GLOG_minloglevel"] = "3"


class ScorePaddlePaddle(ClassProvider):
    """Reference collector for PaddlePaddle models."""

    __action_name__ = "score_paddlepaddle"
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)

    def __init__(self, config):
        """
        ScorePaddlePaddle initialization
        :param config: dictionary with class configuration parameters:
        required config keys:
            model: model path which will be used in get_model() function
        optional config keys:
            params_filename: the name of single binary file to load all model parameters.
            cast_input_data_to_type: type of data model input data cast to.
        """
        self.model = Path(config["model"])
        self.params_filename = config.get("params_filename", None)
        self.cast_input_data_to_type = config.get("cast_input_data_to_type", "float32")
        self.res = {}

    def get_refs(self, input_data: dict):
        """Return PaddlePaddle model reference results."""
        import paddle

        log.info("Running inference with PaddlePaddle ...")

        for layer, data in input_data.items():
            input_data[layer] = data.astype(self.cast_input_data_to_type)

        executor = paddle.fluid.Executor(paddle.fluid.CPUPlace())

        paddle.enable_static()
        inference_program, _, output_layers = paddle.fluid.io.load_inference_model(
            executor=executor,
            dirname=self.model.parent,
            model_filename=self.model.name,
            params_filename=self.params_filename
        )
        out = executor.run(inference_program, feed=input_data, fetch_list=output_layers, return_numpy=False)
        self.res = dict(zip(map(lambda layer: layer.name, output_layers), out))

        log.info("PaddlePaddle reference collected successfully")
        return self.res
