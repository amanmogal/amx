from collections import OrderedDict

from e2e_oss.common_utils.tf_helper import TFVersionHelper
from e2e_oss.pipelines.pipeline_base_classes.common_base_class import CommonConfig
from e2e_oss.pipelines.pipeline_templates.collect_reference_templates import collect_tf_refs_pipeline
from e2e_oss.pipelines.pipeline_templates.collect_reference_templates import read_refs_pipeline
from e2e_oss.pipelines.pipeline_templates.comparators_template import classification_comparators
from e2e_oss.pipelines.pipeline_templates.infer_templates import common_infer_step
from e2e_oss.pipelines.pipeline_templates.input_templates import read_npz_input, read_img_input
from e2e_oss.pipelines.pipeline_templates.ir_gen_templates import common_ir_generation
from e2e_oss.pipelines.pipeline_templates.preproc_templates import assemble_preproc_tf
from e2e_oss.utils.path_utils import prepend_with_env_path, resolve_file_path
from e2e_oss.utils.path_utils import ref_from_model
from e2e_oss.common_utils.pytest_utils import mark

common_input_file = resolve_file_path("test_data/inputs/tf/classification_imagenet.npz")


class TF_ClassificationNet(CommonConfig):
    """Base class for TensorFlow classification nets."""
    model = ''  # model path
    h = 0  # input image height
    w = 0  # input image width
    preproc = {}  # common input preprocessings (i.e. mean=(1, 2, 3))
    postproc = {}  # postprocessor
    infer_args = {}  # additional IE arguments
    model_env_key = "models"
    input_file = resolve_file_path("test_data/inputs/tf/classification_imagenet.npz")

    def __init__(self, batch, device, precision, **kwargs):
        self.__pytest_marks__ += (mark("classification", is_simple_mark=True),
                                  mark("tf", is_simple_mark=True))

        model_path = prepend_with_env_path(self.model_env_key, TFVersionHelper().tf_models_version, self.model)

        preprocess_args = {'batch': batch, 'h': self.h, 'w': self.w, **self.preproc}

        self.ref_collection = collect_tf_refs_pipeline(
            model=model_path, input=self.input_file, h=self.h, w=self.w, preprocessors=self.preproc)

        self.ref_pipeline = read_refs_pipeline(ref_file=ref_from_model(model_name=self.model, framework="tf"),
                                               batch=batch)
        self.ie_pipeline = OrderedDict([
            read_npz_input(path=common_input_file),
            assemble_preproc_tf(**preprocess_args),
            common_ir_generation(mo_runner=self.environment["mo_runner"],
                                 mo_out=self.environment["mo_out"],
                                 model=model_path,
                                 precision=precision, input_shape=(1, self.h, self.w, 3)),
            common_infer_step(device=device, batch=batch, **kwargs)
        ])
        self.comparators = classification_comparators(postproc=self.postproc, precision=precision, device=device)


class TF_ClassificationNet_v1(TF_ClassificationNet):
    """Class for TensorFlow classification nets with changed ie_pipeline behavior:
       input_file is used in ie_pipeline instead common_input_file"""

    def __init__(self, batch, device, precision, **kwargs):
        super().__init__(batch, device, precision, **kwargs)

        model_path = prepend_with_env_path(self.model_env_key, TFVersionHelper().tf_models_version, self.model)
        preprocess_args = {'batch': batch, 'h': self.h, 'w': self.w, **self.preproc}

        self.input_file = {'input_tensor': "test_data/inputs/tf/snake.jpeg"}
        self.ref_input_file = {'input_tensor': "test_data/inputs/tf/snake.jpeg"}

        self.ref_collection = collect_tf_refs_pipeline(
            model=model_path, input=self.ref_input_file, h=self.h, w=self.w, preprocessors=self.preproc)

        self.ie_pipeline = OrderedDict([
            read_img_input(path=self.input_file),
            assemble_preproc_tf(**preprocess_args),
            common_ir_generation(mo_runner=self.environment["mo_runner"],
                                 mo_out=self.environment["mo_out"],
                                 model=model_path,
                                 precision=precision, input_shape=(1, self.h, self.w, 3)),
            common_infer_step(device=device, batch=batch, **kwargs)
        ])

        self.comparators = classification_comparators(postproc=self.postproc, precision=precision, device=device)
