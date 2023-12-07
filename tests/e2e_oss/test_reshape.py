import logging as log
import sys
from copy import deepcopy

from e2e_oss.utils.modify_configs import get_original_model_importer_pipeline_config
from e2e_oss.utils.reshape_pipeline_executers import mo_pipeline_runner, ie_pipeline_runner
from e2e_oss.utils.reshape_tests_utils import compare
from e2e_oss.utils.test_utils import check_mo_precision, set_infer_precision_hint, timestamp

"""
input_descriptor is a dict which describes info about input layers
and dimensions with shapes that supposed to be changed.
Each i-th position in shapes for specified dimension is i-th test case.

Example of how input_descriptor should look like:
input_descriptor = {
                    'data': {'default_shape': (1, 3, 224, 224),
                             'layout': 'NCHW',
                             'changeable_dims': {'N': [[2], [5]], 'HW': [[None, 448]]}},
                    'data2': {'default_shape': (1, 3, 224, 224),
                             'layout': 'NCHW',
                             'changeable_dims': {'N': [[2], [None]]}}
                    }

There are 3 fields that should be described for each input layer:
1. default_shape: default shape of specified input layer 
2. layout: layout of specified input layer
3. changeable_dims: dimensions and their values i.e. shapes are supposed to be used in test
"""

pytest_plugins = ('e2e_oss.plugins.reshape_tests.conftest',)


def test_reshape(instance, configuration, prepare_test_info, inference_precision_hint):
    test_name = instance.__class__.__name__
    instance_ie_pipeline = deepcopy(instance.ie_pipeline)
    reshape_pipelines = configuration.reshape_pair
    check_mo_precision(instance_ie_pipeline)
    instance_ie_pipeline = set_infer_precision_hint(instance, instance_ie_pipeline, inference_precision_hint)

    # Name of tests group
    prepare_test_info['pytestEntrypoint'] = 'E2E: Reshape'
    prepare_test_info['inputsize'] = "__".join([f"{i}_{v}" for i, v in configuration.shapes.items()])

    supported_pipelines = ['MO', 'IE']
    assert all([reshape_pipeline in supported_pipelines for reshape_pipeline in reshape_pipelines]), \
        'Some of requested reshape pipelines is not supported, \nRequested pipelines: {}\n' \
        'Supported by tests: {}'.format(reshape_pipelines, supported_pipelines)

    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.DEBUG, stream=sys.stdout)
    log.info('Running reshape {} test'.format(test_name))

    passed_pipelines = []
    shapes = configuration.shapes

    if instance_ie_pipeline.get('postprocess'):
        del instance_ie_pipeline['postprocess']

    # running stage
    for pipeline in reshape_pipelines:
        if pipeline == 'MO':
            pipeline_runner = mo_pipeline_runner
        else:
            pipeline_runner = ie_pipeline_runner
            if configuration.skip_ir_generation:
                instance_ie_pipeline = get_original_model_importer_pipeline_config(instance_ie_pipeline)

        try:
            pipeline_result = pipeline_runner(instance_ie_pipeline, shapes, test_name)
            log.info("{} reshape pipeline for {} executed successfully\n".format(pipeline, test_name))
            passed_pipelines.append(pipeline_result)
        except Exception as err:
            raise Exception(f"{timestamp()}: {pipeline} reshape pipeline failed") from err

    # results comparing stage
    if len(passed_pipelines) == 1:
        status = compare(instance, None, passed_pipelines[0])
    else:
        status = compare(instance, [passed_pipelines[0]], passed_pipelines[1])

    assert status, 'inferred model results != reference results'
