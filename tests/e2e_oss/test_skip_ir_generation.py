"""Main entry-point to run E2E OSS tests without IR generation.

Default run:
$ pytest test.py

Options[*]:
--modules       Paths to tests
--env_conf      Path to environment config
--test_conf     Path to test config

[*] For more information see conftest.py
"""
# pylint:disable=invalid-name
import logging as log
import os
from shutil import rmtree

import yaml

from e2e_oss.common_utils.logger import get_logger
from e2e_oss.utils.modify_configs import get_original_model_importer_pipeline_config
from e2e_oss.utils.test_utils import log_timestamp, set_infer_precision_hint, store_data_to_csv, timestamp
from e2e_oss.common_utils.parsers import pipeline_cfg_to_string
from utils.e2e.common.pipeline import Pipeline
from utils.e2e.comparator.container import ComparatorsContainer
from utils.e2e.env_tools import Environment

pytest_plugins = ('e2e_oss.plugins.e2e_test.conftest',)

log = get_logger(__name__)


def _test_run(instance, load_net_to_plug_time_csv_name, mem_usage_ie_csv_name, skip_mo_args, prepare_test_info,
              copy_input_files, inference_precision_hint):

    """Parameterized test.

    :param instance: test instance
    :param load_net_to_plug_time_csv_name: name for csv file with load net to plugin time
    :param mem_usage_ie_csv_name: name for csv file with IE memory usage information
    :param skip_mo_args: line with comma separated args that will be deleted from MO cmd line
    :param instance: test instance
    """
    model_framework_name = instance.__class__.__name__.split('_')[0]
    ir_version = "v11"

    if 'ONNX' or 'Precollected' or 'Caffe2' or 'WinML' in instance.__class__.__name__:
        prepare_test_info['pytestEntrypoint'] = 'E2E: ONNX Without MO step'
    if 'PDPD' == model_framework_name:
        prepare_test_info['pytestEntrypoint'] = 'E2E: PDPD Without MO step'
    if 'TF' == model_framework_name:
        prepare_test_info['pytestEntrypoint'] = 'E2E: TF Without MO step'
    if 'TFLite' == model_framework_name:
        prepare_test_info['pytestEntrypoint'] = 'E2E: TFLite Without MO step'

    log.info("Running {test_id} test".format(test_id=instance.test_id))
    instance.prepare_prerequisites()
    instance.use_mo_mapping = False
    instance_ie_pipeline = instance.ie_pipeline
    instance_ref_pipeline = instance.ref_pipeline

    ref_pipeline = Pipeline(instance_ref_pipeline)
    log.debug("Test scenario:")
    log.debug("Reference Pipeline:\n{}".format(pipeline_cfg_to_string(ref_pipeline._config)))
    if ref_pipeline.steps:
        with log_timestamp('reference pipeline'):
            log.info("Running reference pipeline:")
            ref_pipeline.run()
    else:
        log.warning("Reference pipeline is empty, no results comparison will be performed")

    if instance_ie_pipeline.get('infer'):
        instance_ie_pipeline = set_infer_precision_hint(instance, instance_ie_pipeline, inference_precision_hint)

    instance_ie_pipeline = get_original_model_importer_pipeline_config(instance_ie_pipeline)

    ie_pipeline = Pipeline(instance_ie_pipeline)
    log.debug("Inference Pipeline:\n{}".format(pipeline_cfg_to_string(ie_pipeline._config)))
    ie_pipeline.run()

    if load_net_to_plug_time_csv_name or mem_usage_ie_csv_name:
        infer_provider_index = [count for count, step in enumerate(ie_pipeline.steps) if 'infer' in str(step)]
        assert len(infer_provider_index) == 1, 'Several steps for ie_infer'
        if load_net_to_plug_time_csv_name:
            store_data_to_csv(csv_path=os.path.join(Environment.abs_path("mo_out"), load_net_to_plug_time_csv_name),
                              instance=instance, device=ie_pipeline.steps[infer_provider_index[0]].executor.device,
                              ir_version=ir_version, data_name='load_net_to_plug_time',
                              data=ie_pipeline.steps[infer_provider_index[0]].executor.load_net_to_plug_time,
                              skip_mo_args=skip_mo_args)
        if mem_usage_ie_csv_name:
            store_data_to_csv(csv_path=os.path.join(Environment.abs_path("mo_out"), mem_usage_ie_csv_name),
                              instance=instance, device=ie_pipeline.steps[infer_provider_index[0]].executor.device,
                              ir_version=ir_version, data_name='mem_usage_ie',
                              data=ie_pipeline.steps[infer_provider_index[0]].executor.mem_usage_ie,
                              skip_mo_args=skip_mo_args)

    comparators = ComparatorsContainer(
        config=instance.comparators,
        infer_result=ie_pipeline.fetch_results(),
        reference=ref_pipeline.fetch_results(),
        result_aligner=getattr(instance, 'align_results', None, ),
    )

    log.info("Running comparators:")
    with log_timestamp('comparators'):
        comparators.apply_postprocessors()
        comparators.apply_all()
    status = comparators.report_statuses()
    assert status, "inferred model results != reference results"


def empty_dirs(env_conf):
    test_config = None
    with open(env_conf, 'r') as fd:
        test_config = yaml.load(fd, Loader=yaml.FullLoader)

    for env_clean_dir_flag, test_cfg_dir_to_clean in [("TT_CLEAN_MO_OUT_DIR", 'mo_out'),
                                                      ("TT_CLEAN_PREGEN_IRS_DIR", 'pregen_irs_path'),
                                                      ("TT_CLEAN_INPUT_MODEL_DIR", 'input_model_dir')]:
        clean_flag = True if os.environ.get(env_clean_dir_flag, 'False') == 'True' else False
        if clean_flag:
            dir_to_clean = test_config.get(test_cfg_dir_to_clean, '')
            if os.path.exists(dir_to_clean):
                log.info(f"Clear {dir_to_clean} dir")
                rmtree(dir_to_clean)


def test_run(instance, load_net_to_plug_time_csv_name, mem_usage_ie_csv_name, skip_mo_args, prepare_test_info,
             copy_input_files, env_conf, inference_precision_hint):

    try:
        _test_run(instance, load_net_to_plug_time_csv_name, mem_usage_ie_csv_name, skip_mo_args, prepare_test_info,
                  copy_input_files, inference_precision_hint)

    except Exception as ex:
        raise Exception(f'{timestamp()}') from ex
    finally:
        empty_dirs(env_conf)
