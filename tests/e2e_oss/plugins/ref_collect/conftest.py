"""Local pytest plugin for reference collection."""
import logging as log

import yaml

import e2e_oss.plugins.common.base_conftest as base
from utils.e2e.env_tools import Environment
from utils.env_utils import fix_env_conf
from utils.openvino_resources import OpenVINOResources, OpenVINOResourceNotFound


def set_env(metafunc):
    """Setup test environment."""
    with open(metafunc.config.getoption('env_conf'), "r") as f:
        Environment.env = fix_env_conf(yaml.load(f, Loader=yaml.FullLoader),
                                       root_path=str(metafunc.config.rootdir))

    # The mo_runner must be available through the OpenVINOResources.
    # The Environment.env change was made in order not to change the existing logic
    if "mo_runner" not in Environment.env:
        try:
            mo_runner = OpenVINOResources().mo_runner
        except OpenVINOResourceNotFound:
            log.warning("Alias 'mo_runner' not found, using mo.py instead")
            mo_runner = OpenVINOResources().mo_py

        Environment.env.update({"mo_runner": mo_runner,
                                "mo_root": mo_runner.parent})


def pytest_generate_tests(metafunc):
    """Pytest hook for test generation.

    Generate parameterized tests from discovered modules.
    """
    set_env(metafunc)
    test_classes, broken_modules = base.find_tests(metafunc.config.getoption('modules'),
                                                   attributes=['ref_collection', '__is_test_config__'])
    test_case = None
    for module in broken_modules:
        log.error("Broken module: {}. Import failed with error: {}".format(module[0], module[1]))
    test_cases = []
    test_ids = []
    for test in test_classes:
        # TODO: Add broken tests handling like in e2e_tests/conftest.py when `test` instance creation will be added
        name = test.__name__
        skip_ir_generation = metafunc.config.getoption("skip_ir_generation")
        try:
            test_case = test(test_id=name, batch=1, device="CPU", precision="FP32",
                             sequence_length=1, qb=8, device_mode="GNA_AUTO", api_2='api_2',
                             skip_ir_generation=skip_ir_generation).ref_collection
        except Exception as e:
            log.warning(f"Test with name {name} failed to add in row with exception {e}")

        test_ids.append(name)
        test_cases.append(test_case)
    metafunc.parametrize("reference", test_cases, ids=test_ids)


def pytest_collection_modifyitems(items):
    """
    Pytest hook for items collection
    """
    # Sort test cases to support tests' run via pytest-xdist
    items.sort(key=lambda item: item.callspec.params['reference'].__class__.__name__)
