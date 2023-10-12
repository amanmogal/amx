#
# INTEL CONFIDENTIAL
# Copyright (c) 2022 Intel Corporation
#
# The source code contained or described herein and all documents related to
# the source code ("Material") are owned by Intel Corporation or its suppliers
# or licensors. Title to the Material remains with Intel Corporation or its
# suppliers and licensors. The Material contains trade secrets and proprietary
# and confidential information of Intel or its suppliers and licensors. The
# Material is protected by worldwide copyright and trade secret laws and treaty
# provisions. No part of the Material may be used, copied, reproduced, modified,
# published, uploaded, posted, transmitted, distributed, or disclosed in any way
# without Intel's prior express written permission.
#
# No license under any patent, copyright, trade secret or other intellectual
# property right is granted to or conferred upon you by disclosure or delivery
# of the Materials, either expressly, by implication, inducement, estoppel or
# otherwise. Any license under such intellectual property rights must be express
# and approved by Intel in writing.
#
import pathlib
import sys

from cpuinfo import get_cpu_info

from .common_utils.logger import get_logger
from .common_utils.sys_info_utils import get_sys_info
from .common_utils.tf_helper import TFVersionHelper

try:
    # In user_config.py, user might export custom environment variables
    from . import user_config

    print("Successfully imported user_config")
except ImportError:
    pass

from .plugins.common.conftest import *

NODEID_TOKENS_RE = r"(?P<file>.+?)::(?P<func_name>.+?)\[(?P<args>.+?)\]"
VR_FRIENDLY_NODEID = "{class_definition_path}::{class_name}::{func_name}[{args}]"

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
logger = get_logger(__name__)


def pytest_configure(config):
    sys_info = get_sys_info()
    cpu_info = get_cpu_info()

    logger.info(f"System information: {sys_info}")
    logger.info(f"CPU info: {cpu_info}")
    # Fill environment section of HTML report with additional data
    # config._metadata['INTERNAL_GFX_DRIVER_VERSION'] = os.getenv('INTERNAL_GFX_DRIVER_VERSION')
    # Set TensorFlow models version with command line option value
    tf_models_version = config.getoption("tf_models_version")
    TFVersionHelper(tf_models_version)


@pytest.mark.hookwrapper
def pytest_runtest_makereport(item, call):
    pytest_html = item.config.pluginmanager.getplugin('html')
    report = (yield).get_result()
    extra = getattr(report, 'extra', [])
    ir_links = []
    if report.when == 'call':
        ir_link = next((p[1] for p in report.user_properties if p[0] == "ir_link"), None)
        if ir_link:
            extra.append(pytest_html.extras.url(ir_link, name="xml"))
            extra.append(pytest_html.extras.url(ir_link.replace(".xml", ".bin"), name="bin"))
            extra.append(pytest_html.extras.url(ir_link.replace(".xml", ".mo_log.txt"), name="mo_log"))

            ir_links.append(f"<a class=\"url\" href=\"{ir_link}\" target=\"_blank\">xml</a>")
            ir_links.append(f"<a class=\"url\" href=\"{ir_link.replace('.xml', '.bin')}\" target=\"_blank\">bin</a>")
            ir_links.append(f"<a class=\"url\" href=\"{ir_link.replace('.xml', '.mo_log.txt')}\" "
                            f"target=\"_blank\">mo_log</a>")
        if getattr(item._request, 'test_info', None):
            item._request.test_info.update(
                {"links": " ".join(ir_links),
                 "log": "\n\n\n".join([report.caplog, report.longreprtext]),
                 "insertTime": report.duration,
                 "duration": report.duration,
                 "result": report.outcome}
            )
    report.extra = extra

