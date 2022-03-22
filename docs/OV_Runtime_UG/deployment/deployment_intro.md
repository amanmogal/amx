# Deploy with OpenVINO {#openvino_deployment_guide}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   openvino_docs_install_guides_deployment_manager_tool
   openvino_docs_deploy_local_distribution

@endsphinxdirective

Once the [OpenVINO application development](../integrate_with_your_application.md) has been finished, usually application developers need to deploy their applications to end users. There are several ways how to achieve that:

- Set a dependency on existing prebuilt packages (so called _centralized distribution_):
    - Using Debian / RPM packages, a recommended way for a family of Linux operation systems
    - Using pip package manager on PyPi, default approach for Python-based applications
    - Using Docker images
- Grab a necessary functionality of OpenVINO together with your application (so-called _local distribution_):
    - Using [OpenVINO Deployment manager](deployment-manager-tool.md) providing a convinient way create a distribution package
    - Using advanced [Local distribution](local-distribution.md) approach
    - Using [static version of OpenVINO Runtime linked into the final app](https://github.com/openvinotoolkit/openvino/wiki/StaticLibraries)

The table below shows which distribution type can be used depending on target operation system:

| Distribution type | Operation systems |
|------- ---------- | ----------------- |
| Debian packages | Ubuntu 18.04 long-term support (LTS), 64-bit; Ubuntu 20.04 long-term support (LTS), 64-bit |
| RMP packages | Red Hat Enterprise Linux 8, 64-bit |
| Docker images | All operation systems |
| PyPi (pip package manager) | All operation systems |
| [OpenVINO Deployment Manager](deployment-manager-tool.md) | All operation systems |
| [Local distribution](local-distribution.md) | All operation systems |
| [Build OpenVINO statically and link into the final app](https://github.com/openvinotoolkit/openvino/wiki/StaticLibraries) | All operation systems |

Dependning on the distribution type, the granularity of OpenVINO packages may vary: PyPi distribution [OpenVINO has a single package 'openvino'](https://pypi.org/project/openvino/) containing all the runtime libraries and plugins, while more configurable ways like [Local distribution](local-distribution.md) provide higher granularity, so it is important to now some details about the set of libraries which are part of OpenVINO Runtime package:

![deployment_simplified]

- The main library `openvino` is used by C++ user's applications to link against with. The library provides all OpenVINO Runtime public API for both OpenVINO API 2.0 and Inference Engine, nGraph APIs. For C language applications `openvino_c` is additionally required for distribution.
- The _optional_ plugin libraries like `openvino_intel_cpu_plugin` (matching `openvino_.+_plugin` pattern) are used to provide inference capabilities on specific devices or additional capabitilies like [Hetero execution](../hetero_execution.md) or [Multi-Device execution](../multi_device.md).
- The _optional_ plugin libraries like `openvino_ir_frontnend` (matching `openvino_.+_frontend`) are used to provide capabilities to read models of different file formats like OpenVINO IR, ONNX or Paddle.

The _optional_ means that if the application does not use the capability enabled by the plugin, the plugin's library or package with the plugin is not needed in the final distribution.

The information above covers granularity aspects of majority distribution types, more detailed information is only needed and provided in [Local Distribution](local-distribution.md).

> **NOTE**: Depending on target OpenVINO devices, you also have to use [Configurations for GPU](../../install_guides/configurations-for-intel-gpu.md), [Configurations for GNA](../../install_guides/configurations-for-intel-gna.md), [Configurations for NCS2](../../install_guides/configurations-for-ncs2.md) or [Configurations for GNA](../../install_guides/installing-openvino-config-ivad-vpu.md) for proper configuration of deployed machines.

[deployment_simplified]: ../../img/deployment_simplified.png
