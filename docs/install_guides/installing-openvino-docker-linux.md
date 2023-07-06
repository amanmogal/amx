# Install Intel® Distribution of OpenVINO™ toolkit for Linux from a Docker Image {#openvino_docs_install_guides_installing_openvino_docker_linux}

This guide provides steps on creating a Docker image with Intel® Distribution of OpenVINO™ toolkit for Linux and using the image on different devices.

## <a name="system-requirements-docker-linux"></a>System Requirements

@sphinxdirective
.. tab:: Target Operating Systems with Python Versions

  +----------------------------------------------+-------------------------+
  | Operating System                             | Included Python Version |
  +==============================================+=========================+
  | Ubuntu 18.04 long-term support (LTS), 64-bit |  3.8                    |
  +----------------------------------------------+-------------------------+
  | Ubuntu 20.04 long-term support (LTS), 64-bit |  3.8                    |
  +----------------------------------------------+-------------------------+
  | Red Hat Enterprise Linux 8, 64-bit           |  3.8                    |
  +----------------------------------------------+-------------------------+

.. tab:: Host Operating Systems

  * Linux
  * Windows Subsystem for Linux 2 (WSL2) on CPU or GPU
  * macOS on CPU only

  To launch a Linux image on WSL2 when trying to run inferences on a GPU, make sure that the following requirements are met:

  - Only Windows 10 with 21H2 update or above installed and Windows 11 are supported.
  - Intel GPU driver for Windows, version 30.0.100.9684 or newer needs to be installed. For more details, refer to
     `this article at intel.com <https://www.intel.com/content/www/us/en/artificial-intelligence/harness-the-power-of-intel-igpu-on-your-machine.html#articleparagraph_983312434>`__.
  - Currently, the Docker images contain preinstalled recommended version of OpenCL Runtime with WSL2 support.

 
@endsphinxdirective

## Installation Flow

There are two ways to install OpenVINO with Docker. You can choose either of them according to your needs:
* Use a prebuilt image. Do the following steps:
  1. <a href="#get-prebuilt-image-docker-linux">Get a prebuilt image from provided sources</a>.
  2. <a href="#run-image-docker-linux">Run the image on different devices</a>. To run inferences on Intel® Vision Accelerator Design with Intel® Movidius™ VPUs, <a href="#set-up-hddldaemon">configure the Docker image</a> first before you run the image.
  3. <a href="#run-samples-docker-linux">(Optional) Run samples in the Docker image</a>.
* If you want to customize your image, you can also build a Docker image manually by using the following steps:
  1. <a href="#prepare-dockerfile-linux">Prepare a Dockerfile</a>.
  2. <a href="#configure-image-docker-linux">Configure the Docker image</a>.
  3. <a href="#run-image-docker-linux">Run the image on different devices</a>.
  4. <a href="#run-samples-docker-linux">(Optional) Run samples in the Docker image</a>.

## <a name="get-prebuilt-image-docker-linux"></a>Getting a Prebuilt Image from Provided Sources

You can find prebuilt images on:

- [Docker Hub](https://hub.docker.com/u/openvino)
- [Red Hat Quay.io](https://quay.io/organization/openvino)
- [Red Hat Ecosystem Catalog (runtime image)](https://catalog.redhat.com/software/containers/intel/openvino-runtime/606ff4d7ecb5241699188fb3)
- [Red Hat Ecosystem Catalog (development image)](https://catalog.redhat.com/software/containers/intel/openvino-dev/613a450dc9bc35f21dc4a1f7)
- [Azure Marketplace](https://azuremarketplace.microsoft.com/en-us/marketplace/apps/intel_corporation.openvino)

## <a name="prepare-dockerfile-linux"></a>Preparing a Dockerfile

You can use the [available Dockerfiles on GitHub](https://github.com/openvinotoolkit/docker_ci/tree/master/dockerfiles) or generate a Dockerfile with your settings via [DockerHub CI Framework](https://github.com/openvinotoolkit/docker_ci) which can generate a Dockerfile, build, test and deploy an image with the Intel® Distribution of OpenVINO™ toolkit.
You can also try our [Tutorials](https://github.com/openvinotoolkit/docker_ci/tree/master/docs/tutorials) which demonstrate the usage of Docker containers with OpenVINO.

## <a name="configure-image-docker-linux"></a>Configuring the Image for Different Devices

If you want to run inference on a CPU or Intel® Neural Compute Stick 2, no extra configuration is needed. Go to <a href="#run-image-docker-linux">Running the image on different devices</a> for the next step.

### Configuring Docker Image for GPU

@sphinxdirective
If you want to run inference on a GPU, follow the instructions provided in the guide on 
:doc:`Configuration for Intel GPU <openvino_docs_install_guides_configurations_for_intel_gpu>`
@endsphinxdirective

### <a name="set-up-hddldaemon"></a>Configuring Docker Image for Intel® Vision Accelerator Design with Intel® Movidius™ VPUs

> **NOTE**: When building the Docker image, create a user in the Dockerfile that has the same UID (User Identifier) and GID (Group Identifier) as the user which that runs hddldaemon on the host, and then run the application in the Docker image with this user. This step is necessary to run the container as a non-root user.

To s:use the Docker container for inference on Intel® Vision Accelerator Design with Intel® Movidius™ VPUs, do the following:

1. Set up the environment on the host machine to be used for running Docker. It is required to execute `hddldaemon`, which is responsible for communication between the HDDL plugin and the board. To learn how to set up the environment (the OpenVINO package or HDDL package must be pre-installed), see [Configuration guide for HDDL device](https://github.com/openvinotoolkit/docker_ci/blob/master/install_guide_vpu_hddl.md) or [Configurations for Intel® Vision Accelerator Design with Intel® Movidius™ VPUs on Linux](configurations-for-ivad-vpu.md).
2. Run `hddldaemon` on the host in a separate terminal session using the following command:
    ```sh
    $HDDL_INSTALL_DIR/hddldaemon
    ```

## <a name="run-image-docker-linux"></a>Running the Docker Image on Different Devices

### Running the Image on CPU

Run the Docker image with the following command:
```
docker run -it --rm <image_name>
```

Note the following things:

- Kernel reports the same information for all containers as for native application, for example, CPU, memory information.
- All instructions that are available to host process available for process in container, including, for example, AVX2, AVX512. No restrictions.
- Docker does not use virtualization or emulation. The process in Docker is just a regular Linux process, but it is isolated from external world on kernel level. Performance loss is minor.


### Running the Image on GPU

> **NOTE**: Only Intel® integrated graphics are supported.

Note the following things:

- GPU is not available in the container by default. You must attach it to the container.
- Kernel driver must be installed on the host.
- In the container, non-root user must be in the `video` and `render` groups. To add a user to the render group, follow the [Configuration Guide for the Intel® Graphics Compute Runtime for OpenCL™ on Ubuntu 20.04](https://github.com/openvinotoolkit/docker_ci/blob/master/configure_gpu_ubuntu20.md).

To make GPU available in the container, attach the GPU to the container using `--device /dev/dri` option and run the container:

* Ubuntu 18 or RHEL 8:
    ```sh
    docker run -it --rm --device /dev/dri <image_name>
    ```
    > **NOTE**: If your host system is Ubuntu 20, follow the [Configuration Guide for the Intel® Graphics Compute Runtime for OpenCL™ on Ubuntu* 20.04](https://github.com/openvinotoolkit/docker_ci/blob/master/configure_gpu_ubuntu20.md).

* WSL2:
    ```sh
    docker run -it --rm --device /dev/dxg --volume /usr/lib/wsl:/usr/lib/wsl <image_name>
    ```
    > **NOTE**: To launch a Linux image on WSL2, make sure that the additional requirements in <a href="#system-requirements">System Requirements</a> are met.


### Running the Image on Intel® Neural Compute Stick 2

Run the Docker image with the following command:
```sh
docker run -it --rm --device-cgroup-rule='c 189:* rmw' -v /dev/bus/usb:/dev/bus/usb <image_name>
```

While the command above is not working, you can also run container in the privileged mode, enable the Docker network configuration as host, and mount all devices to the container. Run the following command:
```sh
docker run -it --rm --privileged -v /dev:/dev --network=host <image_name>
```

> **NOTE**: This option is not recommended, as conflicts with Kubernetes and other tools that use orchestration and private networks may occur. Please use it with caution and only for troubleshooting purposes.

#### Known Limitations

- Intel® Neural Compute Stick 2 device changes its VendorID and DeviceID during execution and each time looks for a host system as a brand new device. It means it cannot be mounted as usual.
- UDEV events are not forwarded to the container by default, and it does not know about the device reconnection. The prebuilt Docker images and provided Dockerfiles include `libusb` rebuilt without UDEV support.
- Only one NCS2 device connected to the host can be used when running inference in a container.


### Running the Image on Intel® Vision Accelerator Design with Intel® Movidius™ VPUs

> **NOTE**: To run inferences on Intel® Vision Accelerator Design with Intel® Movidius™ VPUs, make sure that you have <a href="#set-up-hddldaemon">configured the Docker image</a> first.

Use the following command:
```sh
docker run -it --rm --device=/dev/ion:/dev/ion -v /var/tmp:/var/tmp <image_name>
```

If your application runs inference of a network with a big size (>4MB) of input/output, the HDDL plugin will use shared memory. In this case, you must mount `/dev/shm` as volume:
```sh
docker run -it --rm --device=/dev/ion:/dev/ion -v /var/tmp:/var/tmp -v /dev/shm:/dev/shm <image_name>
```

Note the following things:
* The device `/dev/ion` needs to be shared to be able to use ion buffers among the plugin, `hddldaemon` and the kernel.
* Since separate inference tasks share the same HDDL service communication interface (the service creates mutexes and a socket file in `/var/tmp`), `/var/tmp` needs to be mounted and shared among them.


#### If the ion Driver is Not Enabled

In some cases, the ion driver is not enabled (for example, due to a newer kernel version or iommu (Input-Output Memory Management Unit) incompatibility). `lsmod | grep myd_ion` returns empty output. To resolve this issue, use the following command:
```sh
docker run -it --rm --ipc=host --net=host -v /var/tmp:/var/tmp <image_name>
```
If that still does not solve the issue, try starting `hddldaemon` with the root user on host. However, this approach is not recommended. Please use with caution.


## <a name="run-samples-docker-linux"></a>Running Samples in Docker Image

To run the `Hello Classification Sample` on a specific inference device, run the following commands:

**CPU**:

```sh
docker run -it --rm <image_name>
/bin/bash -c "cd ~ && omz_downloader --name googlenet-v1 --precisions FP16 && omz_converter --name googlenet-v1 --precision FP16 && curl -O https://storage.openvinotoolkit.org/data/test_data/images/car_1.bmp && python3 /opt/intel/openvino/samples/python/hello_classification/hello_classification.py public/googlenet-v1/FP16/googlenet-v1.xml car_1.bmp CPU"
```

**GPU**:

```sh
docker run -itu root:root  --rm --device /dev/dri:/dev/dri <image_name>
/bin/bash -c "omz_downloader --name googlenet-v1 --precisions FP16 && omz_converter --name googlenet-v1 --precision FP16 && curl -O https://storage.openvinotoolkit.org/data/test_data/images/car_1.bmp && python3 samples/python/hello_classification/hello_classification.py public/googlenet-v1/FP16/googlenet-v1.xml car_1.bmp GPU"
```

**MYRIAD**:

```sh
docker run -itu root:root --rm --device-cgroup-rule='c 189:* rmw' -v /dev/bus/usb:/dev/bus/usb <image_name>
/bin/bash -c "omz_downloader --name googlenet-v1 --precisions FP16 && omz_converter --name googlenet-v1 --precision FP16 && curl -O https://storage.openvinotoolkit.org/data/test_data/images/car_1.bmp && python3 samples/python/hello_classification/hello_classification.py public/googlenet-v1/FP16/googlenet-v1.xml car_1.bmp MYRIAD"
```

**HDDL**:

```sh
docker run -itu root:root --rm --device=/dev/ion:/dev/ion -v /var/tmp:/var/tmp -v /dev/shm:/dev/shm <image_name>
/bin/bash -c "omz_downloader --name googlenet-v1 --precisions FP16 && omz_converter --name googlenet-v1 --precision FP16 && curl -O https://storage.openvinotoolkit.org/data/test_data/images/car_1.bmp && umask 000 && python3 samples/python/hello_classification/hello_classification.py public/googlenet-v1/FP16/googlenet-v1.xml car_1.bmp HDDL"
```

## Additional Resources

- [DockerHub CI Framework](https://github.com/openvinotoolkit/docker_ci) for Intel® Distribution of OpenVINO™ toolkit. The Framework can generate a Dockerfile, build, test, and deploy an image with the Intel® Distribution of OpenVINO™ toolkit. You can reuse available Dockerfiles, add your layer and customize the image of OpenVINO™ for your needs.
- Intel® Distribution of OpenVINO™ toolkit home page: [https://software.intel.com/en-us/openvino-toolkit](https://software.intel.com/en-us/openvino-toolkit)
- Intel® Neural Compute Stick 2 Get Started: [https://software.intel.com/en-us/neural-compute-stick/get-started](https://software.intel.com/en-us/neural-compute-stick/get-started)
- [OpenVINO Installation Selector Tool](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html)