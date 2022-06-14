# Troubleshooting Steps for OpenVINO™ Installation and Configurations {#openvino_docs_get_started_guide_troubleshooting_steps}

If you run into issues while installing or configuring OpenVINO™, you can try the following methods to do some quick checks first. 

## Check the versions of OpenVINO Runtime and Developement Tools

* To check the version of OpenVINO Development Tools, use the following command:
   ```sh
   mo --version
   ```
* To check the version of OpenVINO Runtime, use the following command:
   ```sh
   from openvino.runtime import get_version get_version()
   ```

## Check the versions of Python and PIP

To check your Python version, run `python -VV` or `python --version`. The supported Python versions should be 64-bit and between 3.7 and 3.9. Note that Python 3.6 is not supported anymore.

If your Python version does not meet the requirements, update Python:

* For Windows, **do not install Python from a Windows Store** as it can cause issues. You are highly recommended to install Python from <https://www.python.org/>.
* For Ubuntu and other Linux systems, use the Python version comes with the system, or install the `libpython3.X` libraries via the following commands (taking Python 3.7 as an example):
```sh
sudo apt-get install libpython3.7
sudo apt-get install libpython3-dev
```
* For macOS, download a proper Python version from <https://www.python.org/> and install it. Note that macOS 10.x comes with python 2.7 installed, which is not supported, so you still need install Python from its official website.

For PIP, make sure that you have installed the latest version. To check and upgrade your PIP version, run the following command:
'''sh
python -m pip install --upgrade pip
'''

<!--## Check the special tips for Anaconda installation-->

<!--missing part-->


## Check if required external dependencies are installed

For Ubuntu and RHEL 8 systems, if you installed OpenVINO Runtime via the installer, APT, or YUM repository, and decided to [install OpenVINO Development Tools](installing-model-dev-tools.md), make sure that you <a href="openvino_docs_install_guides_installing_openvino_linux.html#install-external-dependencies">Install External Software Dependencies</a> first. 

For C++ developers with Windows systems, make sure that Microsoft Visual Studio 2019 with MSBuild and CMake 3.14 or higher (64-bit) are installed. While installing Microsoft Visual Studio 2019, make sure that you have selected **Desktop development with C++** in the **Workloads** tab. If not, launch the installer again to select that option. For more information on modifying the installation options for Microsoft Visual Studio, see its [official support page](https://docs.microsoft.com/en-us/visualstudio/install/modify-visual-studio?view=vs-2019).

## Check if environment variables are set correctly 

- If you have installed OpenVINO before, you probably have added `setupvars` to your `PATH /.bashrc` or Windows environment variables. After restarting your environment, you should see similar information as below:
```sh
[setupvars.sh] OpenVINO™ environment initialized
```
   - If you don't see the information, your PATH variable may be configured incorrectly. Check if you have written the correct <INSTALL_DIR> or tried to activate it in the right folder.
   - If you added it to a `.bashrc` file, make sure that the command is correctly written and the file is found in the `~/.bashrc` folder.

- For Python developers, if you installed OpenVINO using the installer previously, and now are installing OpenVINO with PIP, remove all the PATH settings and the lines with `setupvars` from `.bashrc`.

## Verify if OpenVINO is correctly installed

For Python developers, to verify if OpenVINO is correctly installed, use the following command:
```sh
python -c "from openvino.runtime import Core"
```
If OpenVINO was successfully installed, nothing will happen. If not, an error will be displayed. 

## Check if GPU drvier is installed

[Additional configurations](configurations-header.md) are required in order to use OpenVINO on different hardware.

To run inference on GPU, make sure that you have installed the correct GPU driver. To check that, see [additional configurations for GPU](configurations-for-intel-gpu.md).

## Firewall/Network Issues

Make sure that your firewall and network settings are set correctly. For example, consider configuring system-wide proxy settings and specifying options for using PIP behind the proxy: `pip install --proxy http://address:port --trusted-host pypi.org openvino`. 

For specific issues, see <a href="openvino_docs_get_started_guide_troubleshooting_issues.html#install-for-prc">Errors with Installing via PIP for PRC Users</a> and <a href="openvino_docs_get_started_guide_troubleshooting_issues.html#proxy-issues">proxy issues with installing OpenVINO on Linux from Docker</a>.