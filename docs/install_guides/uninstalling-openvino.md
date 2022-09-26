# Uninstalling the Intel® Distribution of OpenVINO™ Toolkit {#openvino_docs_install_guides_uninstalling_openvino}

> **NOTE**: Uninstallation procedures remove all Intel® Distribution of OpenVINO™ Toolkit component files but don't affect user files in the installation directory.

## Uninstalling OpenVINO 2022.1

### Uninstalling Using the Original Installation Package

@sphinxdirective
.. tab:: Windows

  Use initial bootstrapper file ``w_openvino_toolkit_p_<version>.exe`` to select product for uninstallation. Follow the wizard instructions. Select **Remove** option when presented. If you have more product versions installed, you can select one from a drop-down menu in GUI.

  .. image:: _static/images/openvino-uninstall-dropdown-win.png
    :width: 500px
    :align: center
    
.. tab:: Linux

  If you want to use graphical user interface (GUI) installation wizard, run the script without any parameters:
  
  .. code-block:: sh
  
    ./l_openvino_toolkit_p_<version>.sh

  Follow the wizard instructions.

  Otherwise, you can add parameters `-a` for additional arguments and `--cli` to run installation in command line (CLI):
  
  .. code-block:: sh
    
    ./l_openvino_toolkit_p_<version>.sh -a --cli

  Follow the wizard. Select **Remove** option when presented. If you have more product versions installed, you can select one from a drop-down menu in GUI and from a list in CLI.

  .. image:: _static/images/openvino-uninstall-dropdown-linux.png
    :width: 500px
    :align: center

.. tab:: macOS

  Use initial bootstrapper file ``m_openvino_toolkit_p_<version>.dmg`` to select product for uninstallation. Mount the file and double-click ``bootstrapper.app``. Follow the wizard instructions. Select **Remove** option when presented. If you have more product versions installed, you can select one from a drop-down menu in GUI.

  .. image:: _static/images/openvino-uninstall-dropdown-macos.png
    :width: 500px
    :align: center

@endsphinxdirective

### Uninstalling Using the Intel® Software Installer

@sphinxdirective
.. tab:: Windows

  1. Choose the **Apps & Features** option from the Windows Settings app.
  2. From the list of installed applications, select the Intel® Distribution of OpenVINO™ Toolkit and click **Uninstall**.
  3. Follow the uninstallation wizard instructions.

  Alternatively, follow the steps:
  
  1. Go to installation directory.
  2. In ``OpenVINO`` directory find ``Installer`` folder and open it.
  3. Double-click on ``installer.exe`` and you will be presented with dialog box shown below.

.. tab:: Linux

  1. Run the installer file from the user mode installation directory:
   
  .. code-block:: sh
  
    /home/<user>/intel/openvino_installer/installer

  or in a case of administrative installation:

  .. code-block:: sh

    /opt/intel/openvino_installer/installer

  2. Follow the uninstallation wizard instructions.
  
.. tab:: macOS

  1. Open the installer file from the installation directory:
   
  .. code-block:: sh
  
    open /opt/intel/openvino_installer/installer.app

  2. Follow the uninstallation wizard instructions.

Finally, complete the procedure with clicking on **Modify** and then selecting **Uninstall** option:

.. tab:: Windows
  
  .. image:: _static/images/openvino-uninstall-win.png
    :width: 500px
    :align: center

.. tab:: Linux
 
  .. image:: _static/images/openvino-uninstall-linux.png
    :width: 500px
    :align: center
    
  if GUI is not available, installer also could be run in a CLI mode:

  .. image:: _static/images/openvino-uninstall-cli.png
     :width: 500px
     :align: center
  
.. tab:: macOS

  .. image:: _static/images/openvino-uninstall-macos.png
    :width: 500px
    :align: center

@endsphinxdirective

## Uninstalling OpenVINO 2022.1.1

If you have installed OpenVINO Runtime 2022.1.1 from archive files, you can uninstall it by deleting the archive files and the extracted folders.

@sphinxdirective
.. tab:: Windows

  If you have created the symbolic link, remove the link first.

  Use either of the following methods to delete the files:

  * Use Windows Explorer to remove the files.
  * Open a Command Prompt and run:
    
    .. code-block:: sh
  
      rmdir /s <extracted_folder>
      del <path_to_archive>

    
.. tab:: Linux & macOS
  
  If you have created the symbolic link, remove the link first:

  .. code-block:: sh
  
   `rm /home/<USER>/intel/openvino_2022`

  To delete the files:

  .. code-block:: sh
  
    rm -r <extracted_folder> && rm <path_to_archive>

@endsphinxdirective