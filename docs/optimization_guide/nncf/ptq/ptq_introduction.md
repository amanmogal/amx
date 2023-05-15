# Post-training Quantization with NNCF {#nncf_ptq_introduction}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   basic_quantization_flow
   quantization_w_accuracy_control


Neural Network Compression Framework (NNCF) provides a post-training quantization API available in Python that is aimed at reusing the code for model training or validation that is usually available with the model in the source framework, for example, PyTorch or TensroFlow. The NNCF API is cross-framework and currently supports models in the following frameworks: OpenVINO, PyTorch, TensorFlow 2.x, and ONNX. Currently, post-training quantization for models in OpenVINO Intermediate Representation is the most mature in terms of supported methods and models coverage. 

This API has two main capabilities to apply 8-bit post-training quantization:

* :doc:`Basic quantization <basic_quantization_flow>` - the simplest quantization flow that allows applying 8-bit integer quantization to the model. A representative calibration dataset is only needed in this case.
* :doc:`Quantization with accuracy control <quantization_w_accuracy_control>` - the most advanced quantization flow that allows applying 8-bit quantization to the model with accuracy control. Calibration and validation datasets, and a validation function to calculate the accuracy metric are needed in this case.

Additional Resources
####################

* :doc:`Optimizing Models at Training Time <tmo_introduction>`
* `NNCF GitHub <https://github.com/openvinotoolkit/nncf>`__

@endsphinxdirective
