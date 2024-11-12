Run LLMs with OpenVINO GenAI Flavor on NPU
==========================================

.. meta::
   :description: Learn how to use the OpenVINO GenAI flavor to execute LLM models on NPU.

This guide will give you extra details on how to utilize NPU with the GenAI flavor.
:doc:`See the installation guide <../../get-started/install-openvino/install-openvino-genai>`
for information on how to start.

Prerequisites
#####################

Install required dependencies:

.. code-block:: console

   python -m venv npu-env
   npu-env\Scripts\activate
   pip install nncf==2.12 onnx==1.16.1 optimum-intel==1.19.0
   pip install --pre openvino openvino-tokenizers openvino-genai --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly

Export an LLM model via Hugging Face Optimum-Intel
##################################################

As **NPU supports only symmetrically-quantized 4-bit (INT4) models**, make sure to export
the model with the proper conversion and optimization settings.

You may export LLMs via Optimum-Intel, using one of two compression methods:
channel-wise quantization or group quantization. You do so by setting the ``--group-size``
parameter to ``-1`` or ``128``, respectively. See the following example, using a
chat-tuned TinyLlama model:
``



.. tab-set::

   .. tab-item:: Channel-wise quantization

      .. code-block:: console
         :name: channel-wise-quant

         optimum-cli export openvino -m TinyLlama/TinyLlama-1.1B-Chat-v1.0 --weight-format int4 --sym --ratio 1.0 --group_size -1

   .. tab-item:: Group quantization

      .. code-block:: console
         :name: group-quant

         optimum-cli export openvino -m TinyLlama/TinyLlama-1.1B-Chat-v1.0 --weight-format int4 --sym --group-size 128 --ratio 1.0 TinyLlama

**For models exceeding 1 billion parameters**, it is recommended to use **channel-wise
quantization** that is remarkably effective. You can also use group quantization,
however it might not be so efficient. You can try the suggested approach with
the llama-2-7b-chat-hf model:

.. code-block:: console

   optimum-cli export openvino -m meta-llama/Llama-2-7b-chat-hf --weight-format int4 --sym --group-size -1 --ratio 1.0 Llama-2-7b-chat-hf

.. important::

   To enable channel-wise quantization, the ``--group-size`` parameter must take value of ``-1`` (negative), not ``1``.

You can also try using 4-bit (INT4)
`GPTQ models <https://huggingface.co/models?other=gptq,4-bit&sort=trending>`__,
which do not require specifying quantization parameters:

.. code-block:: console

   optimum-cli export openvino -m TheBloke/Llama-2-7B-Chat-GPTQ


| Remember, NPU supports GenAI models quantized symmetrically to INT4.
| Below is a list of such models:

* meta-llama/Meta-Llama-3-8B-Instruct
* microsoft/Phi-3-mini-4k-instruct
* Qwen/Qwen2-7B
* mistralai/Mistral-7B-Instruct-v0.2
* openbmb/MiniCPM-1B-sft-bf16
* TinyLlama/TinyLlama-1.1B-Chat-v1.0
* TheBloke/Llama-2-7B-Chat-GPTQ
* Qwen/Qwen2-7B-Instruct-GPTQ-Int4


Run generation using OpenVINO GenAI
###################################

It is recommended to install the latest available
`driver <https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html>`__.

**Currently the NPU pipeline supports greedy decoding only, so you need to
add** ``do_sample=False`` **to the** ``generate()`` **method.**
Use the following code snippet to perform generation with OpenVINO GenAI API:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. code-block:: python
         :emphasize-lines: 4-5

         import openvino_genai as ov_genai
         model_path = "TinyLlama"
         pipe = ov_genai.LLMPipeline(model_path, "NPU")
         # Add 'do_sample=False' to 'generate()' method.
         print(pipe.generate("The Sun is yellow because", max_new_tokens=100, do_sample=False))

   .. tab-item:: C++
      :sync: cpp

      .. code-block:: cpp
         :emphasize-lines: 7-8, 10

         #include "openvino/genai/llm_pipeline.hpp"
         #include <iostream>

         int main(int argc, char* argv[]) {
            std::string model_path = "TinyLlama";
            ov::genai::GenerationConfig config;
            // Add 'do_sample=false' to 'generate()' method.
            config.do_sample=false;
            config.max_new_tokens=100;
            std::cout << pipe.generate("The Sun is yellow because", config);
         }


Additional configuration options
################################

Prompt and response length options
++++++++++++++++++++++++++++++++++

The LLM pipeline for NPUs leverages the static shape approach, optimizing execution performance,
while potentially introducing certain usage limitations. By default, the LLM pipeline supports
input prompts up to 1024 tokens in length. It also ensures that the generated response contains
at least 150 tokens, unless the generation encounters the end-of-sequence (EOS) token or the
user explicitly sets a lower length limit for the response.

You may configure both the 'maximum input prompt length' and 'minimum response length' using
the following parameters:

* ``MAX_PROMPT_LEN`` - defines the maximum number of tokens that the LLM pipeline can process
  for the input prompt (default: 1024),
* ``MIN_RESPONSE_LEN`` - defines the minimum number of tokens that the LLM pipeline will generate
  in its response (default: 150).

Use the following code snippet to change the default settings:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. code-block:: python

         pipeline_config = { "MAX_PROMPT_LEN": 1024, "MIN_RESPONSE_LEN": 512 }
         pipe = ov_genai.LLMPipeline(model_path, "NPU", pipeline_config)

   .. tab-item:: C++
      :sync: cpp

      .. code-block:: cpp

         ov::AnyMap pipeline_config = { { "MAX_PROMPT_LEN",  1024 }, { "MIN_RESPONSE_LEN", 512 } };
         ov::genai::LLMPipeline pipe(model_path, "NPU", pipeline_config);

Cache compiled models
+++++++++++++++++++++

Specify the ``NPUW_CACHE_DIR`` option in ``pipeline_config`` for NPU pipeline to
cache compiled models. Using the code snippet below will help shorten
initialization time of the next pipeline runs:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. code-block:: python

         pipeline_config = { "NPUW_CACHE_DIR": ".npucache" }
         pipe = ov_genai.LLMPipeline(model_path, "NPU", pipeline_config)

   .. tab-item:: C++
      :sync: cpp

      .. code-block:: cpp

         ov::AnyMap pipeline_config = { { "NPUW_CACHE_DIR",  ".npucache" } };
         ov::genai::LLMPipeline pipe(model_path, "NPU", pipeline_config);


Disable memory allocation
+++++++++++++++++++++++++

In case of execution failures, either silent or with errors, try to update the NPU driver to
`newer than 31.0.100.3053 <https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html>`__.
If the update is not possible, set the ``DISABLE_OPENVINO_GENAI_NPU_L0``
environment variable to disable NPU memory allocation, which might be supported
only on newer drivers for Lunar Lake (LNL) processors.

Set the environment variable in a terminal:

.. tab-set::

   .. tab-item:: Linux
      :sync: linux

      .. code-block:: console

         export DISABLE_OPENVINO_GENAI_NPU_L0=1

   .. tab-item:: Windows
      :sync: win

      .. code-block:: console

         set DISABLE_OPENVINO_GENAI_NPU_L0=1


Performance modes
+++++++++++++++++++++

You can configure the NPU pipeline with the ``GENERATE_HINT`` option to switch
between two different performance modes:

* ``FAST_COMPILE`` (default) - enables fast compilation at the expense of performance,
* ``BEST_PERF`` - ensures best possible performance at lower compilation speed.

Use the following code snippet:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. code-block:: python

         pipeline_config = { "GENERATE_HINT": "BEST_PERF" }
         pipe = ov_genai.LLMPipeline(model_path, "NPU", pipeline_config)

   .. tab-item:: C++
      :sync: cpp

      .. code-block:: cpp

         ov::AnyMap pipeline_config = { { "GENERATE_HINT",  "BEST_PERF" } };
         ov::genai::LLMPipeline pipe(model_path, "NPU", pipeline_config);


Additional Resources
####################

* :doc:`NPU Device <../../openvino-workflow/running-inference/inference-devices-and-modes/npu-device>`
* `OpenVINO GenAI Repo <https://github.com/openvinotoolkit/openvino.genai>`__
* `Neural Network Compression Framework <https://github.com/openvinotoolkit/nncf>`__
