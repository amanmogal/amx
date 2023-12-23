Stateful models and State API {#openvino_docs_OV_UG_stateful_models_intro}
==============================

@sphinxdirective

.. toctree::
    :maxdepth: 1
    :hidden:

    openvino_docs_OV_UG_ways_to_get_stateful_model

@endsphinxdirective

## What is Stateful model?

 Several use cases require processing of data sequences. When length of a sequence is known and small enough, 
 we can process it with RNN like models that contain a cycle inside. But in some cases, like online speech recognition of time series 
 forecasting, length of data sequence is unknown. Then data can be divided in small portions and processed step-by-step. But dependency 
 between data portions should be addressed. For that, models save some data between inferences - state. When one dependent sequence is over,
 state should be reset to initial value and new sequence can be started.
 
Deep learning frameworks provide a dedicated API to build models with state. For example, Keras has special option for RNNs `stateful` that turns on saving state 
 between inferences. Kaldi contains special specifier `Offset` to define time offset in a model. 
 
 OpenVINO also contains special API to simplify work with models with states. State is automatically saved between inferences, 
 and there is a way to reset state when needed. You can also read state or set it to some new value between inferences.
 
## OpenVINO State Representation

 OpenVINO contains a special abstraction `Variable` to represent a state in a model. There are two operations to work with the state: 
* `Assign` to save value to the state
* `ReadValue` to read value saved on previous iteration

![state_model_example](./img/state_model_example.png)

The left side of the picture shows the usual inputs and outputs to the model: Parameter/Result operations. They have no connection with each other and in order to copy data from output to input users need to put extra effort writing and maintaining additional code.
In addition, this may impose additional overhead due to data representation conversion.

Having operations such as ReadValue and Assign allows users to replace the looped Parameter/Result pairs of operations and shift the work of copying data to OpenVINO. After the replacement, the OpenVINO model no longer contains inputs and outputs with such names, all internal work on data copying is hidden from the user, but data from the intermediate inference can always be retrieved using State API methods.

In some cases, users need to set an initial value for State, or it may be necessary to reset the value of State at a certain inference to the initial value. For such situations, an initializing subgraph for the ReadValue operation and a special "reset" method are provided.

You can find more details on these operations in [ReadValue specification](../ops/infrastructure/ReadValue_3.md) and 
[Assign specification](../ops/infrastructure/Assign_3.md).

## How to get the OpenVINO Model with states

* [Convert Kaldi model to IR via Model Optimizer.](../MO_DG/prepare_model/convert_model/kaldi_specific)
   If the original Kaldi model contains RNN-like operations with `stateful` option, then after ModelOptimizer conversion,
   the resulting OpenVINO model will also contain states.

* [Apply LowLatency2 transformation.](./ways_to_get_stateful_model.md#)
   If a model contains a loop that runs over some sequence of input data,
   the LowLatency2 transformation can be applied to get model with states.
   Note: there are some [specific limitations]() to use the transformation.

* [Apply MakeStateful transformation.](./ways_to_get_stateful_model.md)
   If after conversion from original model to OpenVINO representation, the resulting model contains Parameter and Result operations,
   which pairwise have the same shape and element type, the MakeStateful transformation can be applied to get model with states.

* [Create the model via OpenVINO API.](./ways_to_get_stateful_model.md)
   For testing purposes or for some specific cases, when the ways to get OpenVINO model with states described above are not enough for your purposes,
   you can use OpenVINO API and create `ov::opset8::ReadValue` and `ov::opset8::Assign` operations directly.

## OpenVINO State API

OpenVINO runtime has the `ov::InferRequest::query_state` method  to get the list of states from a model and `ov::VariableState` class to operate with states. 
 Below you can find brief description of methods and the example of how to use this interface.

 `ov::InferRequest` methods:
 
 * `std::vector<VariableState> query_state();`
   allows to get all available stats for the given inference request.
 * `void reset_state()`
   allows to reset all States to their default values.

 `ov::VariableState` methods:

 * `std::string get_name() const`
   returns name(variable_id) of the according State(Variable)
 * `void reset()`
   reset state to the default value
 * `void set_state(const Tensor& state)`
   set new value for State
 * `Tensor get_state() const`
   returns current value of State

## Example of Stateful model Inference

The example below demonstrates inference of three independent sequences of data. State should be reset between these sequences.

One infer request and one thread will be used in this example. Using several threads is possible if you have several independent sequences. Then each sequence can be processed in its own infer request. Inference of one sequence in several infer requests is not recommended. In one infer request state will be saved automatically between inferences, but 
if the first step is done in one infer request and the second in another, state should be set in new infer request manually (using `ov::VariableState::set_state` method).

@sphinxdirective

.. tab:: C++

      .. doxygensnippet:: docs/snippets/ov_model_state_intro.cpp
         :language: cpp
         :fragment: [ov:state_api_usage]

@endsphinxdirective

You can find more powerful examples demonstrating how to work with models with states in speech sample and demo. 
Descriptions can be found in [Samples Overview](./Samples_Overview.md)
