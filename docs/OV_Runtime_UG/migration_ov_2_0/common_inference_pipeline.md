# Inference Pipeline {#openvino_2_0_inference_pipeline}

Usually to infer models with OpenVINO™ Runtime, you need to do the following steps in the application pipeline:
- 1. Create Core object
 - 1.1. (Optional) Load extensions
- 2. Read model from the disk
 - 2.1. (Optional) Model preprocessing
- 3. Load the model to the device
- 4. Create an inference request
- 5. Fill input tensors with data
- 6. Start inference
- 7. Process the inference results

The following code shows how to change the application code in each step to migrate to OpenVINO™ Runtime 2.0.

## 1. Create Core

Inference Engine API:

@snippet docs/snippets/ie_common.cpp ie:create_core

OpenVINO™ Runtime API 2.0:

@snippet docs/snippets/ov_common.cpp ov_api_2_0:create_core

### 1.1 (Optional) Load extensions

To load model with custom operation, you need to add extensions for these operations. We highly recommend to use [OpenVINO Extensibility API](../../Extensibility_UG/Intro.md) to write extensions, but if you already have old extensions you can load it to new OpenVINO™ Runtime:

Inference Engine API:

@snippet docs/snippets/ie_common.cpp ie:load_old_extension

OpenVINO™ Runtime API 2.0:

@snippet docs/snippets/ov_common.cpp ov_api_2_0:load_old_extension

## 2. Read model from the disk

Inference Engine API:

@snippet docs/snippets/ie_common.cpp ie:read_model

OpenVINO™ Runtime API 2.0:

@snippet docs/snippets/ov_common.cpp ov_api_2_0:read_model

Read model has the same structure as in the example from [Model Creation](./graph_construction.md) migration guide.

Note, you can combine read and compile model stages into a single call `ov::Core::compile_model(filename, devicename)`.

### 2.1 (Optional) Model preprocessing

When application's input data doesn't perfectly match with model's input format, preprocessing steps may need to be added.
See detailed guide [how to migrate preprocessing in OpenVINO Runtime API 2.0](./preprocessing.md)

## 3. Load the Model to the Device

Inference Engine API:

@snippet docs/snippets/ie_common.cpp ie:compile_model

OpenVINO™ Runtime API 2.0:

@snippet docs/snippets/ov_common.cpp ov_api_2_0:compile_model

If you need to configure OpenVINO Runtime devices with additional configuration parameters, please, refer to the migration [Configure devices](./configure_devices.md) guide.

## 4. Create an Inference Request

Inference Engine API:

@snippet docs/snippets/ie_common.cpp ie:create_infer_request

OpenVINO™ Runtime API 2.0:

@snippet docs/snippets/ov_common.cpp ov_api_2_0:create_infer_request

## 5. Fill input tensors

Inference Engine API fills inputs as `I32` precision (**not** aligned with the original model):

@sphinxtabset

@sphinxtab{IR v10}

@snippet docs/snippets/ie_common.cpp ie:get_input_tensor

@endsphinxtab

@sphinxtab{IR v11}

@snippet docs/snippets/ie_common.cpp ie:get_input_tensor

@endsphinxtab

@sphinxtab{ONNX}

@snippet docs/snippets/ie_common.cpp ie:get_input_tensor

@endsphinxtab

@sphinxtab{Model created in code}

@snippet docs/snippets/ie_common.cpp ie:get_input_tensor

@endsphinxtab

@endsphinxtabset

OpenVINO™ Runtime API 2.0 fills inputs as `I64` precision (aligned with the original model):

@sphinxtabset

@sphinxtab{IR v10}

@snippet docs/snippets/ov_common.cpp ov_api_2_0:get_input_tensor_v10

@endsphinxtab

@sphinxtab{IR v11}

@snippet docs/snippets/ov_common.cpp ov_api_2_0:get_input_tensor_aligned

@endsphinxtab

@sphinxtab{ONNX}

@snippet docs/snippets/ov_common.cpp ov_api_2_0:get_input_tensor_aligned

@endsphinxtab

@sphinxtab{Model created in code}

@snippet docs/snippets/ov_common.cpp ov_api_2_0:get_input_tensor_aligned

@endsphinxtab

@endsphinxtabset

## 6. Start Inference

Inference Engine API:

@sphinxtabset

@sphinxtab{Sync}

@snippet docs/snippets/ie_common.cpp ie:inference

@endsphinxtab

@sphinxtab{Async}

@snippet docs/snippets/ie_common.cpp ie:start_async_and_wait

@endsphinxtab

@endsphinxtabset

OpenVINO™ Runtime API 2.0:

@sphinxtabset

@sphinxtab{Sync}

@snippet docs/snippets/ov_common.cpp ov_api_2_0:inference

@endsphinxtab

@sphinxtab{Async}

@snippet docs/snippets/ov_common.cpp ov_api_2_0:start_async_and_wait

@endsphinxtab

@endsphinxtabset

## 7. Process the Inference Results

Inference Engine API processes outputs as `I32` precision (**not** aligned with the original model):

@sphinxtabset

@sphinxtab{IR v10}

@snippet docs/snippets/ov_common.cpp ov_api_2_0:inference

@endsphinxtab

@sphinxtab{IR v11}

@snippet docs/snippets/ie_common.cpp ie:get_output_tensor

@endsphinxtab

@sphinxtab{ONNX}

@snippet docs/snippets/ie_common.cpp ie:get_output_tensor

@endsphinxtab

@sphinxtab{Model created in code}

@snippet docs/snippets/ie_common.cpp ie:get_output_tensor

@endsphinxtab

@endsphinxtabset

OpenVINO™ Runtime API 2.0 processes outputs:
- For IR v10 as `I32` precision (**not** aligned with the original model) to match **old** behavior
- For IR v11, ONNX, ov::Model, Paddle as `I64` precision (aligned with the original model) to match **new** behavior

@sphinxtabset

@sphinxtab{IR v10}

@snippet docs/snippets/ov_common.cpp ov_api_2_0:get_output_tensor_v10

@endsphinxtab

@sphinxtab{IR v11}

@snippet docs/snippets/ov_common.cpp ov_api_2_0:get_output_tensor_aligned

@endsphinxtab

@sphinxtab{ONNX}

@snippet docs/snippets/ov_common.cpp ov_api_2_0:get_output_tensor_aligned

@endsphinxtab

@sphinxtab{Model created in code}

@snippet docs/snippets/ov_common.cpp ov_api_2_0:get_output_tensor_aligned

@endsphinxtab

@endsphinxtabset
