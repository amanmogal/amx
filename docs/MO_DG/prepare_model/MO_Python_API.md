## Model Optimizer Python API {#openvino_docs_MO_DG_Python_API}

Model Optimizer has Python API for model conversion, which is represented by `convert_model()` method in openvino.tools.mo namespace.
  `convert_model()` has all the functionality available from the command-line tool, plus the ability to pass Python objects (models, extensions) to `convert_model()` directly from memory without leaving the training environment (Jupyter Notebook or training scripts).
  `convert_model()` returns an openvino.runtime.Model object which can be compiled and inferred or serialized to IR.

```sh
from openvino.tools.mo import convert_model

ov_model = convert_model("resnet.onnx")
```

MO Python API allows the conversion of PyTorch models.

```sh
import torchvision

model = torchvision.models.resnet50(pretrained=True)
ov_model = convert_model(model, input_shape=[1,3,100,100])
```

Following types are supported as input model for `convert_model()`:

* PyTorch - `torch.nn.Module`, `torch.jit.ScriptModule`, `torch.jit.ScriptFunction`. Refer to the [Converting a PyTorch Model](@ref openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_PyTorch) article for more details.
* TensorFlow/ TensorFlow2 / Keras - `tf.keras.Model`, `tf.keras.layers.Layer`, `tf.compat.v1.GraphDef`, `tf.Module`, `tf.compat.v1.wrap_function`, `tf.compat.v1.session`, `tf.train.checkpoint`, `tf.python.training.tracking.base.Trackable`(with limitations). Refer to the [Converting a TensorFlow Model](@ref openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow) article for more details.

`convert_model()` accepts all parameters available in MO command line tool. Parameters can be specified by Python classes or string analogs identically to command line tool.
Example 1:

```sh
from openvino.runtime import PartialShape, Layout

ov_model = convert_model(model, input_shape=PartialShape([1,3,100,100]), mean_values=[127, 127, 127], layout=Layout("NCHW"))
```

Example 2:

```sh
ov_model = convert_model(model, input_shape="[1,3,100,100]", mean_values="[127,127,127]", layout="NCHW")
```

Command-line flags like `--compress_to_fp16` can be set in Python API by providing a boolean value (`True` or `False`).

```sh
ov_model = convert_model(model, compress_to_fp16=True)
```

`input` parameter can be set by `tuple` with name, shape and type. Input name of type string is required in such tuple. Shape and type are optional. 
Shape can be a `list` or `tuple` of dimensions (`int` or `openvino.runtime.Dimension`) or `openvino.runtime.PartialShape` or `openvino.runtime.Shape`. Type can be of numpy type or `openvino.runtime.Type`.

```sh
ov_model = convert_model(model, input=("input_name", [3], np.float32))
```

For complex cases when value needs to be set in `input` parameter `InputCutInfo` class can be used. `InputCutInfo` accepts four parameters: `name`, `shape`, `type`, `value`.
  `InputCutInfo("input_name", [3], np.float32, [0.5, 2.1, 3.4])` is equivalent of `InputCutInfo(name="input_name", shape=[3], type=np.float32, value=[0.5, 2.1, 3.4])`.
Supported types for `InputCutInfo`:
- name: `string`.
- shape: `list` or `tuple` of dimensions (`int` or `openvino.runtime.Dimension`), `openvino.runtime.PartialShape`,` openvino.runtime.Shape`.
- type: `numpy type`, `openvino.runtime.Type`.
- value: `numpy.ndarray`, `list` of numeric values, `bool`.

```sh
from openvino.tools.mo import convert_model, InputCutInfo

ov_model = convert_model(model, input=InputCutInfo("input_name", [3], np.float32, [0.5, 2.1, 3.4]))
```

`layout`, `source_layout` and `dest_layout` accept `openvino.runtime.Layout` object or `string`. 

```sh
from openvino.runtime import Layout
from openvino.tools.mo import convert_model

ov_model = convert_model(model, source_layout=Layout("NCHW"))
```

To set both source and destination layouts in `layout` parameter `LayoutMap` class can be used. `LayoutMap` accepts two parameters: `source_layout`, `target_layout`.
`LayoutMap("NCHW", "NHWC")` is equivalent of `LayoutMap(source_layout="NCHW", target_layout="NHWC")`.

```sh
from openvino.tools.mo import convert_model, LayoutMap

ov_model = convert_model(model, layout=LayoutMap("NCHW", "NHWC"))
```
