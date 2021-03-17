## ExperimentalDetectronPriorGridGenerator <a name="ExperimentalDetectronPriorGridGenerator"></a> {#openvino_docs_ops_detection_ExperimentalDetectronPriorGridGenerator_6}

**Versioned name**: *ExperimentalDetectronPriorGridGenerator-6*

**Category**: Object detection

**Short description**: An operation *ExperimentalDetectronPriorGridGenerator* operation generates prior grids of specified sizes.

**Attributes**:

* *flatten*

    * **Description**: *flatten* attribute specifies whether the output tensor should be 2D or 4D.
    * **Range of values**:
      * `true` - the output tensor should be 2D tensor
      * `false` - the output tensor should be 4D tensor
    * **Type**: boolean
    * **Default value**: true
    * **Required**: *no*

* *h*

    * **Description**: *h* attribute specifies number of cells of the generated grid with respect to height.
    * **Range of values**: non-negative integer number
    * **Type**: int
    * **Default value**: 0
    * **Required**: *no*
    
* *w*

    * **Description**: *w* attribute specifies number of cells of the generated grid with respect to width.
    * **Range of values**: non-negative integer number
    * **Type**: int
    * **Default value**: 0
    * **Required**: *no*

* *stride_x*

    * **Description**: *stride_x* attribute specifies the step of generated grid with respect to x coordinate.
    * **Range of values**: non-negative float number
    * **Type**: float
    * **Default value**: 0.0
    * **Required**: *no*
    
* *stride_y*

    * **Description**: *stride_y* attribute specifies the step of generated grid with respect to y coordinate.
    * **Range of values**: non-negative float number
    * **Type**: float
    * **Default value**: 0.0
    * **Required**: *no*

**Inputs**

* **1**: A tensor of type *T* with priors. Rank must be equal to 2 and The last dimension must be equal to 4: `[number_of_priors, 4]`. **Required.**

* **2**: A 4D tensor of type *T* with input feature map. **Required.**

* **3**: A 4D tensor of type *T* with input image. The number of channels of both feature map and input image tensors must match. **Required.**

**Outputs**

* **1**: A tensor of type *T* with priors grid with shape `[featmap_height * featmap_width * number_of_priors, 4]` if flatten is `true` or `[featmap_height, featmap_width, number_of_priors, 4]` otherwise, where `featmap_height` and `featmap_width` are spatial dimensions values from second input.
The output tensor is filled with -1s for output tensor elements if the total number of selected boxes is less than the output tensor size.

**Types**

* *T*: any supported floating point type.

**Detailed description**: 

Operation computes prior grids by following:

    for (int ih = 0; ih < featmap_height; ++ih)
        for (int iw = 0; iw < featmap_width; ++iw)
            for (int s = 0; s < number_of_priors; ++s)
                data[0] = priors[4 * s + 0] + stride_x * (iw + 0.5)
                data[1] = priors[4 * s + 1] + stride_y * (ih + 0.5)
                data[2] = priors[4 * s + 2] + stride_x * (iw + 0.5)
                data[3] = priors[4 * s + 3] + stride_y * (ih + 0.5)
                data += 4


**Example**

```xml
<layer ... type="ExperimentalDetectronPriorGridGenerator" version="opset6">
    <data flatten="true" h="0" stride_x="32.0" stride_y="32.0" w="0"/>
    <input>
        <port id="0">
            <dim>3</dim>
            <dim>4</dim>
        </port>
        <port id="1">
            <dim>1</dim>
            <dim>256</dim>
            <dim>25</dim>
            <dim>42</dim>
        </port>
        <port id="2">
            <dim>1</dim>
            <dim>3</dim>
            <dim>800</dim>
            <dim>1344</dim>
        </port>
    </input>
    <output>
        <port id="3" precision="FP32">
            <dim>3150</dim>
            <dim>4</dim>
        </port>
    </output>
</layer>
```
