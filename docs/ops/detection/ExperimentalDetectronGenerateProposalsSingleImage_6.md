## ExperimentalDetectronGenerateProposalsSingleImage <a name="ExperimentalDetectronGenerateProposalsSingleImage"></a> {#openvino_docs_ops_detection_ExperimentalDetectronGenerateProposalsSingleImage_6}

**Versioned name**: *ExperimentalDetectronGenerateProposalsSingleImage-6*

**Category**: Object detection

**Short description**: An operation *ExperimentalDetectronGenerateProposalsSingleImage* computes ROIs and ROI's scores based on input data.

**Detailed description**:

   Transpose and reshape predicted bbox transformations to get them into the same order as the anchors:
     - bbox deltas will be (4 * A, H, W) format from convolution output
     - transpose to (H, W, 4 * A)
     - reshape to (H * W * A, 4) where rows are ordered by (H, W, A) in slowest to fastest order to match the enumerated anchors
       
   Same story for the scores:
     - Scores are (A, H, W) format from convolution output
     - Transpose to (H, W, A)
     - Reshape to (H * W * A, 1) where rows are ordered by (H, W, A) to match the order of anchors and bbox_deltas

   Transform anchors into proposals and clip proposals to image.
   Remove predicted boxes with either height or width < *min_size*, sort all `(proposal, score)` pairs by score from highest to lowest.
   Take top *pre_nms_count*.
   
   Apply *nms_threshold*.
   Take *post_nms_count*.
   Return the top proposals (-> ROIs top).
       
**Attributes**:

* *min_size*

    * **Description**: *min_size* attribute specifies minimum box width & height.
    * **Range of values**: non-negative floating point number
    * **Type**: float
    * **Default value**: 0.0
    * **Required**: *yes*

* *nms_threshold*

    * **Description**: *nms_threshold* attribute specifies threshold to be used in the NMS stage.
    * **Range of values**: non-negative floating point number
    * **Type**: float
    * **Default value**: 0.7
    * **Required**: *yes*

* *post_nms_count*

    * **Description**: *post_nms_count* attribute specifies number of top-n proposals after NMS.
    * **Range of values**: non-negative integer number
    * **Type**: int
    * **Default value**: 1000
    * **Required**: *yes*

* *pre_nms_count*

    * **Description**: *pre_nms_count* attribute specifies number of top-n proposals before NMS.
    * **Range of values**: non-negative integer number
    * **Type**: int
    * **Default value**: 1000
    * **Required**: *yes*

**Inputs**

* **1**: A 1D tensor of type *T* with shape `[3]` with input data. **Required.**

* **2**: A 2D tensor of type *T* with input anchors. The second dimension of 'input_anchors' should be 4. **Required.**

* **3**: A 3D tensor of type *T* with input deltas. Height and width for third and fourth inputs must be equal. **Required.** 

* **4**: A 3D tensor of type *T* with input scores. **Required.**

**Outputs**

* **1**: A 2D tensor of type *T* with shape `[post_nms_count, 4]` describing ROIs.

* **2**: A 1D tensor of type *T* with shape `[post_nms_count]` describing ROIs scores.

**Types**

* *T*: any supported numeric type.

**Example**

```xml
<layer ... type="ExperimentalDetectronGenerateProposalsSingleImage" version="opset6">
    <data min_size="0.0" nms_threshold="0.699999988079071" post_nms_count="1000" pre_nms_count="1000"/>
    <input>
        <port id="0">
            <dim>3</dim>
        </port>
        <port id="1">
            <dim>12600</dim>
            <dim>4</dim>
        </port>
        <port id="2">
            <dim>12</dim>
            <dim>50</dim>
            <dim>84</dim>
        </port>
        <port id="3">
            <dim>3</dim>
            <dim>50</dim>
            <dim>84</dim>
        </port>
    </input>
    <output>
        <port id="4" precision="FP32">
            <dim>1000</dim>
            <dim>4</dim>
        </port>
        <port id="5" precision="FP32">
            <dim>1000</dim>
        </port>
    </output>
</layer>
```
