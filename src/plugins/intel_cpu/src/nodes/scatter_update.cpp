// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "scatter_update.h"

#include "common/cpu_memcpy.h"
#include "dnnl_extension_utils.h"
#include "onednn/dnnl.h"
#include "openvino/core/parallel.hpp"
#include "openvino/opsets/opset3.hpp"
#include "openvino/opsets/opset4.hpp"
#include "openvino/opsets/opset12.hpp"
#include "utils/plain_tensor.hpp"

#include <algorithm>
#include <string>
#include <vector>

using namespace dnnl;

namespace ov {
namespace intel_cpu {
namespace node {

bool ScatterUpdate::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        auto scatterElemUpd3 = ov::as_type_ptr<const ov::opset3::ScatterElementsUpdate>(op);
        auto scatterElemUpd12 = ov::as_type_ptr<const ov::opset12::ScatterElementsUpdate>(op);
        auto scatterUpd = ov::as_type_ptr<const ov::opset3::ScatterUpdate>(op);
        auto scatterNdUpd = ov::as_type_ptr<const ov::opset4::ScatterNDUpdate>(op);
        if (!scatterElemUpd3 && !scatterElemUpd12 && !scatterUpd && !scatterNdUpd) {
            const std::string opType = op->get_type_name();
            errorMessage = std::string("Type ") + opType + " is not supported.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

bool ScatterUpdate::isExecutable() const {
    return !isInputTensorAtPortEmpty(DATA_ID);
}

ScatterUpdate::ScatterUpdate(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context)
        : Node(op, context, NgraphShapeInferFactory(op, EMPTY_PORT_MASK)),
          dataSize(0lu), indicesSize(0lu), axisSize(0lu),
          dataPrec(ov::element::undefined),
          indicesPrec(ov::element::undefined),
          axisPrec(ov::element::undefined) {
    std::string errorMessage;
    if (isSupportedOperation(op, errorMessage)) {
        errorPrefix = std::string(op->get_type_name()) + " node with name '" + getName() + "'";
    } else {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    const auto node = std::dynamic_pointer_cast<const ov::op::v12::ScatterElementsUpdate>(op);
    if (node) {
        reduction_type = node->get_reduction();
        use_init_val = node->get_use_init_val();
    } else {
        reduction_type = ov::op::v12::ScatterElementsUpdate::Reduction::NONE;
    }
}

void ScatterUpdate::getSupportedDescriptors() {
    if ((getParentEdges().size() != 3) && (getParentEdges().size() != 4))
        OPENVINO_THROW(errorPrefix, " has incorrect number of input edges");
    if (getChildEdges().empty())
        OPENVINO_THROW(errorPrefix, " has incorrect number of output edges");

    if (getInputShapeAtPort(DATA_ID).getRank() < 1 ||
        getInputShapeAtPort(INDICES_ID).getRank() < 1 ||
            getInputShapeAtPort(UPDATE_ID).getRank() < 1) {
        OPENVINO_THROW(errorPrefix, " do not support scalar input");
    }

    Type scatterUpdateType = getType();
    if (scatterUpdateType == Type::ScatterUpdate) {
        scatterUpdateMode = ScatterUpdateMode::ScatterUpdate;
        axisRelaxed = true;
    } else if (scatterUpdateType == Type::ScatterElementsUpdate) {
        scatterUpdateMode = ScatterUpdateMode::ScatterElementsUpdate;
        axisRelaxed = true;
    } else if (scatterUpdateType == Type::ScatterNDUpdate) {
        scatterUpdateMode = ScatterUpdateMode::ScatterNDUpdate;
        axisRelaxed = false;
    } else {
        OPENVINO_THROW(errorPrefix, " is not supported");
    }
}

void ScatterUpdate::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    const auto& srcDataDim = getInputShapeAtPort(DATA_ID).getDims();
    const auto& indicesDim = getInputShapeAtPort(INDICES_ID).getDims();
    const auto& updateDim =  getInputShapeAtPort(UPDATE_ID).getDims();
    const auto& dstDataDim = getOutputShapeAtPort(0).getDims();

    size_t srcRank = srcDataDim.size();
    size_t indicesRank = indicesDim.size();
    size_t updateRank = updateDim.size();
    size_t dstRank = dstDataDim.size();

    // common check
    if (srcRank != dstRank) {
        OPENVINO_THROW(errorPrefix, " should have same rank for input and output tensor");
    } else {
        for (size_t r = 0; r < srcRank; r++) {
            if (!dimsEqualWeak(srcDataDim[r], dstDataDim[r])) {
                OPENVINO_THROW(errorPrefix,
                               " should have same shape for input and output tensor. The input shape is ",
                               srcDataDim[r],
                               ", while output shape is ",
                               dstDataDim[r],
                               " for ",
                               r,
                               "th dimension");
            }
        }
    }
    // specific check
    switch (scatterUpdateMode) {
        case ScatterUpdateMode::ScatterUpdate: {
            if (updateRank != (srcRank + indicesRank - 1)) {
                OPENVINO_THROW(errorPrefix,
                               " do not have matched tensor rank relationship for input, indices and update");
            }
            break;
        }
        case ScatterUpdateMode::ScatterNDUpdate: {
            if (indicesDim[indicesRank - 1] != Shape::UNDEFINED_DIM) {
                size_t k = indicesDim[indicesRank - 1];
                if (k > srcRank) {
                    OPENVINO_THROW(errorPrefix,
                                   "' do not have an correct indices' last dimension value, ",
                                   "which should be smaller than or equal to input tensor rank");
                }

                size_t tupleRank = indicesRank - 1;
                VectorDims expectUpdateShape(tupleRank + srcRank - k, 0);
                int updateAxisIter = 0;
                for (size_t ri = 0; ri < tupleRank; ri++) {
                    expectUpdateShape[updateAxisIter] = indicesDim[ri];
                    updateAxisIter++;
                }
                for (size_t rd = k; rd < srcRank; rd++) {
                    expectUpdateShape[updateAxisIter] = srcDataDim[rd];
                    updateAxisIter++;
                }
                if (expectUpdateShape.size() != updateRank) {
                    OPENVINO_THROW(errorPrefix,
                                   " do not have matched tensor rank relationship for input, indices and update");
                }
                for (size_t ru = 0; ru < updateRank; ru++) {
                    if (!dimsEqualWeak(updateDim[ru], expectUpdateShape[ru])) {
                        OPENVINO_THROW(errorPrefix,
                                       " do not have matched tensor shape relationship for input, indices and update");
                    }
                }
            }
            break;
        }
        case ScatterUpdateMode::ScatterElementsUpdate: {
            if (srcRank != indicesRank || srcRank != updateRank) {
                OPENVINO_THROW(errorPrefix, " do not have the same tensor rank for input, indices and update");
            }
            for (size_t ri = 0; ri < indicesRank; ri++) {
                if (!dimsEqualWeak(indicesDim[ri], updateDim[ri])) {
                    OPENVINO_THROW(errorPrefix, " do not have the same tensor shape for indices and update");
                }
            }
            break;
        }
        default: {
            OPENVINO_THROW(errorPrefix, " is not supported");
        }
    }

    indicesPrec = getOriginalInputPrecisionAtPort(INDICES_ID);
    auto indicesType = DnnlExtensionUtils::ElementTypeToDataType(indicesPrec);
    indicesSize = DnnlExtensionUtils::sizeOfDataType(indicesType);
    if (indicesSize >= 8) {
        indicesPrec = ov::element::i64;
        indicesSize = 8;
    } else {
        indicesPrec = ov::element::i32;
        indicesSize = 4;
    }

    if (axisRelaxed) {
        axisPrec = getOriginalInputPrecisionAtPort(AXIS_ID);
        auto axisType = DnnlExtensionUtils::ElementTypeToDataType(axisPrec);
        axisSize = DnnlExtensionUtils::sizeOfDataType(axisType);
        if (axisSize >= 8) {
            axisPrec = ov::element::i64;
            axisSize = 8;
        } else {
            axisPrec = ov::element::i32;
            axisSize = 4;
        }
    }

    dataPrec = getOriginalInputPrecisionAtPort(DATA_ID);
    dataSize = dataPrec.size();

    bool canBeInplace = !isDynamicNode() && getParentEdgeAt(DATA_ID)->getParent()->getChildEdges().size() == 1 &&
                        !getParentEdgeAt(DATA_ID)->getParent()->isConstant();

    NodeConfig config;
    if (axisRelaxed) {
        config.inConfs.resize(4);
    } else {
        config.inConfs.resize(3);
    }
    config.outConfs.resize(1);
    config.inConfs[DATA_ID].constant(false);
    config.inConfs[INDICES_ID].constant(false);
    config.inConfs[UPDATE_ID].constant(false);
    config.outConfs[0].constant(false);
    config.inConfs[DATA_ID].inPlace(canBeInplace ? 0 : -1);
    config.inConfs[INDICES_ID].inPlace(-1);
    config.inConfs[UPDATE_ID].inPlace(-1);
    config.outConfs[0].inPlace(canBeInplace ? 0 : -1);
    if (axisRelaxed) {
        config.inConfs[AXIS_ID].constant(false);
        config.inConfs[AXIS_ID].inPlace(-1);
    }

    std::vector<PortConfigurator> inPortConfig{{LayoutType::ncsp, dataPrec}, {LayoutType::ncsp, indicesPrec}, {LayoutType::ncsp, dataPrec}};
    if (axisRelaxed)
        inPortConfig.emplace_back(LayoutType::ncsp, axisPrec);
    addSupportedPrimDesc(inPortConfig,
                         {{LayoutType::ncsp, dataPrec}},
                          impl_desc_type::unknown);
}

bool ScatterUpdate::needPrepareParams() const {
    return false;
}

void ScatterUpdate::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

int64_t ScatterUpdate::getIndicesValue(uint8_t *indices, size_t offset) {
    auto *indicesPtr = indices + offset * indicesSize;
    int64_t ret = 0;
    if (indicesSize == 4) {
        auto *indicesPtr32 = reinterpret_cast<int32_t*>(indicesPtr);
        ret = *indicesPtr32;
    } else {
        auto *indicesPtr64 = reinterpret_cast<int64_t*>(indicesPtr);
        ret = *indicesPtr64;
    }
    return ret;
}

// 5D example:
// shapeND: n     c     d     h    w
// blockND: ncdhw cdhw  dhw   hw   w    1
// index  : 0      1    2     3    4    5
static std::vector<size_t> getBlockND(const VectorDims& shape) {
    size_t shapeRank = shape.size();
    std::vector<size_t> blockND(shapeRank + 1, 1);
    for (int i = shapeRank - 1; i >= 0; i--) {
        blockND[i] = shape[i] * blockND[i+1];
    }
    return blockND;
}

template <typename T>
static T reduction_neutral_value(const Reduction reduction_type) {
    switch (reduction_type) {
    case Reduction::MAX:
        return std::numeric_limits<T>::lowest();
    case Reduction::MIN:
        return std::numeric_limits<T>::max();
    case Reduction::PROD:
        return T{1};
    case Reduction::SUM:
    case Reduction::MEAN:
    case Reduction::NONE:
        return T{0};
    default:
        OPENVINO_THROW("Neutral value not available for this type of reduction");
        return 0;
    }
}

static ReduceMultiply reduce_multiply;
static ReduceAdd reduce_add;
static ReduceMean reduce_mean;
static ReduceMaximum reduce_maximum;
static ReduceMinimum reduce_minimum;
static TensorAssign tensor_assign;

void ScatterUpdate::execute(dnnl::stream strm) {
    auto srcMemPtr = getSrcMemoryAtPort(DATA_ID);
    auto dstMemPtr = getDstMemoryAtPort(0);
    auto indicesMemPtr = getSrcMemoryAtPort(INDICES_ID);
    auto updateMemPtr = getSrcMemoryAtPort(UPDATE_ID);

    uint8_t *dstPtr = dstMemPtr->getDataAs<uint8_t>();
    uint8_t *srcPtr = srcMemPtr->getDataAs<uint8_t>();
    uint8_t *indicesPtr = indicesMemPtr->getDataAs<uint8_t>();
    uint8_t *updatePtr = updateMemPtr->getDataAs<uint8_t>();

    const auto& srcDataDim = getParentEdgeAt(DATA_ID)->getMemory().getStaticDims();
    const auto& indicesDim = getParentEdgeAt(INDICES_ID)->getMemory().getStaticDims();
    size_t srcRank = srcDataDim.size();

    // 1d short vector scatter update optimized for shape inference subgraph
    if (scatterUpdateMode == ScatterUpdateMode::ScatterUpdate && srcDataDim.size() == 1 && indicesDim.size() <= 1 &&
        indicesPrec == ov::element::i32 && dataPrec == ov::element::i32 && srcDataDim[0] <= 64) {
        auto updateDims = updateMemPtr->getStaticDims();
        if (updateDims.size() <= 1) {
            DEBUG_LOG(getName(), " exec1DCase");
            auto updateCnt = (updateDims.size() == 0) ? 1 : updateDims[0];
            auto srcLength = srcMemPtr->getStaticDims()[0];
            auto* psrc = reinterpret_cast<int32_t*>(srcPtr);
            auto* pdst = reinterpret_cast<int32_t*>(dstPtr);
            for (size_t i = 0; i < srcLength; i++) {
                pdst[i] = psrc[i];
            }
            auto* pindices = reinterpret_cast<int32_t*>(indicesPtr);
            auto* pupdate = reinterpret_cast<int32_t*>(updatePtr);
            for (size_t i = 0; i < updateCnt; i++) {
                pdst[pindices[i]] = pupdate[i];
            }
            return;
        }
    }

    int axis = 0;
    if (axisRelaxed) {
        auto axisMemPtr = getSrcMemoryAtPort(AXIS_ID);
        uint8_t *axisPtr = axisMemPtr->getDataAs<uint8_t>();
        if (axisSize == 4) {
            auto *axisPtr32 = reinterpret_cast<int32_t*>(axisPtr);
            axis = *axisPtr32;
        } else {
            auto *axisPtr64 = reinterpret_cast<int64_t*>(axisPtr);
            axis = *axisPtr64;
        }

        if (axis >= static_cast<int>(srcRank) || axis < (static_cast<int>(srcRank) * - 1)) {
            OPENVINO_THROW(errorPrefix
           , " should have axis value in range [-r, r - 1], where r is the rank of input data");
        }
        axis = axis < 0 ? (axis + srcRank) : axis;

        size_t srcDimAxis = srcDataDim[axis];
        std::vector<size_t> indicesBlockND = getBlockND(indicesDim);
        parallel_nt(0, [&](const int ithr, const int nthr) {
            size_t start = 0, end = 0;
            splitter(indicesBlockND[0], nthr, ithr, start, end);
            for (size_t i = start; i < end; i++) {
                int64_t idxValue =  getIndicesValue(indicesPtr, i);
                if (idxValue >= static_cast<int64_t>(srcDimAxis) ||
                    (idxValue < 0 && scatterUpdateMode != ScatterUpdateMode::ScatterElementsUpdate)) {
                    OPENVINO_THROW(errorPrefix
                              , " have indices value that points to non-existing output tensor element");
                }
            }
        });

        if (scatterUpdateMode == ScatterUpdateMode::ScatterUpdate) {
            VectorDims indicesDim = getParentEdgeAt(INDICES_ID)->getMemory().getStaticDims();
            VectorDims updateDim = getParentEdgeAt(UPDATE_ID)->getMemory().getStaticDims();
            size_t indicesRank = indicesDim.size();
            size_t updateRank = updateDim.size();
            VectorDims expectUpdateShape(srcRank + indicesRank - 1, 0);
            int axisIter = 0;
            for (size_t rs = 0; rs < srcRank; rs++) {
                if (rs != static_cast<size_t>(axis)) {
                    expectUpdateShape[axisIter] = srcDataDim[rs];
                    axisIter++;
                } else {
                    for (size_t ri = 0; ri < indicesRank; ri++) {
                        expectUpdateShape[axisIter] = indicesDim[ri];
                        axisIter++;
                    }
                }
            }
            if (updateRank > expectUpdateShape.size())
                OPENVINO_THROW(errorPrefix,
                               " cannot update shape. New rank: ",
                               updateRank,
                               ", expected: ",
                               expectUpdateShape.size());
            for (size_t ru = 0; ru < updateRank; ru++) {
                if (updateDim[ru] != expectUpdateShape[ru]) {
                    OPENVINO_THROW(errorPrefix,
                                   " do not have matched tensor shape relationship for input, indices and update");
                }
            }
        }
    }

    if (srcPtr != dstPtr) {
        std::vector<size_t> srcBlockND = getBlockND(srcDataDim);
        parallel_nt(0, [&](const int ithr, const int nthr) {
            size_t start = 0, end = 0;
            splitter(srcBlockND[0], nthr, ithr, start, end);
            size_t size = (end - start) * dataSize;
            start *= dataSize;
            cpu_memcpy(dstPtr + start, srcPtr + start, size);
        });
    }

    if (isInputTensorAtPortEmpty(INDICES_ID)) {
        return;
    }

    switch (scatterUpdateMode) {
        case ScatterUpdateMode::ScatterUpdate: {
            scatterUpdate(indicesPtr, updatePtr, axis, dstPtr);
            break;
        }
        case ScatterUpdateMode::ScatterNDUpdate: {
            scatterNDUpdate(indicesPtr, updatePtr, dstPtr);
            break;
        }
        case ScatterUpdateMode::ScatterElementsUpdate: {
            switch (reduction_type) {
            case Reduction::NONE :
                scatterElementsUpdate<float, int32_t>(dstMemPtr, indicesMemPtr, updateMemPtr, axis, tensor_assign);
                break;
            case Reduction::SUM:
                scatterElementsUpdate<float, int32_t>(dstMemPtr, indicesMemPtr, updateMemPtr, axis, reduce_add);
                break;
            case Reduction::MAX :
                scatterElementsUpdate<float, int32_t>(dstMemPtr, indicesMemPtr, updateMemPtr, axis, reduce_maximum);
                break;
            case Reduction::MIN :
                scatterElementsUpdate<float, int32_t>(dstMemPtr, indicesMemPtr, updateMemPtr, axis, reduce_minimum);
                break;
            case Reduction::PROD :
                scatterElementsUpdate<float, int32_t>(dstMemPtr, indicesMemPtr, updateMemPtr, axis, reduce_multiply);
                break;
            case Reduction::MEAN :
                scatterElementsUpdate<float, int32_t>(dstMemPtr, indicesMemPtr, updateMemPtr, axis, reduce_mean);
                break;
            default :
                break;
            }
            break;
        }
        default: {
            OPENVINO_THROW(errorPrefix
           , " is not supported");
        }
    }
}

// For the data tensor of shape [d_0, d_1, ..., d_n],
// and indices tensor of shape [i_0, i_1, ..., i_k].
// Updates tensor shape should be [d_0, d_1, ... d_(axis - 1), i_0, i_1, ..., i_k, d_(axis + 1), ..., d_n].
void ScatterUpdate::scatterUpdate(uint8_t *indices, uint8_t *update, int axis, uint8_t *dstData) {
    const auto& srcDataDim = getParentEdgeAt(DATA_ID)->getMemory().getStaticDims();
    const auto& indicesDim = getParentEdgeAt(INDICES_ID)->getMemory().getStaticDims();
    const auto& updateDim = getParentEdgeAt(UPDATE_ID)->getMemory().getStaticDims();
    size_t indicesRank = indicesDim.size();

    std::vector<size_t> srcBlockND = getBlockND(srcDataDim);
    std::vector<size_t> updateBlockND = getBlockND(updateDim);

    const size_t mulIdentity = 1;
    size_t idxLength = mulIdentity;
    for (size_t ri = 0; ri < indicesRank; ri++) {
        idxLength *= indicesDim[ri];
    }
    size_t batchToUpdate = mulIdentity;
    for (int x = 0; x < axis; x++) {
        batchToUpdate *= srcDataDim[x];
    }
    // blockToUpdate is srcBlockND[axis + 1], also is updateBlockND[axis + indicesRank]
    size_t blockToUpdate = srcBlockND[axis + 1];
    size_t blockToUpdateSize = blockToUpdate * dataSize;

    parallel_for2d(batchToUpdate, idxLength, [&](size_t b, size_t idx) {
        int64_t idxValue = getIndicesValue(indices, idx);
        uint8_t *dstEntry = dstData + (b * srcBlockND[axis] + idxValue * blockToUpdate) * dataSize;
        uint8_t *updateEntry = update + (b * updateBlockND[axis] + idx * blockToUpdate) * dataSize;
        cpu_memcpy(dstEntry, updateEntry, blockToUpdateSize);
    });
}

// indices is a (q-1)-dimension tensor of k-tuple,
// k is indices.shape[-1] and should not be greater than rank of input, q is rank of indicies.
// updates is a (q-1)-dimension tensor of replacement-slice-values
void ScatterUpdate::scatterNDUpdate(uint8_t *indices, uint8_t *update, uint8_t *dstData) {
    const auto& srcDataDim = getParentEdgeAt(DATA_ID)->getMemory().getStaticDims();
    const auto& indicesDim = getParentEdgeAt(INDICES_ID)->getMemory().getStaticDims();
    size_t indicesRank = indicesDim.size();

    std::vector<size_t> srcBlockND = getBlockND(srcDataDim);

    size_t k = indicesDim[indicesRank - 1];
    size_t idxTupleNum = 1;
    for (size_t ri = 0; ri < indicesRank - 1; ri++) {
        idxTupleNum *= indicesDim[ri];
    }

    size_t sizeToUpdate = srcBlockND[k] * dataSize;
    parallel_for(idxTupleNum, [&](size_t tupleIdx) {
        size_t indicesOffset = tupleIdx * k;
        size_t dstOffset = 0;
        for (size_t i = 0; i < k; i++) {
            int64_t idxValue = getIndicesValue(indices, indicesOffset + i);
            if (idxValue < 0) {
                // Negative value for indices means counting backwards from the end.
                idxValue += srcDataDim[i];
            }
            dstOffset += idxValue * srcBlockND[i + 1];
        }
        dstOffset *= dataSize;
        size_t updateOffset = tupleIdx * sizeToUpdate;
        cpu_memcpy(dstData + dstOffset, update + updateOffset, sizeToUpdate);
    });
}

static std::vector<size_t> getCoordinate(size_t offset, const VectorDims& shape) {
    size_t shapeRank = shape.size();
    std::vector<size_t> coordinate;
    coordinate.resize(shapeRank);
    for (int i = shapeRank - 1; i >= 0; i--) {
        coordinate[i] = offset % shape[i];
        offset /= shape[i];
    }
    return coordinate;
}

// output[indices[i][j][k]][j][k] = updates[i][j][k] if axis = 0,
// output[i][indices[i][j][k]][k] = updates[i][j][k] if axis = 1,
// output[i][j][indices[i][j][k]] = updates[i][j][k] if axis = 2.
template <typename DataType, typename IndexType, typename func_t>
void ScatterUpdate::scatterElementsUpdate(const MemoryPtr& mem_data, const MemoryPtr& mem_indices, const MemoryPtr& mem_updates, int axis, func_t& kernel_func) {
    PlainTensor data_buf, indices_buf, updates_buf;
    data_buf.reset(mem_data);
    indices_buf.reset(mem_indices);
    updates_buf.reset(mem_updates);

    const auto& data_shape = mem_data->getStaticDims();
    const auto& indices_shape = mem_indices->getStaticDims();
    size_t indices_rank = indices_shape.size();

    const auto data_dim_size = data_shape[axis];
    const auto index_dim_size = indices_shape[axis];

    if (axis < 0)
        axis += indices_rank;

    VectorDims squashed_indices_shape(indices_shape);
    squashed_indices_shape[axis] = 1;

    if (!use_init_val) {
        const auto value = reduction_neutral_value<DataType>(reduction_type);
        parallel_nt(0, [&](const int ithr, const int nthr) {
            size_t start = 0, end = 0;
            splitter(shape_size(squashed_indices_shape), nthr, ithr, start, end);

            for (size_t worker = start; worker < end; worker++) {
                std::vector<size_t> indices_coord = getCoordinate(worker, squashed_indices_shape);
                std::vector<size_t> data_coord(indices_coord);

                for (size_t i = 0; i < index_dim_size; i++) {
                    indices_coord[axis] = i;
                    IndexType idxValue = indices_buf.at<IndexType, size_t>(indices_coord);
                    size_t normalized_idxValue = static_cast<size_t>((idxValue < 0) ? idxValue + data_dim_size : idxValue);
                    if (normalized_idxValue < data_dim_size) {
                        data_coord[axis] = normalized_idxValue;
                        data_buf.at<DataType, size_t>(data_coord) = value;
                    }
                }
            }
        });
    }

    // process serially along 'axis' dimension because of data dependency brought by duplicated value in indices
    if (axis == static_cast<int>(indices_rank - 1)) {
        parallel_nt(0, [&](const int ithr, const int nthr) {
            size_t start = 0, end = 0;
            splitter(shape_size(squashed_indices_shape), nthr, ithr, start, end);

            for (size_t worker = start; worker < end; worker++) {
                std::vector<size_t> indices_coord = getCoordinate(worker, squashed_indices_shape);
                std::vector<size_t> data_coord(indices_coord);

                // inner axis loop for better performance
                for (size_t i = 0; i < index_dim_size; i++) {
                    indices_coord[axis] = i;
                    IndexType idxValue = indices_buf.at<IndexType, size_t>(indices_coord);
                    size_t normalized_idxValue = static_cast<size_t>((idxValue < 0) ? idxValue + data_dim_size : idxValue);
                    if (normalized_idxValue < data_dim_size) {
                        data_coord[axis] = normalized_idxValue;
                        DataType& dst = data_buf.at<DataType, size_t>(data_coord);
                        DataType src = updates_buf.at<DataType, size_t>(indices_coord);
                        kernel_func(dst, src);
                    }
                }
            }
        });
    } else {
        parallel_nt(0, [&](const int ithr, const int nthr) {
            size_t start = 0, end = 0;
            splitter(shape_size(squashed_indices_shape), nthr, ithr, start, end);

            // external axis loop for better performance
            for (size_t i = 0; i < index_dim_size; i++) {
                for (size_t worker = start; worker < end; worker++) {
                    std::vector<size_t> indices_coord = getCoordinate(worker, squashed_indices_shape);
                    std::vector<size_t> data_coord(indices_coord);
                    indices_coord[axis] = i;
                    IndexType idxValue = indices_buf.at<IndexType, size_t>(indices_coord);
                    size_t normalized_idxValue = static_cast<size_t>((idxValue < 0) ? idxValue + data_dim_size : idxValue);
                    if (normalized_idxValue < data_dim_size) {
                        data_coord[axis] = normalized_idxValue;
                        DataType& dst = data_buf.at<DataType, size_t>(data_coord);
                        DataType src = updates_buf.at<DataType, size_t>(indices_coord);
                        kernel_func(dst, src);
                    }
                }
            }
        });
    }
}

template <typename DataType, typename IndexType>
void ScatterUpdate::scatterElementsUpdate(const MemoryPtr& mem_data, const MemoryPtr& mem_indices, const MemoryPtr& mem_updates, int axis, ReduceMean& kernel_func) {
    PlainTensor data_buf, indices_buf, updates_buf;
    data_buf.reset(mem_data);
    indices_buf.reset(mem_indices);
    updates_buf.reset(mem_updates);

    const auto& data_shape = mem_data->getStaticDims();
    const auto& indices_shape = mem_indices->getStaticDims();
    size_t indices_rank = indices_shape.size();

    const auto data_dim_size = data_shape[axis];
    const auto index_dim_size = indices_shape[axis];

    if (axis < 0)
        axis += indices_rank;

    VectorDims squashed_indices_shape(indices_shape);
    squashed_indices_shape[axis] = 1;

    if (!use_init_val) {
        const auto value = reduction_neutral_value<DataType>(reduction_type);
        parallel_nt(0, [&](const int ithr, const int nthr) {
            size_t start = 0, end = 0;
            splitter(shape_size(squashed_indices_shape), nthr, ithr, start, end);

            for (size_t worker = start; worker < end; worker++) {
                std::vector<size_t> indices_coord = getCoordinate(worker, squashed_indices_shape);
                std::vector<size_t> data_coord(indices_coord);

                for (size_t i = 0; i < index_dim_size; i++) {
                    indices_coord[axis] = i;
                    IndexType idxValue = indices_buf.at<IndexType, size_t>(indices_coord);
                    size_t normalized_idxValue = static_cast<size_t>((idxValue < 0) ? idxValue + data_dim_size : idxValue);
                    if (normalized_idxValue < data_dim_size) {
                        data_coord[axis] = normalized_idxValue;
                        data_buf.at<DataType, size_t>(data_coord) = value;
                    }
                }
            }
        });
    }

    // process serially along 'axis' dimension because of data dependency brought by duplicated value in indices
    parallel_nt(0, [&](const int ithr, const int nthr) {
        size_t start = 0, end = 0;
        splitter(shape_size(squashed_indices_shape), nthr, ithr, start, end);

        for (size_t worker = start; worker < end; worker++) {
            std::vector<size_t> indices_coord = getCoordinate(worker, squashed_indices_shape);
            std::vector<size_t> data_coord(indices_coord);

            std::unordered_map<size_t, int32_t> mean_reduction_counters;

            // inner axis loop for better performance
            for (size_t i = 0; i < index_dim_size; i++) {
                indices_coord[axis] = i;
                IndexType idxValue = indices_buf.at<IndexType, size_t>(indices_coord);
                size_t normalized_idxValue = static_cast<size_t>((idxValue < 0) ? idxValue + data_dim_size : idxValue);
                if (normalized_idxValue < data_dim_size) {
                    data_coord[axis] = normalized_idxValue;
                    DataType& dst = data_buf.at<DataType, size_t>(data_coord);
                    DataType src = updates_buf.at<DataType, size_t>(indices_coord);
                    kernel_func(dst, src);
                }

                if (reduction_type == Reduction::MEAN) {
                    mean_reduction_counters[normalized_idxValue] += 1;
                }
            }

            if (reduction_type == Reduction::MEAN) {
                for (const auto& counter : mean_reduction_counters) {
                    data_coord[axis] = counter.first;
                    DataType& dst = data_buf.at<DataType, size_t>(data_coord);
                    const auto N = counter.second + static_cast<int32_t>(use_init_val);
                    dst = static_cast<DataType>(static_cast<double>(dst) / N);
                }
            }
        }
    });
}

bool ScatterUpdate::created() const {
    return getType() == Type::ScatterUpdate
            || getType() == Type::ScatterElementsUpdate
            || getType() == Type::ScatterNDUpdate;
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
