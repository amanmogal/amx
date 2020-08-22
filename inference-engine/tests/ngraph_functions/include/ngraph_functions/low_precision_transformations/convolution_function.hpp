// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/ngraph.hpp>
#include "ngraph_functions/low_precision_transformations/common/fake_quantize_on_weights.hpp"
#include "ngraph_functions/low_precision_transformations/common/fake_quantize_on_data.hpp"
#include "ngraph_functions/low_precision_transformations/common/dequantization_operations.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

class ConvolutionFunction {
public:
    class ActualValues {
    public:
        ngraph::element::Type lowPrecision;
        std::vector<float> subtractValues;
        std::vector<float> mutliplyValues;
        std::vector<float> weightsValues;
        builder::subgraph::FakeQuantizeOnWeights fakeQuantizeOnWeights;
    };

    class ExpectedValues {
    public:
        ngraph::element::Type activationPrecision;
        std::vector<float> subtractValues;
        ngraph::element::Type weightsPrecision;
        std::vector<float> weightsValues;
        builder::subgraph::FakeQuantizeOnWeights fakeQuantizeOnWeights;;
        std::vector<float> mutliplyValues;
    };

    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const bool updatePrecisions,
        const ActualValues& actualValues);

    static std::shared_ptr<ngraph::Function> getOriginalWithIncorrectWeights(
        const ngraph::Shape& inputShape,
        ngraph::element::Type precision,
        ngraph::builder::subgraph::FakeQuantizeOnWeights fakeQuantizeOnWeights,
        ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantizeOnData,
        bool isCrorrect);

    static std::shared_ptr<ngraph::Function> getReferenceWithIncorrectWeights(
        const ngraph::Shape& inputShape,
        ngraph::element::Type precision,
        ngraph::element::Type dataPrecision,
        ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantizeOnData,
        ngraph::builder::subgraph::DequantizationOperations dequantizationBefore,
        ngraph::element::Type weightsPrecision,
        std::vector<float> weightsValues,
        ngraph::builder::subgraph::FakeQuantizeOnWeights fakeQuantizeOnWeights,
        ngraph::builder::subgraph::DequantizationOperations dequantizationAfter,
        bool isCorrect);

    static std::shared_ptr<ngraph::Function> getReference(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const bool updatePrecisions,
        const ExpectedValues& expectedValues);
};

inline std::ostream& operator<<(std::ostream& out, const ConvolutionFunction::ActualValues& values) {
    return out << "_" << values.lowPrecision <<
        "_subtract" << values.subtractValues.size() <<
        "_mutliply" << values.mutliplyValues.size() << "_" <<
        values.fakeQuantizeOnWeights;
}

inline std::ostream& operator<<(std::ostream& out, const ConvolutionFunction::ExpectedValues& values) {
    return out << "_" << values.activationPrecision <<
        "_subtract" << values.subtractValues.size() <<
        "_weightsPrecision" << values.weightsPrecision << "_" <<
        values.fakeQuantizeOnWeights;
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
