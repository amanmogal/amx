// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/node.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/util/unary_elementwise_arithmetic.hpp"

namespace ngraph {
namespace op {
namespace v0 {
/// \brief Gaussian Error Linear Unit
/// f(x) = 0.5 * x * (1 + erf( x / sqrt(2) )
class NGRAPH_API Gelu : public Op {
public:
    NGRAPH_RTTI_DECLARATION;

    Gelu();
    /// \brief Constructs a Gelu operation.
    ///
    /// \param data Input tensor
    Gelu(const Output<Node>& data);

    bool visit_attributes(AttributeVisitor& visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
};
}  // namespace v0
using v0::Gelu;

/// \brief Specifies the approximation to calculate Gelu
enum class GeluApproximationMode { TANH, ERF };
NGRAPH_API std::ostream& operator<<(std::ostream& s, const GeluApproximationMode& type);

namespace v7 {
/// \brief Gaussian Error Linear Unit
/// f(x) = 0.5 * x * (1 + erf( x / sqrt(2) ) for "approximation" = "erf"
/// f(x) = 0.5 * x * (1 + tanh([sqrt(2 / pi)] * [x + 0.044715^3]) for "approximation" =
/// "tanh"
class NGRAPH_API Gelu : public util::UnaryElementwiseArithmetic {
public:
    NGRAPH_RTTI_DECLARATION;

    Gelu() = default;
    /// \brief Constructs a Gelu operation.
    ///
    /// \param data Input tensor
    /// \param mode Approximation mode
    Gelu(const Output<Node>& data, GeluApproximationMode mode = GeluApproximationMode::ERF);

    bool visit_attributes(AttributeVisitor& visitor) override;

    void validate_and_infer_types() override;

    bool evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const override;
    bool has_evaluate() const override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    GeluApproximationMode get_approximation_mode() const;

private:
    GeluApproximationMode m_approximation_mode = GeluApproximationMode::ERF;
};
}  // namespace v7
}  // namespace op
}  // namespace ngraph

namespace ov {
template <>
class NGRAPH_API AttributeAdapter<ngraph::op::GeluApproximationMode>
    : public EnumAttributeAdapterBase<ngraph::op::GeluApproximationMode> {
public:
    AttributeAdapter(ngraph::op::GeluApproximationMode& value)
        : EnumAttributeAdapterBase<ngraph::op::GeluApproximationMode>(value) {}

    static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<op::GeluApproximationMode>", 0};
    const DiscreteTypeInfo& get_type_info() const override {
        return type_info;
    }
};

}  // namespace ov
