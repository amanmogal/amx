// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <transformations/common_optimizations/transpose_sinking_binary.hpp>

#include <transformations/init_node_info.hpp>
#include <openvino/frontend/manager.hpp>
#include <openvino/opsets/opset9.hpp>
#include <openvino/pass/manager.hpp>
#include "common_test_utils/ngraph_test_utils.hpp"

#include <functional>

#include "gtest/gtest.h"

#include "ngraph/pass/visualize_tree.hpp" // DEBUG

namespace {

using NodePtr = std::shared_ptr<ov::Node>;
using Nodes = std::vector<NodePtr>;
using ModelPtr = std::shared_ptr<ov::Model>;
using Output = ov::Output<ov::Node>;

using FloatPtr = std::unique_ptr<float[]>;

template <class IterT, class T>
void Fill(IterT begin_iter, IterT end_iter, T value, T step)
{
    while (begin_iter != end_iter)
    {
        *begin_iter = value;
        value += step;
        ++begin_iter;
    }
}

FloatPtr GenerateTestInput(const ov::Shape & input_shape)
{
    const size_t size = std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<size_t>());

    FloatPtr input(new float[size]);
    Fill(input.get(), input.get() + size, 0.01, 0.01);

    return input;
}

std::string GetFinalNodeName(std::shared_ptr<ov::Model> model, int index = 0)
{
    NodePtr result_node = model->get_results()[index];
    return result_node->get_input_node_ptr(0)->get_friendly_name();
}

}

TEST(TransposeSinkingBinaryTest, TransposeSinkingConcatMultTransposesForward) {

    ngraph::Shape input_shape{1, 4, 1, 4};
    auto input_type = ngraph::element::f32;

    std::shared_ptr<ngraph::Function> function, reference_function, original_function;
    {
        auto X = std::make_shared<ov::opset9::Parameter>(input_type, input_shape);

        auto const1 = ov::opset9::Constant::create(input_type, ngraph::Shape{1, 4, 1, 4}, {1});

        auto const2 = ov::opset9::Constant::create(input_type, ngraph::Shape{1, 4, 4, 1}, {2});
        auto ng_order2 = std::make_shared<ov::opset9::Constant>(ngraph::element::u64, ngraph::Shape{4}, ngraph::Shape{0, 2, 3, 1});
        auto transpose2 = std::make_shared<ov::opset9::Transpose>(const2, ng_order2);

        auto const3 = ov::opset9::Constant::create(input_type, ngraph::Shape{1, 4, 1, 4}, {3});

        auto const4 = ov::opset9::Constant::create(input_type, ngraph::Shape{1, 4, 4, 1}, {4});
        auto ng_order4 = std::make_shared<ov::opset9::Constant>(ngraph::element::u64, ngraph::Shape{4}, ngraph::Shape{0, 2, 3, 1});
        auto transpose4 = std::make_shared<ov::opset9::Transpose>(const4, ng_order4);

        auto const5 = ov::opset9::Constant::create(input_type, ngraph::Shape{1, 4, 1, 4}, {5});
        auto concat = std::make_shared<ov::opset9::Concat>(ov::OutputVector{X, const1, transpose2, const3, transpose4, const5}, 1);

        function = std::make_shared<ngraph::Function>(concat, ngraph::ParameterVector{X});
        original_function = function->clone();

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::TransposeSinkingConcatForward>();
        manager.run_passes(function);
        ASSERT_NO_THROW(check_rt_info(function));

        EXPECT_EQ(GetFinalNodeName(original_function), GetFinalNodeName(function));
    }

    {
        auto X = std::make_shared<ov::opset9::Parameter>(input_type, input_shape);

        auto ng_order = std::make_shared<ov::opset9::Constant>(ngraph::element::u64, ngraph::Shape{4}, ngraph::Shape{0, 3, 1, 2});
        auto transpose = std::make_shared<ov::opset9::Transpose>(X, ng_order);

        auto const1 = ov::opset9::Constant::create(input_type, ngraph::Shape{1, 4, 1, 4}, {1});
        auto ng_order1 = std::make_shared<ov::opset9::Constant>(ngraph::element::u64, ngraph::Shape{4}, ngraph::Shape{0, 3, 1, 2});
        auto transpose1 = std::make_shared<ov::opset9::Transpose>(const1, ng_order1);
        
        auto const2 = ov::opset9::Constant::create(input_type, ngraph::Shape{1, 4, 4, 1}, {2});

        auto const3 = ov::opset9::Constant::create(input_type, ngraph::Shape{1, 4, 1, 4}, {3});
        auto ng_order3 = std::make_shared<ov::opset9::Constant>(ngraph::element::u64, ngraph::Shape{4}, ngraph::Shape{0, 3, 1, 2});
        auto transpose3 = std::make_shared<ov::opset9::Transpose>(const3, ng_order3);

        auto const4 = ov::opset9::Constant::create(input_type, ngraph::Shape{1, 4, 4, 1}, {4});
        auto ng_order4 = std::make_shared<ov::opset9::Constant>(ngraph::element::u64, ngraph::Shape{4}, ngraph::Shape{0, 2, 3, 1});
        auto transpose4 = std::make_shared<ov::opset9::Transpose>(const4, ng_order4);
        auto ng_order4_1 = std::make_shared<ov::opset9::Constant>(ngraph::element::u64, ngraph::Shape{4}, ngraph::Shape{0, 3, 1, 2});
        auto transpose4_1 = std::make_shared<ov::opset9::Transpose>(transpose4, ng_order4_1);

        auto const5 = ov::opset9::Constant::create(input_type, ngraph::Shape{1, 4, 1, 4}, {5});
        auto ng_order5 = std::make_shared<ov::opset9::Constant>(ngraph::element::u64, ngraph::Shape{4}, ngraph::Shape{0, 3, 1, 2});
        auto transpose5 = std::make_shared<ov::opset9::Transpose>(const5, ng_order5);

        auto concat = std::make_shared<ov::opset9::Concat>(ov::OutputVector{transpose, transpose1, const2, transpose3, transpose4_1, transpose5}, 2);

        auto ng_order_after = std::make_shared<ov::opset9::Constant>(ngraph::element::u64, ngraph::Shape{4}, ngraph::Shape{0, 2, 3, 1});
        auto transpose_after = std::make_shared<ov::opset9::Transpose>(concat, ng_order_after);

        reference_function = std::make_shared<ngraph::Function>(transpose_after, ngraph::ParameterVector{X});
    }

    FunctionsComparator func_comparator = FunctionsComparator::with_default();
    func_comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    const FunctionsComparator::Result result = func_comparator(function, reference_function);
    ASSERT_TRUE(result.valid) << result.message;

    FloatPtr test_input = GenerateTestInput(input_shape);
    ov::Tensor input_tensor{input_type, input_shape, test_input.get()};
    ov::TensorVector function_result(1), reference_function_result(1);
    ASSERT_TRUE(original_function->evaluate(function_result, ov::TensorVector{input_tensor}));
    EXPECT_EQ(function_result.size(), 1);
    EXPECT_EQ(function_result[0].get_element_type(), ngraph::element::f32);

    ASSERT_TRUE(reference_function->evaluate(reference_function_result, ov::TensorVector{input_tensor}));
    EXPECT_EQ(reference_function_result.size(), 1);
    EXPECT_EQ(reference_function_result[0].get_element_type(), ngraph::element::f32);

    EXPECT_EQ(reference_function_result[0].get_shape(), function_result[0].get_shape());
    EXPECT_EQ(reference_function_result[0].get_size(), function_result[0].get_size());

    const float * function_result_data = function_result[0].data<float>();
    const float * reference_function_result_data = reference_function_result[0].data<float>();
    for (size_t i = 0; i < reference_function_result[0].get_size(); ++i)
        EXPECT_EQ(function_result_data[i], reference_function_result_data[i]);
}

TEST(TransposeSinkingBinaryTest, TransposeSinkingMultipleAddBackward) {

    ngraph::Shape input_shape{1, 4, 1, 4};
    auto input_type = ngraph::element::f32;
    const size_t bin_ops_num = 10; 

    std::shared_ptr<ngraph::Function> function, reference_function, original_function;
    {
        auto X = std::make_shared<ov::opset9::Parameter>(input_type, input_shape);

        std::shared_ptr<ov::Node> in_operation = X;
        for (size_t i = 0; i < bin_ops_num; ++i) {
            auto right_const = ov::opset9::Constant::create(input_type, ngraph::Shape{1, 4, 1, 4}, {2});
            in_operation = std::make_shared<ov::opset9::Add>(in_operation, right_const);
        }

        auto ng_order = std::make_shared<ov::opset9::Constant>(ngraph::element::u64, ngraph::Shape{4}, ngraph::Shape{0, 2, 3, 1});
        auto transpose = std::make_shared<ov::opset9::Transpose>(in_operation, ng_order);

        function = std::make_shared<ngraph::Function>(transpose, ngraph::ParameterVector{X});
        original_function = function->clone();

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::VisualizeTree>("./0before.png"); // DEBUG
        manager.register_pass<ngraph::pass::TransposeSinkingBinaryBackward>();
        manager.register_pass<ngraph::pass::VisualizeTree>("./1after.png"); // DEBUG
        manager.run_passes(function);
        ASSERT_NO_THROW(check_rt_info(function));

        EXPECT_EQ(GetFinalNodeName(original_function), GetFinalNodeName(function));
    }

    {
        auto X = std::make_shared<ov::opset9::Parameter>(input_type, input_shape);

        std::shared_ptr<ov::Node> in_operation = X;

        {
            auto ng_order = std::make_shared<ov::opset9::Constant>(ngraph::element::u64, ngraph::Shape{4}, ngraph::Shape{0, 2, 3, 1});
            in_operation = std::make_shared<ov::opset9::Transpose>(in_operation, ng_order);
        }

        for (size_t i = 0; i < bin_ops_num; ++i) {
            auto right_const = ov::opset9::Constant::create(input_type, ngraph::Shape{1, 4, 1, 4}, {2});
            auto ng_order = std::make_shared<ov::opset9::Constant>(ngraph::element::u64, ngraph::Shape{4}, ngraph::Shape{0, 2, 3, 1});
            auto right_operation = std::make_shared<ov::opset9::Transpose>(right_const, ng_order);
            in_operation = std::make_shared<ov::opset9::Add>(in_operation, right_operation);
        }

        reference_function = std::make_shared<ngraph::Function>(in_operation, ngraph::ParameterVector{X});
    }

    FunctionsComparator func_comparator = FunctionsComparator::with_default();
    func_comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    const FunctionsComparator::Result result = func_comparator(function, reference_function);
    ASSERT_TRUE(result.valid) << result.message;

    FloatPtr test_input = GenerateTestInput(input_shape);
    ov::Tensor input_tensor{input_type, input_shape, test_input.get()};
    ov::TensorVector function_result(1), reference_function_result(1);
    ASSERT_TRUE(original_function->evaluate(function_result, ov::TensorVector{input_tensor}));
    EXPECT_EQ(function_result.size(), 1);
    EXPECT_EQ(function_result[0].get_element_type(), ngraph::element::f32);

    ASSERT_TRUE(reference_function->evaluate(reference_function_result, ov::TensorVector{input_tensor}));
    EXPECT_EQ(reference_function_result.size(), 1);
    EXPECT_EQ(reference_function_result[0].get_element_type(), ngraph::element::f32);

    EXPECT_EQ(reference_function_result[0].get_shape(), function_result[0].get_shape());
    EXPECT_EQ(reference_function_result[0].get_size(), function_result[0].get_size());

    const float * function_result_data = function_result[0].data<float>();
    const float * reference_function_result_data = reference_function_result[0].data<float>();
    for (size_t i = 0; i < reference_function_result[0].get_size(); ++i)
        EXPECT_EQ(function_result_data[i], reference_function_result_data[i]);
}

// --------------------------------------------------------------------------------------

struct GraphDesc
{
    std::shared_ptr<ov::opset9::Parameter> input;
    Nodes tail_nodes;
};

class GraphBuilder {
using SelfPtr = std::unique_ptr<GraphBuilder>;
public:
    GraphBuilder() = default;
    virtual ~GraphBuilder() = default;

    void SetNextBuilder(SelfPtr next_builder) { _next_builder.swap(next_builder); }
    void build(GraphDesc & graph) const
    {
        buildNodes(graph);
        if (_next_builder)
            _next_builder->build(graph);
    }

    static ov::element::Type GetElementType() { return _element_type; }
    static void SetElementType(ov::element::Type type) { _element_type = type; }
protected:
    virtual void buildNodes(GraphDesc & graph) const = 0;
private:
    static ov::element::Type _element_type;
    SelfPtr _next_builder = nullptr;
};
ov::element::Type GraphBuilder::_element_type = ov::element::f32;
using GraphBuilderPtr = std::unique_ptr<GraphBuilder>;

class CreateInput : public GraphBuilder {
public:
    CreateInput(const ov::Shape & input_shape) :
        _input_shape(input_shape) {}
    void buildNodes(GraphDesc & graph) const override;
private:
    const ov::Shape _input_shape;
};

void CreateInput::buildNodes(GraphDesc & graph) const
{
    graph.input = std::make_shared<ov::opset9::Parameter>(GetElementType(), _input_shape);
}

namespace {

std::shared_ptr<ov::opset9::Constant> CreateConstant(ov::element::Type element_type, const ov::Shape & shape)
{
    const size_t size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());

    std::vector<float> const_values(size);
    Fill(const_values.begin(), const_values.end(), 0.01, 0.01);

    return ov::opset9::Constant::create(element_type, shape, const_values);
}

} // namespace

class CreateConstants : public GraphBuilder {
public:
    CreateConstants(size_t constants_num,
                    const ov::Shape & constant_shape) :
        _constants_num(constants_num),
        _constant_shape(constant_shape) {}
    void buildNodes(GraphDesc & graph) const override;
private:
    const size_t _constants_num;
    const ov::Shape _constant_shape;
};

void CreateConstants::buildNodes(GraphDesc & graph) const
{
    graph.tail_nodes.reserve(graph.tail_nodes.size() + _constants_num);
    for (int i = 0; i < _constants_num; ++i) {
        graph.tail_nodes.push_back(CreateConstant(GetElementType(), _constant_shape));
    }
}

class AppendInput : public GraphBuilder {
public:
    AppendInput() = default;
    void buildNodes(GraphDesc & graph) const override;
};

void AppendInput::buildNodes(GraphDesc & graph) const
{
    graph.tail_nodes.push_back(graph.input);
}

class AppendInputTranpose : public GraphBuilder {
public:
    AppendInputTranpose(const ov::AxisVector & transpose_axis_order) :
                        _transpose_axis_order(transpose_axis_order) {}
    void buildNodes(GraphDesc & graph) const override;
private:
    const ov::AxisVector _transpose_axis_order;
};

void AppendInputTranpose::buildNodes(GraphDesc & graph) const
{
    auto tranpose_constant = std::make_shared<ov::opset9::Constant>(ngraph::element::u64,
                                                                    ov::Shape{_transpose_axis_order.size()},
                                                                    _transpose_axis_order);
    graph.tail_nodes.push_back(std::make_shared<ov::opset9::Transpose>(graph.input, tranpose_constant));
}

class AppendTranpose : public GraphBuilder {
public:
    AppendTranpose(const ov::AxisVector & transpose_axis_order,
                           const ov::AxisVector & input_axis) :
                           _transpose_axis_order(transpose_axis_order),
                           _input_axis(input_axis) {}
    void buildNodes(GraphDesc & graph) const override;
private:
    const ov::AxisVector _transpose_axis_order;
    const ov::AxisVector _input_axis;
};

void AppendTranpose::buildNodes(GraphDesc & graph) const
{
    for (size_t i : _input_axis) {
        auto tranpose_constant = std::make_shared<ov::opset9::Constant>(ngraph::element::u64,
                                                                    ov::Shape{_transpose_axis_order.size()},
                                                                    _transpose_axis_order);
        graph.tail_nodes[i] = std::make_shared<ov::opset9::Transpose>(graph.tail_nodes[i], tranpose_constant);
    }
}

class AppendConcat : public GraphBuilder {
public:
    AppendConcat(size_t concat_axis,
                 const ov::AxisVector & input_axis_order) :
                    _concat_axis(concat_axis),
                    _input_axis_order(input_axis_order) {}
    void buildNodes(GraphDesc & graph) const override;
private:
    const size_t _concat_axis;
    const ov::AxisVector _input_axis_order;
};

void AppendConcat::buildNodes(GraphDesc & graph) const
{
    ov::OutputVector input_nodes(_input_axis_order.size());
    for (size_t i = 0; i < _input_axis_order.size(); ++i) {
        input_nodes[i] = graph.tail_nodes[_input_axis_order[i]];
    }

    graph.tail_nodes[0] = std::make_shared<ov::opset9::Concat>(input_nodes, _concat_axis);
    graph.tail_nodes.resize(1);
}

template <typename BinaryT>
class AppendBinary : public GraphBuilder {
public:
    AppendBinary(const ov::AxisVector & input_axis_order) :
                    _input_axis_order(input_axis_order) {}
    void buildNodes(GraphDesc & graph) const override;
private:
    const ov::AxisVector _input_axis_order;
};

template <typename BinaryT>
void AppendBinary<BinaryT>::buildNodes(GraphDesc & graph) const
{
    ov::OutputVector input_nodes(_input_axis_order.size());
    for (size_t i = 0; i < _input_axis_order.size(); ++i) {
        input_nodes[i] = graph.tail_nodes[_input_axis_order[i]];
    }

    graph.tail_nodes[0] = std::make_shared<BinaryT>(input_nodes[0], input_nodes[1]);
    graph.tail_nodes.resize(1);
}

// --------------------------------------------------------------------------------------

namespace {
#if (__cplusplus < 201402L)
template<typename T, typename... Args>
std::unique_ptr<T> openvino_make_unique(Args&&... args)
{
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}
#else
#define openvino_make_unique std::make_unique
#endif
} // namespace

#define NEW(type, ...) \
    std::move(openvino_make_unique<type>(__VA_ARGS__))

template <typename T>
GraphBuilderPtr CreateBuilder(T builder)
{
    return builder;
}

template<typename T, typename... Args>
GraphBuilderPtr CreateBuilder(T builder, Args... args)
{
    GraphBuilderPtr prev_builder = CreateBuilder(std::forward<Args>(args)...);
    builder->SetNextBuilder(std::move(prev_builder));
    return builder;
}

template<typename T, typename... Args>
ModelPtr CreateModel(T builder, Args... args)
{
    GraphBuilderPtr graph_builder = CreateBuilder(std::forward<T>(builder), std::forward<Args>(args)...);

    GraphDesc graph_desc;
    graph_builder->build(graph_desc);

    return std::make_shared<ov::Model>(graph_desc.tail_nodes[0], ngraph::ParameterVector{graph_desc.input});
}

// ----------------------------------------------------------------------------

class IPassManagerFactory {
public:
    IPassManagerFactory() = default;
    virtual ~IPassManagerFactory() = default;
    virtual ngraph::pass::Manager createManager() const = 0;
};

using PassManagerFactoryPtr = std::shared_ptr<IPassManagerFactory>;

template <typename PassT>
class PassManagerFactory : public IPassManagerFactory {
public:
    ngraph::pass::Manager createManager() const override;
};

template <typename PassT>
ngraph::pass::Manager PassManagerFactory<PassT>::createManager() const
{
    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::InitNodeInfo>();
    manager.register_pass<PassT>();
    return manager;
}

template <typename PassT>
PassManagerFactoryPtr CreatePassManagerFactory()
{
    return std::make_shared<PassManagerFactory<PassT>>();
}

// ----------------------------------------------------------------------------

using TestTuple = std::tuple<ModelPtr /* function */,
                             ModelPtr /* reference_function */,
                             PassManagerFactoryPtr>;

class TransposeSinkingBinaryTestFixture1: public CommonTestUtils::TestsCommon,
                                        public ::testing::WithParamInterface<TestTuple> {
public:
    void SetUp() override;
public:
    ModelPtr model, reference_model;
    ngraph::pass::Manager pass_manager;
};

void TransposeSinkingBinaryTestFixture1::SetUp() {
    // TODO: use auto & [ ... ] = this->GetParam() when C++17
    PassManagerFactoryPtr pass_manager_factory;
    std::tie(model, reference_model, pass_manager_factory) = this->GetParam();
    pass_manager = pass_manager_factory->createManager();
}

namespace {

void execute_test(ModelPtr model,
                  ModelPtr reference_model,
                  ngraph::pass::Manager pass_manager)
{
    ModelPtr original_model = model->clone();

    pass_manager.run_passes(model);
    
    ASSERT_NO_THROW(check_rt_info(model));
    EXPECT_EQ(GetFinalNodeName(model), GetFinalNodeName(original_model));

    FunctionsComparator func_comparator = FunctionsComparator::with_default();
    func_comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    const FunctionsComparator::Result result = func_comparator(model, reference_model);
    ASSERT_TRUE(result.valid) << result.message;
}

} // namespace

// ----------------------------------------------------------------------------

TEST_P(TransposeSinkingBinaryTestFixture1, CompareFunctions) {
    execute_test(model, reference_model, pass_manager);
}

template <typename BinaryT>
TestTuple CreateBinaryBackwardTestTuple(const ov::AxisVector & binary_input_axis_order)
{
    return std::make_tuple(CreateModel(NEW(CreateInput, ov::Shape{1, 4, 4, 1} /* input_shape */),
                                       NEW(CreateConstants,
                                           1 /* constants_num */,
                                           ov::Shape{1, 4, 1, 4} /* constant_shape */),
                                       NEW(AppendInput),
                                       NEW(AppendBinary<BinaryT>, binary_input_axis_order),
                                       NEW(AppendTranpose,
                                           ov::Shape{0, 3, 1, 2} /* transpose_axis_order */,
                                           ov::AxisVector{0} /* input_axis */)),
                           CreateModel(NEW(CreateInput,
                                           ov::Shape{1, 4, 4, 1} /* input_shape */),
                                       NEW(CreateConstants,
                                           1 /* constants_num */,
                                           ov::Shape{1, 4, 1, 4} /* constant_shape */),
                                       NEW(AppendInput),
                                       NEW(AppendTranpose,
                                           ov::Shape{0, 3, 1, 2} /* transpose_axis_order */,
                                           ov::AxisVector{0, 1} /* input_axis */),
                                       NEW(AppendBinary<BinaryT>, binary_input_axis_order)),
                           CreatePassManagerFactory<ngraph::pass::TransposeSinkingBinaryBackward>());
}

template <typename BinaryT>
TestTuple CreateBinaryForwardTestTuple(const ov::AxisVector & binary_input_axis_order)
{
    return std::make_tuple(CreateModel(NEW(CreateInput,
                                           ov::Shape{1, 4, 4, 1} /* input_shape */),
                                       NEW(CreateConstants,
                                           1 /* constants_num */,
                                           ov::Shape{1, 4, 1, 4} /* constant_shape */),
                                       NEW(AppendInputTranpose,
                                           ov::AxisVector{0, 2, 3, 1} /* transpose_axis_order */),
                                       NEW(AppendBinary<BinaryT>, binary_input_axis_order)),
                           CreateModel(NEW(CreateInput,
                                           ov::Shape{1, 4, 4, 1} /* input_shape */),
                                       NEW(CreateConstants,
                                           1 /* constants_num */,
                                           ov::Shape{1, 4, 1, 4} /* constant_shape */),
                                       NEW(AppendInput),
                                       NEW(AppendTranpose,
                                           ov::Shape{0, 3, 1, 2} /* transpose_axis_order */,
                                           ov::AxisVector{0} /* input_axis */),
                                       NEW(AppendBinary<BinaryT>, binary_input_axis_order),
                                       NEW(AppendTranpose,
                                           ov::Shape{0, 2, 3, 1} /* transpose_axis_order */,
                                           ov::AxisVector{0} /* input_axis */)),
                           CreatePassManagerFactory<ngraph::pass::TransposeSinkingBinaryForward>());
}

#define NEW_BINARY_TEST(BinaryT) \
    CreateBinaryBackwardTestTuple<BinaryT>(ov::AxisVector{0, 1} /* binary_input_axis_order */), \
    CreateBinaryBackwardTestTuple<BinaryT>(ov::AxisVector{1, 0} /* binary_input_axis_order */), \
    CreateBinaryForwardTestTuple<BinaryT>(ov::AxisVector{0, 1} /* binary_input_axis_order */), \
    CreateBinaryForwardTestTuple<BinaryT>(ov::AxisVector{1, 0} /* binary_input_axis_order */)

INSTANTIATE_TEST_SUITE_P(TransposeSinkingBinaryTestSuite, TransposeSinkingBinaryTestFixture1,
                         ::testing::Values(
                                            NEW_BINARY_TEST(ov::opset9::Add),
                                            NEW_BINARY_TEST(ov::opset9::Divide),
                                            NEW_BINARY_TEST(ov::opset9::FloorMod),
                                            NEW_BINARY_TEST(ov::opset9::Maximum),
                                            NEW_BINARY_TEST(ov::opset9::Minimum),
                                            NEW_BINARY_TEST(ov::opset9::Mod),
                                            NEW_BINARY_TEST(ov::opset9::Multiply),
                                            NEW_BINARY_TEST(ov::opset9::Power),
                                            NEW_BINARY_TEST(ov::opset9::SquaredDifference),
                                            NEW_BINARY_TEST(ov::opset9::Subtract)
                                          )
                        );

INSTANTIATE_TEST_SUITE_P(TransposeSinkingConcatTestSuite, TransposeSinkingBinaryTestFixture1,
                         ::testing::Values(std::make_tuple(CreateModel(NEW(CreateInput,
                                                                           ov::Shape{1, 4, 4, 1})  /* input_shape */,
                                                                       NEW(CreateConstants,
                                                                           5 /* constants_num */,
                                                                           ov::Shape{1, 4, 1, 4} /* constant_shape */),
                                                                       NEW(AppendInputTranpose,
                                                                           ov::AxisVector{0, 2, 3, 1} /* transpose_axis_order */),
                                                                       NEW(AppendConcat,
                                                                           1 /* concat_axis */,
                                                                           ov::AxisVector{0, 1, 2, 5, 3, 4}) /* input_axis_order */),
                                                           CreateModel(NEW(CreateInput,
                                                                           ov::Shape{1, 4, 4, 1}  /* input_shape */),
                                                                       NEW(CreateConstants,
                                                                           5 /* constants_num */,
                                                                           ov::Shape{1, 4, 1, 4} /* constant_shape */),
                                                                       NEW(AppendTranpose,
                                                                           ov::Shape{0, 3, 1, 2} /* transpose_axis_order */,
                                                                           ov::AxisVector{0, 1, 2, 3, 4} /* input_axis */),
                                                                       NEW(AppendInput),
                                                                       NEW(AppendConcat,
                                                                           2 /* concat_axis */,
                                                                           ov::AxisVector{0, 1, 2, 5, 3, 4} /* input_axis_order */),
                                                                       NEW(AppendTranpose,
                                                                           ov::Shape{0, 2, 3, 1} /* transpose_axis_order */,
                                                                           ov::AxisVector{0} /* input_axis */)),
                                                           CreatePassManagerFactory<ngraph::pass::TransposeSinkingConcatForward>()),
                                           std::make_tuple(CreateModel(NEW(CreateInput,
                                                                           ov::Shape{1, 4, 4, 1}  /* input_shape */),
                                                                       NEW(CreateConstants,
                                                                           5 /* constants_num */,
                                                                           ov::Shape{1, 4, 4, 1} /* constant_shape */),
                                                                       NEW(AppendInput),
                                                                       NEW(AppendConcat,
                                                                           1 /* concat_axis */,
                                                                           ov::AxisVector{0, 1, 2, 5, 3, 4} /* input_axis_order */),
                                                                       NEW(AppendTranpose,
                                                                           ov::Shape{0, 2, 3, 1} /* transpose_axis_order */,
                                                                           ov::AxisVector{0} /* input_axis */)),
                                                           CreateModel(NEW(CreateInput,
                                                                           ov::Shape{1, 4, 4, 1}  /* input_shape */),
                                                                       NEW(CreateConstants,
                                                                           5 /* constants_num */,
                                                                           ov::Shape{1, 4, 4, 1} /* constant_shape */),
                                                                       NEW(AppendInput),
                                                                       NEW(AppendTranpose,
                                                                           ov::Shape{0, 2, 3, 1} /* transpose_axis_order */,
                                                                           ov::AxisVector{0, 1, 2, 3, 4, 5} /* input_axis */),
                                                                       NEW(AppendConcat,
                                                                           3 /* concat_axis */,
                                                                           ov::AxisVector{0, 1, 2, 5, 3, 4} /* input_axis_order */)),
                                                           CreatePassManagerFactory<ngraph::pass::TransposeSinkingConcatBackward>()) 
                                          )
                        );

// --------------------------------------------------------------------------------------

class IBinaryFactory {
public:
    IBinaryFactory() = default;
    virtual ~IBinaryFactory() = default;
    virtual NodePtr create(NodePtr parent_left_node, NodePtr parent_right_node) const = 0;
};

using BinaryFactoryPtr = std::shared_ptr<IBinaryFactory>;

template <typename BinaryT>
class BinaryFactory : public IBinaryFactory {
public:
    BinaryFactory() = default;
    NodePtr create(NodePtr parent_left_node, NodePtr parent_right_node) const override {
        return std::make_shared<BinaryT>(parent_left_node, parent_right_node);
    }
};

template <typename BinaryT>
BinaryFactoryPtr CreateBinaryFactory() {
    return std::make_shared<BinaryFactory<BinaryT>>();
}

// ----------------------------------------------------------------------------

class IPassFactory {
public:
    IPassFactory() = default;
    virtual ~IPassFactory() = default;
    virtual void registerPass(ov::pass::Manager& pass_manager) const = 0;
};

using PassFactoryPtr = std::shared_ptr<IPassFactory>;

template <typename PassT>
class PassFactory : public IPassFactory {
public:
    void registerPass(ov::pass::Manager& pass_manager) const override {
        //pass_manager.register_pass<ngraph::pass::VisualizeTree>("./0before.png"); // DEBUG
        pass_manager.register_pass<PassT>();
        //pass_manager.register_pass<ngraph::pass::VisualizeTree>("./1after.png"); // DEBUG
    }
};

template <typename PassT>
PassFactoryPtr CreatePassFactory() {
    return std::make_shared<PassFactory<PassT>>();
}

}  // namespace


namespace {

std::vector<BinaryFactoryPtr> binary_factories = {
    CreateBinaryFactory<ov::opset9::Add>(),
    CreateBinaryFactory<ov::opset9::Divide>(),
    CreateBinaryFactory<ov::opset9::Maximum>(),
    CreateBinaryFactory<ov::opset9::Minimum>(),
    CreateBinaryFactory<ov::opset9::Mod>(),
    CreateBinaryFactory<ov::opset9::Multiply>(),
    CreateBinaryFactory<ov::opset9::Power>(),
    CreateBinaryFactory<ov::opset9::SquaredDifference>(),
    CreateBinaryFactory<ov::opset9::Subtract>()
};

std::vector<size_t> binary_operations_numbers = {1, 10};

std::vector<size_t> binary_transpose_input_indexes = {0, 1};

} // namespace


namespace binary {
namespace single_consumer {
namespace forward {
namespace one_input_tranpose {

std::shared_ptr<ov::Model> CreateFunction(BinaryFactoryPtr binary_factory,
                                          size_t num_binary_ops,
                                          ov::element::Type input_type,
                                          size_t binary_transpose_input_idx) {
    const ov::Shape input_shape{1, 96, 55, 55};
    const ov::Shape const_shape{1, 55, 55, 96};

    auto X = std::make_shared<ov::opset9::Parameter>(input_type, input_shape);

    auto ng_order0 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<ov::opset9::Transpose>(X, ng_order0);

    NodePtr in_op = transpose0;
    for (size_t i = 0; i < num_binary_ops; ++i) {
        auto in_constant = std::make_shared<ov::opset9::Constant>(input_type, const_shape, ov::Shape{1});
        if (!binary_transpose_input_idx)
            in_op = binary_factory->create(in_op, in_constant);
        else
            in_op = binary_factory->create(in_constant, in_op);
    }

    return std::make_shared<ov::Model>(ov::OutputVector{in_op}, ov::ParameterVector{X});
}

std::shared_ptr<ov::Model> CreateReferenceFunction(BinaryFactoryPtr binary_factory,
                                                   size_t num_binary_ops,
                                                   ov::element::Type input_type,
                                                   size_t binary_transpose_input_idx) {
    const ov::Shape input_shape{1, 96, 55, 55};
    const ov::Shape const_shape{1, 55, 55, 96};

    auto X = std::make_shared<ov::opset9::Parameter>(input_type, input_shape);

    NodePtr in_op = X;
    for (size_t i = 0; i < num_binary_ops; ++i) {
        auto in_constant = std::make_shared<ov::opset9::Constant>(input_type, const_shape, ov::Shape{1});

        auto transpose_reversed_const =
            std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 3, 1, 2});
        auto transpose_reversed = std::make_shared<ov::opset9::Transpose>(in_constant, transpose_reversed_const);

        if (!binary_transpose_input_idx)
            in_op = binary_factory->create(in_op, transpose_reversed);
        else
            in_op = binary_factory->create(transpose_reversed, in_op);
    }

    auto ng_order0 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<ov::opset9::Transpose>(in_op, ng_order0);

    return std::make_shared<ov::Model>(ov::OutputVector{transpose0}, ov::ParameterVector{X});
}

}  // namespace one_input_tranpose

namespace double_transpose {
std::shared_ptr<ov::Model> CreateFunction(BinaryFactoryPtr binary_factory,
                                          size_t num_binary_ops,
                                          ov::element::Type input_type) {
    const ov::Shape input_shape{1, 96, 55, 55};

    auto X = std::make_shared<ov::opset9::Parameter>(input_type, input_shape);

    auto ng_order0 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<ov::opset9::Transpose>(X, ng_order0);

    NodePtr in_op = transpose0;
    for (size_t i = 0; i < num_binary_ops; ++i) {
        auto in_constant = std::make_shared<ov::opset9::Constant>(input_type, input_shape, ov::Shape{1});
        auto ng_order1 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
        auto transpose1 = std::make_shared<ov::opset9::Transpose>(in_constant, ng_order1);

        in_op = binary_factory->create(in_op, transpose1);
    }

    return std::make_shared<ov::Model>(ov::OutputVector{in_op}, ov::ParameterVector{X});
}

std::shared_ptr<ov::Model> CreateReferenceFunction(BinaryFactoryPtr binary_factory,
                                                   size_t num_binary_ops,
                                                   ov::element::Type input_type) {
    const ov::Shape input_shape{1, 96, 55, 55};

    auto X = std::make_shared<ov::opset9::Parameter>(input_type, input_shape);

    NodePtr in_op = X;
    for (size_t i = 0; i < num_binary_ops; ++i) {
        auto in_constant = std::make_shared<ov::opset9::Constant>(input_type, input_shape, ov::Shape{1});

        auto ng_order1 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
        auto transpose1 = std::make_shared<ov::opset9::Transpose>(in_constant, ng_order1);

        auto transpose_reversed_const =
            std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 3, 1, 2});
        auto transpose_reversed = std::make_shared<ov::opset9::Transpose>(transpose1, transpose_reversed_const);

        in_op = binary_factory->create(in_op, transpose_reversed);
    }

    auto ng_order0 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<ov::opset9::Transpose>(in_op, ng_order0);

    return std::make_shared<ov::Model>(ov::OutputVector{transpose0}, ov::ParameterVector{X});
}

}  // namespace double_transpose
} // namespace forward

namespace backward {
namespace one_input_tranpose {
std::shared_ptr<ov::Model> CreateFunction(BinaryFactoryPtr binary_factory,
                                          size_t num_binary_ops,
                                          ov::element::Type input_type,
                                          size_t binary_transpose_input_idx) {
    const ov::Shape input_shape{1, 96, 55, 55};

    auto X = std::make_shared<ov::opset9::Parameter>(input_type, input_shape);

    NodePtr in_op = X;
    for (size_t i = 0; i < num_binary_ops; ++i) {
        auto in_constant = std::make_shared<ov::opset9::Constant>(input_type, input_shape, ov::Shape{1});
        if (!binary_transpose_input_idx)
            in_op = binary_factory->create(in_op, in_constant);
        else
            in_op = binary_factory->create(in_constant, in_op);
    }

    auto ng_order0 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<ov::opset9::Transpose>(in_op, ng_order0);

    return std::make_shared<ov::Model>(ov::OutputVector{transpose0}, ov::ParameterVector{X});
}

std::shared_ptr<ov::Model> CreateReferenceFunction(BinaryFactoryPtr binary_factory,
                                                   size_t num_binary_ops,
                                                   ov::element::Type input_type,
                                                   size_t binary_transpose_input_idx) {
    const ov::Shape input_shape{1, 96, 55, 55};

    auto X = std::make_shared<ov::opset9::Parameter>(input_type, input_shape);

    auto ng_order0 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<ov::opset9::Transpose>(X, ng_order0);

    NodePtr in_op = transpose0;
    for (size_t i = 0; i < num_binary_ops; ++i) {
        auto in_constant = std::make_shared<ov::opset9::Constant>(input_type, input_shape, ov::Shape{1});

        auto ng_order = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
        auto transpose = std::make_shared<ov::opset9::Transpose>(in_constant, ng_order);

        if (!binary_transpose_input_idx)
            in_op = binary_factory->create(in_op, transpose);
        else
            in_op = binary_factory->create(transpose, in_op);
    }

    return std::make_shared<ov::Model>(ov::OutputVector{in_op}, ov::ParameterVector{X});
}
} // namespace one_input_tranpose
} // namespace backward
} // namespace single_consumer
} // namespace binary

using CreateGraphBinaryF = std::function<std::shared_ptr<ov::Model>(BinaryFactoryPtr unary_factory,
                                                                    size_t num_binary_ops,
                                                                    ov::element::Type input_type,
                                                                    size_t binary_transpose_input_idx)>;

using TestBinaryParams = std::tuple<BinaryFactoryPtr,
                                    PassFactoryPtr,
                                    size_t,             /* num_binary_ops */
                                    CreateGraphBinaryF, /* model_factory */
                                    CreateGraphBinaryF, /* reference_model_factory */
                                    ov::element::Type,  /* input type */
                                    size_t>;            /* binary_transpose_input_idx */

class TransposeSinkingBinaryTestFixture : public ::testing::WithParamInterface<TestBinaryParams>,
                                          public TransformationTestsF {};


TEST_P(TransposeSinkingBinaryTestFixture, CompareFunctions) {
    BinaryFactoryPtr unary_factory;
    PassFactoryPtr pass_factory;
    size_t num_binary_ops;
    CreateGraphBinaryF model_factory;
    CreateGraphBinaryF reference_model_factory;
    ov::element::Type input_type;
    size_t binary_transpose_input_idx;
    std::tie(unary_factory,
             pass_factory,
             num_binary_ops,
             model_factory,
             reference_model_factory,
             input_type,
             binary_transpose_input_idx) = this->GetParam();

    model = model_factory(unary_factory, num_binary_ops, input_type, binary_transpose_input_idx);
    model_ref = reference_model_factory(unary_factory, num_binary_ops, input_type, binary_transpose_input_idx);
    pass_factory->registerPass(manager);
}

INSTANTIATE_TEST_SUITE_P(TransposeSinkingBinaryForwardTestSuite, TransposeSinkingBinaryTestFixture,
                         ::testing::Combine(::testing::ValuesIn(binary_factories),
                                            ::testing::Values(CreatePassFactory<ngraph::pass::TransposeSinkingBinaryForward>()),
                                            ::testing::ValuesIn(binary_operations_numbers),
                       ::testing::Values(binary::single_consumer::forward::one_input_tranpose::CreateFunction),
                       ::testing::Values(binary::single_consumer::forward::one_input_tranpose::CreateReferenceFunction),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::ValuesIn(binary_transpose_input_indexes)));

INSTANTIATE_TEST_SUITE_P(
    TransposeSinkingBinaryBackwardTestSuite,
    TransposeSinkingBinaryTestFixture,
                         ::testing::Combine(::testing::ValuesIn(binary_factories),
                                            ::testing::Values(CreatePassFactory<ngraph::pass::TransposeSinkingBinaryBackward>()),
                                            ::testing::ValuesIn(binary_operations_numbers),
                       ::testing::Values(binary::single_consumer::backward::one_input_tranpose::CreateFunction),
                       ::testing::Values(binary::single_consumer::backward::one_input_tranpose::CreateReferenceFunction),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::ValuesIn(binary_transpose_input_indexes)));

// --------------------------------------------------------------------------------------

using CreateGraphBinaryTwoTransposeInputsF = std::function<
    std::shared_ptr<ov::Model>(BinaryFactoryPtr unary_factory, size_t num_binary_ops, ov::element::Type input_type)>;

using TestBinaryTwoTransposeInputsParams = std::tuple<BinaryFactoryPtr,
                                    PassFactoryPtr,
                                    size_t,                                  /* num_binary_ops */
                                    CreateGraphBinaryTwoTransposeInputsF,    /* model_factory */
                                    CreateGraphBinaryTwoTransposeInputsF, /* reference_model_factory */
                                    ov::element::Type>;                      /* input type */

class TransposeSinkingBinaryTwoTransposeInputsTestFixture
    : public ::testing::WithParamInterface<TestBinaryTwoTransposeInputsParams>,
                                          public TransformationTestsF {};

TEST_P(TransposeSinkingBinaryTwoTransposeInputsTestFixture, CompareFunctions) {
    BinaryFactoryPtr unary_factory;
    PassFactoryPtr pass_factory;
    size_t num_binary_ops;
    CreateGraphBinaryTwoTransposeInputsF model_factory;
    CreateGraphBinaryTwoTransposeInputsF reference_model_factory;
    ov::element::Type input_type;

    std::tie(unary_factory, pass_factory, num_binary_ops, model_factory, reference_model_factory, input_type) =
        this->GetParam();

    model = model_factory(unary_factory, num_binary_ops, input_type);
    model_ref = reference_model_factory(unary_factory, num_binary_ops, input_type);
    pass_factory->registerPass(manager);
}

INSTANTIATE_TEST_SUITE_P(
    TransposeSinkingBinaryTwoTransposeInputsForwardTestSuite,
    TransposeSinkingBinaryTwoTransposeInputsTestFixture,
    ::testing::Combine(::testing::ValuesIn(binary_factories),
                       ::testing::Values(CreatePassFactory<ngraph::pass::TransposeSinkingBinaryForward>()),
                       ::testing::ValuesIn(binary_operations_numbers),
                       ::testing::Values(binary::single_consumer::forward::double_transpose::CreateFunction),
                       ::testing::Values(binary::single_consumer::forward::double_transpose::CreateReferenceFunction),
                       ::testing::Values(ov::element::f32)));

// --------------------------------------------------------------------------------------

using CreateGraphConcatF = std::function< std::shared_ptr<ov::Model> (size_t num_concat_ops,
                                          ov::element::Type input_type,
                                          size_t concat_transpose_input_idx,
                                          size_t num_concat_inputs) >;

using TestConcatParams = std::tuple<PassFactoryPtr,
                              size_t, /* num_concat_ops */
                              CreateGraphConcatF, /* model_factory */
                              CreateGraphConcatF, /* reference_model_factory */
                              ov::element::Type, /* input type */
                              size_t, /* concat_transpose_input_idx */
                              size_t>; /* num_concat_inputs */

class TransposeSinkingConcatTestFixture: public ::testing::WithParamInterface<TestConcatParams>,
                                        public TransformationTestsF {};

namespace {

std::vector<size_t> concat_operations_numbers = {1, 10};

std::vector<size_t> concat_transpose_input_indexes = {0, 2};

} // namespace

namespace concat {
namespace single_consumer {
namespace forward {
namespace one_input_tranpose {

std::shared_ptr<ov::Model> CreateFunction(size_t num_concat_ops,
                                          ov::element::Type input_type,
                                          size_t concat_transpose_input_idx,
                                          size_t num_concat_inputs) {
    const ov::Shape input_shape{1, 96, 55, 55};
    const ov::Shape const_shape{1, 55, 55, 96};

    auto X = std::make_shared<ov::opset9::Parameter>(input_type, input_shape);

    auto ng_order0 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<ov::opset9::Transpose>(X, ng_order0);

    NodePtr in_op = transpose0;
    for (size_t i = 0; i < num_concat_ops; ++i) {
        ov::OutputVector concat_inputs;
        for (size_t j = 0; j < num_concat_inputs; ++j) {
            if (j == concat_transpose_input_idx)
                concat_inputs.push_back(in_op);
            else
                concat_inputs.push_back(std::make_shared<ov::opset9::Constant>(input_type, const_shape, ov::Shape{1}));
        }
        in_op = std::make_shared<ov::opset9::Concat>(concat_inputs, 1);
    }

    return std::make_shared<ov::Model>(ov::OutputVector{in_op}, ov::ParameterVector{X});
}

std::shared_ptr<ov::Model> CreateReferenceFunction(size_t num_concat_ops,
                                                   ov::element::Type input_type,
                                                   size_t concat_transpose_input_idx,
                                                   size_t num_concat_inputs) {
    const ov::Shape input_shape{1, 96, 55, 55};
    const ov::Shape const_shape{1, 55, 55, 96};

    auto X = std::make_shared<ov::opset9::Parameter>(input_type, input_shape);

    NodePtr in_op = X;
    for (size_t i = 0; i < num_concat_ops; ++i) {
        ov::OutputVector concat_inputs;
        for (size_t j = 0; j < num_concat_inputs; ++j) {
            if (j == concat_transpose_input_idx) {
                concat_inputs.push_back(in_op);
            } else {
                auto in_constant = std::make_shared<ov::opset9::Constant>(input_type, const_shape, ov::Shape{1});

                auto transpose_reversed_const =
                    std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 3, 1, 2});
                auto transpose_reversed =
                    std::make_shared<ov::opset9::Transpose>(in_constant, transpose_reversed_const);

                concat_inputs.push_back(transpose_reversed);
            }
        }
        in_op = std::make_shared<ov::opset9::Concat>(concat_inputs, 2);
    }

    auto ng_order0 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<ov::opset9::Transpose>(in_op, ng_order0);

    return std::make_shared<ov::Model>(ov::OutputVector{transpose0}, ov::ParameterVector{X});
}

}  // namespace one_input_tranpose

namespace double_transpose {

std::shared_ptr<ov::Model> CreateFunction(size_t num_concat_ops,
                                          ov::element::Type input_type,
                                          size_t num_concat_inputs) {
    const ov::Shape input_shape{1, 96, 55, 55};

    auto X = std::make_shared<ov::opset9::Parameter>(input_type, input_shape);

    auto ng_order0 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<ov::opset9::Transpose>(X, ng_order0);

    NodePtr in_op = transpose0;
    for (size_t i = 0; i < num_concat_ops; ++i) {
        ov::OutputVector concat_inputs;
        concat_inputs.push_back(in_op);
        for (size_t j = 1; j < num_concat_inputs; ++j) {
            auto in_constant = std::make_shared<ov::opset9::Constant>(input_type, input_shape, ov::Shape{1});
            auto ng_order1 =
                std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
            auto transpose1 = std::make_shared<ov::opset9::Transpose>(in_constant, ng_order1);
            concat_inputs.push_back(transpose1);
        }
        in_op = std::make_shared<ov::opset9::Concat>(concat_inputs, 1);
    }

    return std::make_shared<ov::Model>(ov::OutputVector{in_op}, ov::ParameterVector{X});
}

std::shared_ptr<ov::Model> CreateReferenceFunction(size_t num_concat_ops,
                                                   ov::element::Type input_type,
                                                   size_t num_concat_inputs) {
    const ov::Shape input_shape{1, 96, 55, 55};

    auto X = std::make_shared<ov::opset9::Parameter>(input_type, input_shape);

    NodePtr in_op = X;
    for (size_t i = 0; i < num_concat_ops; ++i) {
        ov::OutputVector concat_inputs;

        concat_inputs.push_back(in_op);

        for (size_t j = 1; j < num_concat_inputs; ++j) {
                auto in_constant = std::make_shared<ov::opset9::Constant>(input_type, input_shape, ov::Shape{1});

                auto ng_order1 =
                    std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
                auto transpose1 = std::make_shared<ov::opset9::Transpose>(in_constant, ng_order1);

                auto transpose_reversed_const =
                    std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 3, 1, 2});
                auto transpose_reversed = std::make_shared<ov::opset9::Transpose>(transpose1, transpose_reversed_const);

                concat_inputs.push_back(transpose_reversed);
        }
        in_op = std::make_shared<ov::opset9::Concat>(concat_inputs, 2);
    }

    auto ng_order0 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<ov::opset9::Transpose>(in_op, ng_order0);

    return std::make_shared<ov::Model>(ov::OutputVector{transpose0}, ov::ParameterVector{X});
}

} // namespace double_transpose

} // namespace forward

namespace backward {

std::shared_ptr<ov::Model> CreateFunction(size_t num_concat_ops,
                                          ov::element::Type input_type,
                                          size_t concat_transpose_input_idx,
                                          size_t num_concat_inputs) {
        const ov::Shape input_shape{1, 96, 55, 55};

        auto X = std::make_shared<ov::opset9::Parameter>(input_type, input_shape);

        NodePtr in_op = X;
        for (size_t i = 0; i < num_concat_ops; ++i) {
            ov::OutputVector concat_inputs;
            for (size_t j = 0; j < num_concat_inputs; ++j) {
                if (j == concat_transpose_input_idx)
                    concat_inputs.push_back(in_op);
                else
                    concat_inputs.push_back(std::make_shared<ov::opset9::Constant>(input_type, input_shape, ov::Shape{1}));
            }
            in_op = std::make_shared<ov::opset9::Concat>(concat_inputs, 1);
        }

        auto ng_order0 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
        auto transpose0 = std::make_shared<ov::opset9::Transpose>(in_op, ng_order0);

        return std::make_shared<ov::Model>(ov::OutputVector{transpose0}, ov::ParameterVector{X});
}

std::shared_ptr<ov::Model> CreateReferenceFunction(size_t num_concat_ops,
                                                   ov::element::Type input_type,
                                                   size_t concat_transpose_input_idx,
                                                   size_t num_concat_inputs) {
        const ov::Shape input_shape{1, 96, 55, 55};

        auto X = std::make_shared<ov::opset9::Parameter>(input_type, input_shape);

        auto ng_order0 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
        auto transpose0 = std::make_shared<ov::opset9::Transpose>(X, ng_order0);

        NodePtr in_op = transpose0;
        for (size_t i = 0; i < num_concat_ops; ++i) {
            ov::OutputVector concat_inputs;
            for (size_t j = 0; j < num_concat_inputs; ++j) {
                if (j == concat_transpose_input_idx) {
                    concat_inputs.push_back(in_op);
                } else {
                    auto in_constant = std::make_shared<ov::opset9::Constant>(input_type, input_shape, ov::Shape{1});

                    auto transpose_reversed_const = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
                    auto transpose_reversed = std::make_shared<ov::opset9::Transpose>(in_constant, transpose_reversed_const);

                    concat_inputs.push_back(transpose_reversed);
                }
            }
            in_op = std::make_shared<ov::opset9::Concat>(concat_inputs, 3);
        }

        return std::make_shared<ov::Model>(ov::OutputVector{in_op}, ov::ParameterVector{X});
}

} // namespace backward
} // namespace single_consumer
} // namespace concat

TEST_P(TransposeSinkingConcatTestFixture, CompareFunctions) {
    PassFactoryPtr pass_factory;
    size_t num_concat_ops;
    CreateGraphConcatF model_factory;
    CreateGraphConcatF reference_model_factory;
    ov::element::Type input_type;
    size_t concat_transpose_input_idx;
    size_t num_concat_inputs;
    std::tie(pass_factory,
             num_concat_ops,
             model_factory,
             reference_model_factory,
             input_type,
             concat_transpose_input_idx,
             num_concat_inputs) = this->GetParam();

    model = model_factory(num_concat_ops, input_type, concat_transpose_input_idx, num_concat_inputs);
    model_ref = reference_model_factory(num_concat_ops, input_type, concat_transpose_input_idx, num_concat_inputs);
    pass_factory->registerPass(manager);
}

INSTANTIATE_TEST_SUITE_P(TransposeSinkingConcatForwardTestSuite, TransposeSinkingConcatTestFixture,
                         ::testing::Combine(::testing::Values(CreatePassFactory<ngraph::pass::TransposeSinkingConcatForward>()),
                                            ::testing::ValuesIn(concat_operations_numbers),
                       ::testing::Values(concat::single_consumer::forward::one_input_tranpose::CreateFunction),
                       ::testing::Values(concat::single_consumer::forward::one_input_tranpose::CreateReferenceFunction),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::ValuesIn(concat_transpose_input_indexes),
                                            ::testing::Values(5)));

INSTANTIATE_TEST_SUITE_P(TransposeSinkingConcatBackwardTestSuite, TransposeSinkingConcatTestFixture,
                         ::testing::Combine(::testing::Values(CreatePassFactory<ngraph::pass::TransposeSinkingConcatBackward>()),
                                            ::testing::ValuesIn(concat_operations_numbers),
                                            ::testing::Values(concat::single_consumer::backward::CreateFunction),
                                            ::testing::Values(concat::single_consumer::backward::CreateReferenceFunction),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::ValuesIn(concat_transpose_input_indexes),
                                            ::testing::Values(5)));

// --------------------------------------------------------------------------------------

using CreateGraphConcatAllTransposesInputF = std::function<std::shared_ptr<ov::Model>(size_t num_concat_ops,
                                                                    ov::element::Type input_type,
                                                                    size_t num_concat_inputs)>;

using TestConcatAllTransposesInputParams = std::tuple<PassFactoryPtr,
                                    size_t,             /* num_concat_ops */
                                    CreateGraphConcatAllTransposesInputF, /* model_factory */
                                    CreateGraphConcatAllTransposesInputF, /* reference_model_factory */
                                    ov::element::Type,  /* input type */
                                    size_t>;            /* num_concat_inputs */

class TransposeSinkingConcatAllTransposesInputTestFixture
    : public ::testing::WithParamInterface<TestConcatAllTransposesInputParams>,
                                          public TransformationTestsF {};

TEST_P(TransposeSinkingConcatAllTransposesInputTestFixture, CompareFunctions) {
    PassFactoryPtr pass_factory;
    size_t num_concat_ops;
    CreateGraphConcatAllTransposesInputF model_factory;
    CreateGraphConcatAllTransposesInputF reference_model_factory;
    ov::element::Type input_type;
    size_t num_concat_inputs;
    std::tie(pass_factory,
             num_concat_ops,
             model_factory,
             reference_model_factory,
             input_type,
             num_concat_inputs) = this->GetParam();

    model = model_factory(num_concat_ops, input_type, num_concat_inputs);
    model_ref = reference_model_factory(num_concat_ops, input_type, num_concat_inputs);
    pass_factory->registerPass(manager);
}

INSTANTIATE_TEST_SUITE_P(
    TransposeSinkingConcatForwardAllTransposesTestSuite,
    TransposeSinkingConcatAllTransposesInputTestFixture,
    ::testing::Combine(::testing::Values(CreatePassFactory<ngraph::pass::TransposeSinkingConcatForward>()),
                       ::testing::ValuesIn(concat_operations_numbers),
                       ::testing::Values(concat::single_consumer::forward::double_transpose::CreateFunction),
                       ::testing::Values(concat::single_consumer::forward::double_transpose::CreateReferenceFunction),
                       ::testing::Values(ov::element::f32),
                       ::testing::Values(5)));

// --------------------------------------------------------------------------------------

using CreateGraphSplitForwardF = std::function< std::shared_ptr<ov::Model> (size_t num_split_ops,
                                                                            size_t num_split_outputs,
                                                                            ov::element::Type input_type)>;

using TestSplitForwardParams = std::tuple<PassFactoryPtr,
                              size_t, /* num_split_ops */
                              size_t, /* num_split_outputs */
                              CreateGraphSplitForwardF, /* model_factory */
                              CreateGraphSplitForwardF, /* reference_model_factory */
                              ov::element::Type> /* input type */;

class TransposeSinkingSplitForwardTestFixture: public ::testing::WithParamInterface<TestSplitForwardParams>,
                                        public TransformationTestsF {};

namespace {

std::vector<size_t> split_operations_numbers = {1, 10};

std::vector<size_t> split_outputs_numbers = {2, 5};

} // namespace

// --------------------------------------------------------------------------------------

namespace split {
namespace forward {
std::shared_ptr<ov::Model> CreateFunction(size_t num_split_ops,
                                          size_t num_split_outputs,
                                          ov::element::Type input_type) {
        const ov::Shape input_shape{96, static_cast<size_t>(std::pow(num_split_outputs, num_split_ops + 1)), 55, 55};

        auto X = std::make_shared<ov::opset9::Parameter>(input_type, input_shape);

        auto ng_order0 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 3, 1, 2});
        auto transpose0 = std::make_shared<ov::opset9::Transpose>(X, ng_order0);

        ov::OutputVector outputs;
        Output in_op = transpose0->output(0);
        for (size_t i = 0; i < num_split_ops; ++i) {
            auto split_axis_const = std::make_shared<ov::opset9::Constant>(ov::element::u64,
                                                                           ov::Shape{},
                                                                           2);
            auto split = std::make_shared<ov::opset9::Split>(in_op, split_axis_const, num_split_outputs);
            for (size_t num_output = 0; num_output < num_split_outputs - 1; ++num_output) {
                outputs.push_back(split->output(num_output));
            }
            in_op = split->output(num_split_outputs - 1);
        }
        outputs.push_back(in_op);

        return std::make_shared<ov::Model>(outputs, ov::ParameterVector{X});
}

std::shared_ptr<ov::Model> CreateReferenceFunction(size_t num_split_ops,
                                                   size_t num_split_outputs,
                                                   ov::element::Type input_type) {
        const ov::Shape input_shape{96, static_cast<size_t>(std::pow(num_split_outputs, num_split_ops + 1)), 55, 55};

        auto X = std::make_shared<ov::opset9::Parameter>(input_type, input_shape);

        ov::OutputVector outputs;
        Output in_op = X->output(0);
        for (size_t i = 0; i < num_split_ops; ++i) {
            auto split_axis_const = std::make_shared<ov::opset9::Constant>(ov::element::u64,
                                                                           ov::Shape{},
                                                                           1);
            auto split = std::make_shared<ov::opset9::Split>(in_op, split_axis_const, num_split_outputs);
            for (size_t num_output = 0; num_output < num_split_outputs - 1; ++num_output) {
                auto ng_order0 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 3, 1, 2});
                auto transpose0 = std::make_shared<ov::opset9::Transpose>(split->output(num_output), ng_order0);
                outputs.push_back(transpose0);
            }
            in_op = split->output(num_split_outputs - 1);
        }

        auto ng_order0 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 3, 1, 2});
        auto transpose0 = std::make_shared<ov::opset9::Transpose>(in_op, ng_order0);
        outputs.push_back(transpose0);

        return std::make_shared<ov::Model>(outputs, ov::ParameterVector{X});
}

} // namespace forward
} // namespace split

TEST_P(TransposeSinkingSplitForwardTestFixture, CompareFunctions) {
    PassFactoryPtr pass_factory;
    size_t num_split_ops;
    size_t num_split_outputs;
    CreateGraphSplitForwardF model_factory;
    CreateGraphSplitForwardF reference_model_factory;
    ov::element::Type input_type;
    std::tie(pass_factory,
             num_split_ops,
             num_split_outputs,
             model_factory,
             reference_model_factory,
             input_type) = this->GetParam();

    model = model_factory(num_split_ops, num_split_outputs, input_type);
    model_ref = reference_model_factory(num_split_ops, num_split_outputs, input_type);
    pass_factory->registerPass(manager);
}

INSTANTIATE_TEST_SUITE_P(
    TransposeSinkingSplitForwardTestSuite,
    TransposeSinkingSplitForwardTestFixture,
    ::testing::Combine(::testing::Values(CreatePassFactory<ngraph::pass::TransposeSinkingSplitForward>()),
                       ::testing::ValuesIn(split_operations_numbers),
                       ::testing::ValuesIn(split_outputs_numbers),
                       ::testing::Values(split::forward::CreateFunction),
                       ::testing::Values(split::forward::CreateReferenceFunction),
                       ::testing::Values(ov::element::f32)));

// --------------------------------------------------------------------------------------

// TODO: TestSuite class

// --------------------------------------------------------------------------------------

namespace split {
namespace backward {
std::shared_ptr<ov::Model> CreateFunction(size_t num_split_ops,
                                          size_t num_split_outputs,
                                          std::set<size_t> transpose_output_indexes,
                                          ov::element::Type input_type) {
        const ov::Shape input_shape{96, static_cast<size_t>(std::pow(num_split_outputs, num_split_ops + 1)), 55, 55};

        auto X = std::make_shared<ov::opset9::Parameter>(input_type, input_shape);

        ov::OutputVector outputs;
        Output in_op = X->output(0);
        for (size_t i = 0; i < num_split_ops; ++i) {
            auto split_axis_const = std::make_shared<ov::opset9::Constant>(ov::element::u64,
                                                                           ov::Shape{},
                                                                           1);
            auto split = std::make_shared<ov::opset9::Split>(in_op, split_axis_const, num_split_outputs);
            for (size_t num_output = 0; num_output < num_split_outputs - 1; ++num_output) {
                outputs.push_back(split->output(num_output));
            }
            in_op = split->output(num_split_outputs - 1);
        }
        outputs.push_back(in_op);

        for (size_t idx = 0; idx < outputs.size(); ++idx) {
            if (transpose_output_indexes.find(idx) == transpose_output_indexes.end())
                continue;
            const size_t output_idx = outputs.size() - num_split_outputs - 1 + idx;

            auto ng_order = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 3, 1, 2});
            outputs[output_idx] = std::make_shared<ov::opset9::Transpose>(outputs[output_idx], ng_order);
        }

        return std::make_shared<ov::Model>(outputs, ov::ParameterVector{X});
}

std::shared_ptr<ov::Model> CreateReferenceFunction(size_t num_split_ops,
                                                   size_t num_split_outputs,
                                                   std::set<size_t> no_transpose_output_indexes,
                                                   ov::element::Type input_type) {
        const ov::Shape input_shape{96, static_cast<size_t>(std::pow(num_split_outputs, num_split_ops + 1)), 55, 55};

        auto X = std::make_shared<ov::opset9::Parameter>(input_type, input_shape);

        auto ng_order0 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 3, 1, 2});
        auto transpose0 = std::make_shared<ov::opset9::Transpose>(X, ng_order0);

        ov::OutputVector outputs;
        Output in_op = transpose0->output(0);
        for (size_t i = 0; i < num_split_ops - 1; ++i) {
            auto split_axis_const = std::make_shared<ov::opset9::Constant>(ov::element::u64,
                                                                           ov::Shape{},
                                                                           2);
            auto split = std::make_shared<ov::opset9::Split>(in_op, split_axis_const, num_split_outputs);
            for (size_t num_output = 0; num_output < num_split_outputs - 1; ++num_output) {
                auto ng_order0 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
                auto transpose0 = std::make_shared<ov::opset9::Transpose>(split->output(num_output), ng_order0);
                outputs.push_back(transpose0);
            }
            in_op = split->output(num_split_outputs - 1);
        }

        auto split_axis_const = std::make_shared<ov::opset9::Constant>(ov::element::u64,
                                                                           ov::Shape{},
                                                                           2);
        auto last_split = std::make_shared<ov::opset9::Split>(in_op, split_axis_const, num_split_outputs);

        for (size_t output_idx = 0; output_idx < num_split_outputs; ++output_idx) {
            if (no_transpose_output_indexes.find(output_idx) == no_transpose_output_indexes.end()) {
                auto ng_order = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
                auto transpose = std::make_shared<ov::opset9::Transpose>(last_split->output(output_idx), ng_order);

                auto ng_order1 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 3, 1, 2});
                outputs.push_back(std::make_shared<ov::opset9::Transpose>(transpose, ng_order1));
            } else {
                outputs.push_back(last_split->output(output_idx));
            }
        }

        return std::make_shared<ov::Model>(outputs, ov::ParameterVector{X});
}

} // namespace backward
} // namespace split

using FloatPtr = std::unique_ptr<float[]>;

template <class IterT, class T>
void Fill(IterT begin_iter, IterT end_iter, T value, T step) {
    while (begin_iter != end_iter) {
        *begin_iter = value;
        value += step;
        ++begin_iter;
    }
}

FloatPtr GenerateTestInput(const ov::Shape & input_shape) {
    const size_t size = std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<size_t>());

    FloatPtr input(new float[size]);
    Fill(input.get(), input.get() + size, 0.01, 0.01);

    return input;
}

#define EMUTEX_DEBUG_CHECKPOINT std::cout << "[EMUTEX DEBUG] CHECKPOINT " << __FILE__ << ":" << __LINE__ << std::endl;

TEST(TransposeSinkingSplitTests, SplitForward) {
    const size_t num_split_ops = 2;
    const size_t num_split_outputs = 2;

    const ov::element::Type input_type = ov::element::f32;
    const ov::Shape input_shape{96, (1 << (num_split_ops + 1)), 55, 55};

    std::cout << "input shape " << input_shape << std::endl;

    EMUTEX_DEBUG_CHECKPOINT;

    ModelPtr model, original_model, reference_model;
    {
        model = split::backward::CreateFunction(num_split_ops, num_split_outputs, std::set<size_t>{2}, input_type);
        original_model = model->clone();

        EMUTEX_DEBUG_CHECKPOINT;

        ngraph::pass::Manager pass_manager;
        pass_manager.register_pass<ngraph::pass::InitNodeInfo>();
        pass_manager.register_pass<ngraph::pass::VisualizeTree>("./0before.png"); // DEBUG
        pass_manager.register_pass<ngraph::pass::TransposeSinkingSplitBackward>();
        pass_manager.register_pass<ngraph::pass::VisualizeTree>("./1after.png"); // DEBUG
        pass_manager.run_passes(model);
        ASSERT_NO_THROW(check_rt_info(model));
    }
    EMUTEX_DEBUG_CHECKPOINT;
    {
        reference_model = split::backward::CreateReferenceFunction(num_split_ops, num_split_outputs, std::set<size_t>{2}, input_type);
    }

    EMUTEX_DEBUG_CHECKPOINT;

    FunctionsComparator func_comparator = FunctionsComparator::with_default();
    func_comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    const FunctionsComparator::Result result = func_comparator(model, reference_model);
    ASSERT_TRUE(result.valid) << result.message;

    EMUTEX_DEBUG_CHECKPOINT;

    const size_t num_outputs = num_split_ops + 1;

    FloatPtr test_input = GenerateTestInput(input_shape);
    ov::Tensor input_tensor{input_type, input_shape, test_input.get()};
    ov::TensorVector function_result(num_outputs), reference_function_result(num_outputs);
    ASSERT_TRUE(original_model->evaluate(function_result, ov::TensorVector{input_tensor}));
    EXPECT_EQ(function_result.size(), num_outputs);

    EMUTEX_DEBUG_CHECKPOINT;

    for (size_t result_idx = 0; result_idx < num_outputs; ++result_idx) {
        EXPECT_EQ(function_result[result_idx].get_element_type(), ngraph::element::f32);

        ASSERT_TRUE(model->evaluate(reference_function_result, ov::TensorVector{input_tensor}));
        EXPECT_EQ(reference_function_result.size(), num_outputs);
        EXPECT_EQ(reference_function_result[result_idx].get_element_type(), ngraph::element::f32);

        EXPECT_EQ(reference_function_result[result_idx].get_shape(), function_result[result_idx].get_shape());
        EXPECT_EQ(reference_function_result[result_idx].get_size(), function_result[result_idx].get_size());

        const float * function_result_data = function_result[result_idx].data<float>();
        const float * reference_function_result_data = reference_function_result[result_idx].data<float>();
        for (size_t i = 0; i < reference_function_result[result_idx].get_size(); ++i)
            EXPECT_EQ(function_result_data[i], reference_function_result_data[i]);
    }

    EMUTEX_DEBUG_CHECKPOINT;
}
