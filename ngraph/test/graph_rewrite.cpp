// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/pass/manager.hpp>
#include <util/test_tools.hpp>

using namespace ::testing;
using namespace std;
using namespace ngraph;

class TestPass : public ngraph::pass::MatcherPass
{
public:
    TestPass()
        : MatcherPass()
    {
        auto divide = std::make_shared<ngraph::pattern::op::Label>(
            element::f32, Shape{}, pattern::has_class<opset3::Divide>());
        ngraph::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
            if (m_transformation_callback(m.get_match_root()))
            {
                auto relu =
                    std::make_shared<ngraph::opset3::Relu>(m.get_match_root()->input_value(0));
                ngraph::replace_node(m.get_match_root(), relu);
                return true;
            }
            return false;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(divide, "TestMatcher");
        this->register_matcher(m, callback);
    }
};

class Anchor : public ngraph::pass::GraphRewrite
{
public:
    Anchor()
        : GraphRewrite()
    {
    }
};

std::shared_ptr<Function> get_function()
{
    auto data =
        std::make_shared<ngraph::opset3::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});
    auto divide_constant =
        ngraph::opset3::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {1.5});
    auto divide = std::make_shared<ngraph::opset3::Divide>(data, divide_constant);
    return std::make_shared<ngraph::Function>(ngraph::NodeVector{divide},
                                              ngraph::ParameterVector{data});
}

ngraph::pass::param_callback get_callback()
{
    return [](const std::shared_ptr<const Node>& node) -> bool {
        if (std::dynamic_pointer_cast<const opset3::Divide>(node))
        {
            return true;
        }
        else
        {
            return false;
        }
    };
}

TEST(GraphRewriteTest, MatcherPassCallback)
{
    auto f = get_function();

    Anchor anchor;
    anchor.add_matcher<TestPass>()->set_callback(get_callback());
    anchor.run_on_function(f);

    ASSERT_EQ(count_ops_of_type<opset3::Relu>(f), 1);
}

TEST(GraphRewriteTest, GraphRewriteCallback)
{
    auto f = get_function();

    Anchor anchor;
    anchor.add_matcher<TestPass>();
    anchor.set_callback(get_callback());
    anchor.run_on_function(f);

    ASSERT_EQ(count_ops_of_type<opset3::Relu>(f), 1);
}

TEST(GraphRewriteTest, ManagerCallback)
{
    auto f = get_function();

    pass::Manager manager;
    auto anchor = manager.register_pass<Anchor>();
    anchor->add_matcher<TestPass>();
    manager.set_callback(get_callback());
    manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<opset3::Relu>(f), 1);
}

TEST(GraphRewriteTest, ManagerCallback2)
{
    auto f = get_function();

    pass::Manager manager;
    auto anchor = manager.register_pass<TestPass>();
    manager.set_callback(get_callback());
    manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<opset3::Relu>(f), 1);
}
