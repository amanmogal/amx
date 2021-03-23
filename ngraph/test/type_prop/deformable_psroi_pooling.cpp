//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, deformable_psroi_pooling_no_offsets_group_size_3)
{
    const float spatial_scale = 0.0625;
    const int64_t output_dim = 882;
    const int64_t group_size = 3;

    const auto rois_dim = 300;

    auto input_data = make_shared<op::Parameter>(element::f32, PartialShape{2, 7938, 63, 38});
    auto input_coords = make_shared<op::Parameter>(element::f32, PartialShape{rois_dim, 5});

    auto def_psroi_pool = make_shared<op::v1::DeformablePSROIPooling>(
        input_data, input_coords, output_dim, spatial_scale, group_size);

    const PartialShape expected_output{rois_dim, output_dim, group_size, group_size};
    ASSERT_EQ(def_psroi_pool->get_output_partial_shape(0), expected_output);
}

TEST(type_prop, deformable_psroi_pooling_group_size_3)
{
    const float spatial_scale = 0.0625;
    const int64_t output_dim = 882;
    const int64_t group_size = 3;
    const int64_t part_size = 3;
    const double spatial_bins = 4;

    const auto rois_dim = 300;

    auto input_data = make_shared<op::Parameter>(element::f32, PartialShape{2, 7938, 63, 38});
    auto input_coords = make_shared<op::Parameter>(element::f32, PartialShape{rois_dim, 5});
    auto input_offsets = make_shared<op::Parameter>(element::f32, PartialShape{rois_dim, 2, part_size, part_size});

    auto def_psroi_pool = make_shared<op::v1::DeformablePSROIPooling>(
        input_data, input_coords, input_offsets, output_dim, spatial_scale, group_size, "bilinear_deformable", spatial_bins, spatial_bins, 0.1, part_size);

    const PartialShape expected_output{rois_dim, output_dim, group_size, group_size};
    ASSERT_EQ(def_psroi_pool->get_output_partial_shape(0), expected_output);
}

TEST(type_prop, deformable_psroi_pooling_group_size_7)
{
    const float spatial_scale = 0.0625;
    const int64_t output_dim = 162;
    const int64_t group_size = 7;
    const int64_t part_size = 7;
    const double spatial_bins = 4;

    const auto rois_dim = 300;

    auto input_data = make_shared<op::Parameter>(element::f32, PartialShape{2, 7938, 63, 38});
    auto input_coords = make_shared<op::Parameter>(element::f32, PartialShape{rois_dim, 5});
    auto input_offsets = make_shared<op::Parameter>(element::f32, PartialShape{rois_dim, 2, part_size, part_size});

   auto def_psroi_pool = make_shared<op::v1::DeformablePSROIPooling>(
        input_data, input_coords, input_offsets, output_dim, spatial_scale, group_size, "bilinear_deformable", spatial_bins, spatial_bins, 0.1, part_size);

    const PartialShape expected_output{rois_dim, output_dim, group_size, group_size};
    ASSERT_EQ(def_psroi_pool->get_output_partial_shape(0), expected_output);
}

TEST(type_prop, deformable_psroi_pooling_dynamic_rois)
{
    const float spatial_scale = 0.0625;
    const int64_t output_dim = 882;
    const int64_t group_size = 3;

    const auto rois_dim = Dimension(100, 200);

    auto input_data = make_shared<op::Parameter>(element::f32, PartialShape{2, 7938, 63, 38});
    auto input_coords = make_shared<op::Parameter>(element::f32, PartialShape{rois_dim, 5});

    auto def_psroi_pool = make_shared<op::v1::DeformablePSROIPooling>(
        input_data, input_coords, output_dim, spatial_scale, group_size);

    const PartialShape expected_output{rois_dim, output_dim, group_size, group_size};
    ASSERT_EQ(def_psroi_pool->get_output_partial_shape(0), expected_output);
}

TEST(type_prop, deformable_psroi_pooling_fully_dynamic)
{
    const float spatial_scale = 0.0625;
    const int64_t output_dim = 882;
    const int64_t group_size = 3;

    const auto rois_dim = Dimension::dynamic();

    auto input_data = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto input_coords = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());

    auto def_psroi_pool = make_shared<op::v1::DeformablePSROIPooling>(
        input_data, input_coords, output_dim, spatial_scale, group_size);

    const PartialShape expected_output{rois_dim, output_dim, group_size, group_size};
    ASSERT_EQ(def_psroi_pool->get_output_partial_shape(0), expected_output);
}

TEST(type_prop, deformable_psroi_pooling_invalid_group_size)
{
    const float spatial_scale = 0.0625;
    const int64_t output_dim = 882;
    const auto rois_dim = 300;
    try
    {
        const int64_t group_size = 0;

        auto input_data = make_shared<op::Parameter>(element::f32, PartialShape{2, 7938, 63, 38});
        auto input_coords = make_shared<op::Parameter>(element::f32, PartialShape{rois_dim, 5});
        auto def_psroi_pool = make_shared<op::v1::DeformablePSROIPooling>(
            input_data, input_coords, output_dim, spatial_scale, group_size);

        FAIL() << "Ivalid group_size not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("group_size has to be greater than 0"));
    }
    catch (...)
    {
        FAIL() << "Unknown exception was thrown";
    }
}

TEST(type_prop, deformable_psroi_pooling_invalid_data_input_rank)
{
    const float spatial_scale = 0.0625;
    const int64_t output_dim = 162;
    const int64_t group_size = 7;
    const int64_t part_size = 7;
    const double spatial_bins = 4;

    const auto rois_dim = 300;

    auto input_data = make_shared<op::Parameter>(element::f32, PartialShape{7938, 63, 38});
    auto input_coords = make_shared<op::Parameter>(element::f32, PartialShape{rois_dim, 5});
    auto input_offsets = make_shared<op::Parameter>(element::f32, PartialShape{rois_dim, 2, part_size, part_size});

    try
    {
      auto def_psroi_pool = make_shared<op::v1::DeformablePSROIPooling>(
        input_data, input_coords, input_offsets, output_dim, spatial_scale, group_size, "bilinear_deformable", spatial_bins, spatial_bins, 0.1, part_size);

        // Should have thrown, so fail if it didn't
        FAIL() << "Ivalid feature map input rank not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Feature map input rank must equal to 4 (input rank: 3)"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, deformable_psroi_pooling_invalid_box_coordinates_rank)
{
    const int64_t output_dim = 4;
    const float spatial_scale = 0.9;
    const int64_t group_size = 7;
    
    const auto rois_dim = 300;

    auto input_data = make_shared<op::Parameter>(element::f32, PartialShape{2, 7938, 63, 38});
    auto input_coords = make_shared<op::Parameter>(element::f32, PartialShape{2, rois_dim, 5});
    try
    {
        auto def_psroi_pool = make_shared<op::v1::DeformablePSROIPooling>(
            input_data, input_coords, output_dim, spatial_scale, group_size);
        // Should have thrown, so fail if it didn't
        FAIL() << "Ivalid box coordinates input rank not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Box coordinates input rank must equal to 2 (input rank: 3)"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, deformable_psroi_pooling_invalid_offstes_rank)
{
    const float spatial_scale = 0.0625;
    const int64_t output_dim = 162;
    const int64_t group_size = 7;
    const int64_t part_size = 7;
    const double spatial_bins = 4;

    const auto rois_dim = 300;

    auto input_data = make_shared<op::Parameter>(element::f32, PartialShape{2, 7938, 63, 38});
    auto input_coords = make_shared<op::Parameter>(element::f32, PartialShape{rois_dim, 5});
    auto input_offsets = make_shared<op::Parameter>(element::f32, PartialShape{2, rois_dim, 2, part_size, part_size});
    try
    {
      auto def_psroi_pool = make_shared<op::v1::DeformablePSROIPooling>(
        input_data, input_coords, input_offsets, output_dim, spatial_scale, group_size, "bilinear_deformable", spatial_bins, spatial_bins, 0.1, part_size);

       // Should have thrown, so fail if it didn't
        FAIL() << "Offsets input rank not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Offsets input rank must equal to 4 (input rank: 5)"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}
