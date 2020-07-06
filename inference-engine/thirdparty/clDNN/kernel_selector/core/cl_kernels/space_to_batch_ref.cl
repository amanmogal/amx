// Copyright (c) 2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#include "include/include_all.cl"

KERNEL(space_to_batch_ref)(const __global INPUT0_TYPE* input, __global OUTPUT_TYPE* output)
{
    const uint batch = get_global_id(0);
    const uint feature = get_global_id(1);

#ifdef OUTPUT_LAYOUT_BFYX
    const uint w = 0;
    const uint z = 0;
    const uint y = (uint)get_global_id(2) / OUTPUT_SIZE_X;
    const uint x = (uint)get_global_id(2) % OUTPUT_SIZE_X;
#elif OUTPUT_LAYOUT_BFZYX
    const uint w = 0;
    const uint yx = (uint)get_global_id(2) % (OUTPUT_SIZE_X * OUTPUT_SIZE_Y);
    const uint z = (uint)get_global_id(2) / (OUTPUT_SIZE_X * OUTPUT_SIZE_Y);
    const uint y = yx / OUTPUT_SIZE_X;
    const uint x = yx % OUTPUT_SIZE_X;
#elif OUTPUT_LAYOUT_BFWZYX
    const uint zyx = (uint)get_global_id(2) % (OUTPUT_SIZE_X * OUTPUT_SIZE_Y * OUTPUT_SIZE_Z);
    const uint w = (uint)get_global_id(2) / (OUTPUT_SIZE_X * OUTPUT_SIZE_Y * OUTPUT_SIZE_Z);
    const uint yx = zyx % (OUTPUT_SIZE_X * OUTPUT_SIZE_Y);
    const uint z = zyx / (OUTPUT_SIZE_X * OUTPUT_SIZE_Y);
    const uint y = yx / OUTPUT_SIZE_X;
    const uint x = yx % OUTPUT_SIZE_X;
#endif

    const uint input_batch = batch % INPUT0_BATCH_NUM;
    const uint offset_batch =  batch / INPUT0_BATCH_NUM;

    const int input_feature = feature * BLOCK_SHAPE_FEATURE - PADS_BEGIN_FEATURE +
                              offset_batch / (BLOCK_SHAPE_W * BLOCK_SHAPE_Z * BLOCK_SHAPE_Y * BLOCK_SHAPE_X);
    const uint offset_feature = offset_batch % (BLOCK_SHAPE_W * BLOCK_SHAPE_Z * BLOCK_SHAPE_Y * BLOCK_SHAPE_X);

    const int input_w = w * BLOCK_SHAPE_W - PADS_BEGIN_W + offset_feature / (BLOCK_SHAPE_Z * BLOCK_SHAPE_Y * BLOCK_SHAPE_X);
    const uint offset_w = offset_feature % (BLOCK_SHAPE_Z * BLOCK_SHAPE_Y * BLOCK_SHAPE_X);

    const int input_z = z * BLOCK_SHAPE_Z - PADS_BEGIN_Z + offset_w / (BLOCK_SHAPE_Y * BLOCK_SHAPE_X);
    const uint offset_z = offset_w % (BLOCK_SHAPE_Y * BLOCK_SHAPE_X);

    const int input_y = y * BLOCK_SHAPE_Y - PADS_BEGIN_Y + offset_z / BLOCK_SHAPE_X;
    const uint offset_y = offset_z % BLOCK_SHAPE_X;

    const int input_x = x * BLOCK_SHAPE_X - PADS_BEGIN_X + offset_y;

    const int input_index = GET_DATA_INDEX_6D(INPUT0, input_batch, input_feature, input_w, input_z, input_y, input_x);

    const uint output_index = GET_DATA_INDEX_6D(OUTPUT, batch, feature, w, z, y, x);

    const bool out_of_bounds = input_feature < 0 || input_feature >= INPUT0_FEATURE_NUM ||
                               input_w < 0 || input_w >= INPUT0_SIZE_W ||
                               input_z < 0 || input_z >= INPUT0_SIZE_Z ||
                               input_y < 0 || input_y >= INPUT0_SIZE_Y ||
                               input_x < 0 || input_x >= INPUT0_SIZE_X;

    INPUT0_TYPE in = out_of_bounds ? INPUT0_VAL_ZERO : input[input_index];
    output[output_index] = ACTIVATION(in, ACTIVATION_PARAMS);
}
