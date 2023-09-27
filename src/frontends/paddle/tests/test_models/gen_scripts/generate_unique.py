# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# unique paddle model generator
#
import numpy as np
from save_model import saveModel
import paddle
import sys

data_type = "float32"


def unique(
    name: str,
    x,
    reture_index=False,
    reture_inverse=False,
    reture_counts=False,
    dtype="int64",
    axes=None,
):
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name="x", shape=x.shape, dtype=data_type)

        unique_outs = paddle.unique(
            node_x,
            return_index=reture_index,
            return_inverse=reture_inverse,
            return_counts=reture_counts,
            dtype=dtype,
            axis=axes,
        )
        if reture_counts or reture_inverse or reture_index:
            outputs = []
            for out in unique_outs:
                if out is not None:
                    if out.dtype == paddle.int64 or out.dtype == paddle.int32:
                        out = paddle.cast(out, "float32")
                outputs.append(out)
        else:
            outputs = [unique_outs]

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        fetch_vars = [x for x in outputs if x is not None]

        outs = exe.run(feed={"x": x}, fetch_list=fetch_vars)

        saveModel(
            name,
            exe,
            feedkeys=["x"],
            fetchlist=fetch_vars,
            inputs=[x],
            outputs=outs,
            target_dir=sys.argv[1],
        )


def main():
    data = np.array([2, 3, 3, 1, 5, 3]).astype(data_type)
    unique("unique", data)
    unique("unique_ret_index", data, reture_index=True)
    unique("unique_ret_inverse", data, reture_inverse=True)
    unique("unique_ret_counts", data, reture_counts=True)
    unique("unique_ret_index_inverse", data, reture_index=True, reture_inverse=True)
    unique("unique_ret_index_counts", data, reture_index=True, reture_counts=True)
    unique("unique_ret_inverse_counts", data, reture_inverse=True, reture_counts=True)
    unique(
        "unique_ret_index_inverse_counts",
        data,
        reture_index=True,
        reture_inverse=True,
        reture_counts=True,
    )

    data = np.array([[2, 1, 3], [3, 0, 1], [2, 1, 3]]).astype(data_type)
    unique("unique_ret_index_axis", data, reture_index=True, axes=0)
    unique("unique_ret_index_i32", data, reture_index=True, dtype="int32")


if __name__ == "__main__":
    main()
