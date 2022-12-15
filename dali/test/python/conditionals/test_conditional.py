# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from nvidia.dali.pipeline import pipeline_def, Pipeline, experimental
import nvidia.dali.fn as fn
import nvidia.dali
import nvidia.dali.types as types
from nvidia.dali.data_node import _arithm_op

import numpy as np

from test_utils import check_batch, RandomlyShapedDataIterator
from nose_utils import assert_raises
from nose2.tools import params

import itertools

test_iters = 4

from nvidia.dali._autograph.utils.ag_logging import set_verbosity

set_verbosity(10, True)

def consumer(input):
    output = input
    return output


def to_batch(tl, batch_size):
    return [np.array(tl[i]) for i in range(batch_size)]


@pipeline_def
def rotate_pipe(dev):
    input = fn.external_source(name="input", device=dev)
    return fn.rotate(input, angle=15)


@pipeline_def
def flip_pipe(dev):
    input = fn.external_source(name="input", device=dev)
    return fn.flip(input, horizontal=True)


# @experimental.pipeline_def(enable_conditionals=True)
# def conditional_split_merge_pipe(dev):
#     input = fn.external_source(name="input", device=dev)
#     input2 = input
#     def modify_nonlocal():
#         nonlocal input2
#         input2 = input2 + 1
#     pred = fn.external_source(name="predicate")
#     if pred:
#         output = fn.rotate(input, angle=15)
#         modify_nonlocal()
#         input2 = input + 1
#         x = consumer(input)
#     else:
#         output = fn.flip(input, horizontal=True)
#         modify_nonlocal()
#         x = consumer(input)
#     return output, x, input2

def wrap_rotate(input, angle):
    input = input
    angle = angle
    return fn.rotate(input, angle=angle)


@experimental.pipeline_def(enable_conditionals=True)
def conditional_split_merge_pipe(dev):
    input = fn.external_source(name="input", device=dev)
    pred = fn.external_source(name="predicate")

    def wrap_flip(horizontal):
        nonlocal input
        horizontal = horizontal
        return fn.flip(input, horizontal=horizontal)

    if pred:
        output = fn.rotate(input, angle=15)
    else:
        output = wrap_flip(horizontal=True)
    return output


def check_conditional_split_merge(dev, pred_gen):
    bs = 10
    kwargs = {
        "batch_size": bs,
        "num_threads": 4,
        "device_id": 0,
        "prefetch_queue_depth": 1  # so that it's easier to use external source
    }
    pipe_sm = conditional_split_merge_pipe(dev, **kwargs)
    pipe_true = rotate_pipe(dev, **kwargs)
    pipe_false = flip_pipe(dev, **kwargs)
    pipe_sm.build()
    pipe_true.build()
    pipe_false.build()
    data_iter = RandomlyShapedDataIterator(bs, min_shape=(20, 20, 3), max_shape=(40, 30, 3))
    data_iter = iter(data_iter)
    for _ in range(test_iters):
        predicate = [pred_gen(i) for i in range(bs)]
        data = next(data_iter)
        data_true = [data[i] for i in range(bs) if predicate[i]]
        data_false = [data[i] for i in range(bs) if not predicate[i]]
        pipe_sm.feed_input("input", data)
        pipe_sm.feed_input("predicate", predicate)
        if data_true:
            pipe_true.feed_input("input", data_true)
            out_true, = pipe_true.run()
        else:
            out_true = []
        if data_false:
            pipe_false.feed_input("input", data_false)
            out_false, = pipe_false.run()
        else:
            out_false = []
        out, = pipe_sm.run()
        out_baseline = []
        idx_true = 0
        idx_false = 0
        for p in predicate:
            if p:
                out_baseline.append(out_true[idx_true])
                idx_true = idx_true + 1
            else:
                out_baseline.append(out_false[idx_false])
                idx_false = idx_false + 1
        if dev == "gpu":
            out = [out[i].as_cpu() for i in range(bs)]
            out_baseline = [out_baseline[i].as_cpu() for i in range(bs)]
        check_batch(out, out_baseline, bs)


def test_conditional_split_merge():
    rng = np.random.default_rng()
    for dev in ["cpu", "gpu"]:
        for pred_gen in [
                lambda x: np.array(x < 3), lambda x: np.array(x % 2 == 0),
                lambda x: np.array(x % 3 == 0), lambda _: np.array(False),
                lambda _: rng.choice([np.array(True), np.array(False)])
        ]:
            yield check_conditional_split_merge, dev, pred_gen

import pdb

@experimental.pipeline_def(enable_conditionals=True)
def cond_after_cond(dev):
    # need to create them within the pipeline scope
    input = fn.external_source(name="input", device=dev)
    pred_0 = fn.external_source(name="pred_0")
    pred_1 = fn.external_source(name="pred_1")
    if pred_0:
        output = input + 1
    else:
        output = 2 + input
    if pred_1:
        output2 = output + 3
    else:
        output2 = output + 4
    return output, output2

def cond_after_cond_scalar(input, pred_0, pred_1):
    if pred_0:
        output = input + 1
    else:
        output = input + 2

    if pred_1:
        output2 = output + 3
    else:
        output2 = output + 4
    return output, output2




rng = np.random.default_rng( )
pred_gens = [
    lambda x: np.array(x < 3), lambda x: np.array(x % 2 == 0), lambda x: np.array(x % 3 == 0),
    lambda _: np.array(False),
    lambda _: rng.choice([np.array(True), np.array(False)])
]


input_gens = [
    lambda x : np.array(0), lambda x: np.array(x)
]

@params(*itertools.product(["cpu", "gpu"], input_gens, pred_gens, pred_gens))
def test_cond_after_cond(dev, input_gen, pred_gen_0, pred_gen_1):
    bs = 10
    kwargs = {
        "batch_size": bs,
        "num_threads": 4,
        "device_id": 0,
        "prefetch_queue_depth": 1  # so that it's easier to use external source
    }


    input = [input_gen(i) for i in range(bs)]
    pred_0 = [pred_gen_0(i) for i in range(bs)]
    pred_1 = [pred_gen_1(i) for i in range(bs)]

    pipe = cond_after_cond(dev, **kwargs)
    pipe.build()
    pipe.feed_input("input", input)
    pipe.feed_input("pred_0", pred_0)
    pipe.feed_input("pred_1", pred_1)
    output, output2 = pipe.run()
    print(output, output2)
    baseline_output = []
    baseline_output2 = []
    for input_i, pred_0_i, pred_1_i in zip(input, pred_0, pred_1):
        #   print(input_i, pred_0_i, pred_1_i)
      output_i, output2_i = cond_after_cond_scalar(input_i, pred_0_i, pred_1_i)
      baseline_output.append(output_i)
      baseline_output2.append(output2_i)
    check_batch(output, baseline_output, bs)
    check_batch(output2, baseline_output2, bs)



# @pipeline_def
# def conditional_split_merge_reinterpret_pipe(dtype, layout, shape):
#     batch_size = Pipeline.current().max_batch_size
#     input = fn.external_source(
#         source=[[np.full((10, 10, 3), 42, dtype=np.int32) for _ in range(batch_size)]], cycle=True)
#     pred = fn.external_source(
#         source=[[np.array(i % 2 == 0, dtype=np.bool) for i in range(batch_size)]], cycle=True)
#     true_branch, false_branch = fn._conditional.split(input, predicate=pred)
#     false_changed = fn.reinterpret(false_branch, dtype=dtype, layout=layout, shape=shape)
#     return fn._conditional.merge(true_branch, false_changed, predicate=pred)


# def run_conditional_split_merge_reinterpret(dtype, layout, shape):
#     bs = 10
#     kwargs = {
#         "batch_size": bs,
#         "num_threads": 4,
#         "device_id": 0,
#         "prefetch_queue_depth": 1  # so that it's easier to use external source
#     }
#     pipe = conditional_split_merge_reinterpret_pipe(dtype, layout, shape, **kwargs)
#     pipe.build()
#     pipe.run()


# @params((types.UINT32, None, None, "types*"),
#         (None, "HWC", None, "layouts*"),
#         (None, None, [10, -1], "sample dimensions*"))
# def test_fail_conditional_split_merge(dtype, layout, shape, err_glob):
#     base = ("Divergent data found in different branches of conditional operation. All paths in "
#             "conditional operation are merged into one batch which must have consistent type, "
#             "number of dimensions, layout and other metadata. Found distinct ")

#     with assert_raises(RuntimeError, glob=base + err_glob):
#         run_conditional_split_merge_reinterpret(dtype, layout, shape)
