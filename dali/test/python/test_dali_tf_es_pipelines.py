# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import numpy as np
import tensorflow as tf

from nvidia.dali import Pipeline, fn
import nvidia.dali.plugin.tf as dali_tf
from test_utils import RandomlyShapedDataIterator


def foo(x):
    print("SAMPLE CALL", x.idx_in_epoch, x.idx_in_batch, x.iteration)
    if x.iteration > 3:
        raise StopIteration()
    return np.int32([x.idx_in_epoch, x.idx_in_batch, x.iteration])

def foo_batch(x):
    print("BATCH CALL", x)
    if x > 3:
        raise StopIteration()
    return [np.int8([x])] * 10

def magic():
    try:
        magic.counter += 1
    except:
        magic.counter = 0
    print("MAGIC COUNTER", magic.counter)
    if magic.counter > 30:
        magic.counter = 0
        raise StopIteration()
    return np.int32([magic.counter])


def magic_batch():
    try:
        magic_batch.counter += 1
    except:
        magic_batch.counter = 0
    print("MAGIC COUNTER BATCH", magic_batch.counter)
    if magic_batch.counter > 3:
        magic_batch.counter = 0
        raise StopIteration()
    return [np.int32([magic_batch.counter])] * 10

# def get_sample_one_arg_callback(dtype, iter_limit=1000, batch_size=None, uniform=None):
#     def callback(x):
#         if x.iteration > iter_limit:
#             print("RAISING THE StopIteration")
#             raise StopIteration()
#         size = x.idx_in_batch % 16 + 1, x.iteration % 16 + 3
#         result = np.full(size, x.idx_in_epoch, dtype=dtype)
#         result[0][0] = x.idx_in_batch
#         result[0][1] = x.iteration
#         return result
#     return callback

def get_sample_one_arg_callback(dtype, iter_limit=1000, batch_size=None, uniform=None):
    def callback(x):
        # if x.iteration > iter_limit:
        #     print("RAISING THE StopIteration")
        #     raise StopIteration()
        size = x.idx_in_batch % 4 + 1, x.iteration % 4 + 3
        print(size)
        # size = (1, 4)
        result = np.full(size, x.idx_in_epoch, dtype=dtype)
        result[0][0] = x.idx_in_batch
        result[0][1] = x.iteration
        return np.array(result)
        return np.array([x.idx_in_epoch, x.idx_in_batch, x.iteration], dtype=dtype)
    return callback

def get_batch_one_arg_callback(dtype, iter_limit=1000, batch_size=None, uniform=True):
    def callback(x):
        if x > iter_limit:
            raise StopIteration()
        size = (x % 16 + 1,)
        return [np.full(size, x, dtype=dtype)] * batch_size
    return callback

def get_no_arg_callback(dtype, iter_limit=1000, batch_size=None, uniform=None):
    def callback():
        try:
            callback.counter += 1
        except:
            callback.counter = 0
        if callback.counter > iter_limit:
            callback.counter = 0
            raise StopIteration()
        size = (callback.counter % 16 + 1,)
        result = np.full(size, callback.counter, dtype=dtype)
        if batch_size is not None:
            return result
        else:
            return [result] * batch_size
    return callback

def get_sample_iterable(dtype, iter_limit=1000, batch_size=None, uniform=None):
    bs = 1 if batch_size is None else batch_size
    max_shape = (20, 20)
    min_shape = max_shape if uniform else None
    return RandomlyShapedDataIterator(bs, min_shape, max_shape, 42, dtype)

# generator, is_batched, is_uniform
es_configurations = [
    (get_sample_one_arg_callback, False),
    # (get_batch_one_arg_callback, True),
    # (get_no_arg_callback, False),
    # (get_no_arg_callback, True),
    # (get_sample_iterable, False),
    # (get_sample_iterable, True),
    # (get_batch_non_uniform_iterable, True, False),
]

def get_external_source_pipe(es_args, dtype, es_device):
    def get_pipeline_desc(batch_size, num_threads, device, device_id, shard_id, num_shards,
                        def_for_dataset):
        pipe = Pipeline(batch_size, num_threads, device_id)
        with pipe:
            es = fn.external_source(device=es_device, **es_args)
            if device == "gpu" and es_device == "cpu":
                es = es.gpu()
            pipe.set_outputs(es)
        return pipe, None, dtype
    return get_pipeline_desc

def external_source_to_tf_dataset(pipe_desc, device_str): # -> tf.data.Dataset
    pipe, _, dtypes = pipe_desc
    with tf.device(device_str):
        dali_dataset = dali_tf.experimental.DALIDatasetWithInputs(
                input_datasets=None,
                pipeline=pipe,
                batch_size=pipe.max_batch_size,
                output_shapes=None,
                output_dtypes=dtypes,
                num_threads=pipe.num_threads,
                device_id=pipe.device_id).repeat()
    return dali_dataset
