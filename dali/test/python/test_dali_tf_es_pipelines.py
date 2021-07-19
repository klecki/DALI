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

from nvidia.dali import Pipeline, fn
import nvidia.dali.plugin.tf as dali_tf


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

def get_sample_one_arg_callback(dtype, iter_limit=1000):
    def callback(x):
        if x.iteration > iter_limit:
            raise StopIteration()
        # TODO Random shape
        return dtype([x.idx_in_epoch, x.idx_in_batch, x.iteration])
    return callback


args = [
    (get_sample_one_arg_callback, False),
    (get_batch_one_arg_callback, True),
    (get_sample_no_arg_callback, False),
    (get_batch_no_arg_callback, True),
    (get_sample_iterable, False),
    (get_batch_uniform_iterable, True),
    (get_batch_non_uniform_iterable, True),
]


def get_external_source_pipe(es_args, dtype):
    def get_pipeline_desc(batch_size, num_threads, device, device_id, shard_id, num_shards,
                        def_for_dataset):

        pipe = Pipeline(batch_size, num_threads, device_id)
        with pipe:
            pipe.set_outputs(fn.external_source(device=device, **es_args))
        return pipe, None, dtype
    return get_pipeline_desc

def to_tf_dataset(pipe_desc, device_str): # -> tf.data.Dataset
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
