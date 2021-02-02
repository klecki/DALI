# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
from nose.tools import raises

import nvidia.dali as dali

import nvidia.dali.shared_batch as sb

from test_utils import compare_pipelines, check_batch

import pickle


def recursive_equals(left, right, top_level=True):
    if top_level:
        idx_l, left = left
        idx_r, right = right
        assert idx_l == idx_r, "Indexes of samples should be the same"
    if isinstance(left, tuple):
        assert isinstance(right, tuple), "Nesting should be the same"
        assert len(left) == len(right), "Nesting len should be the same"
        for i in range(len(left)):
            recursive_equals(left[i], right[i], False)
    np.testing.assert_array_equal(left, right)


def check_serialize_deserialize(indexed_batch):
    mem_chunk = sb.SharedMemChunk("chunk_0", 10000)
    shared_batch_writer = sb.SharedBatchWriter(mem_chunk)
    shared_batch_writer.write_batch(indexed_batch)
    shared_batch_meta = sb.SharedBatchMeta.from_writer(shared_batch_writer)
    sample_meta = sb.deserialize_sample_meta(mem_chunk.shm_chunk, shared_batch_meta)
    deserlized_indexed_batch = sb.deserialize_batch(mem_chunk.shm_chunk, sample_meta)
    assert len(indexed_batch) == len(
        deserlized_indexed_batch), "Lengths before and after should be the same"
    for i in range(len(deserlized_indexed_batch)):
        recursive_equals(indexed_batch[i], deserlized_indexed_batch[i])


def test_serialize_deserialize():
    for s in [(10, 20)]:
        yield check_serialize_deserialize([(0, np.zeros(s))])
