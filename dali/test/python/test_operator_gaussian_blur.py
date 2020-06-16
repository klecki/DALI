# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.types as types
import nvidia.dali.fn as fn

import numpy as np
import cv2
import os
from nose.tools import raises

from test_utils import get_dali_extra_path

data_root = get_dali_extra_path()
images_dir = os.path.join(data_root, 'db', 'single', 'jpeg')

# TODO need to add a generic data thingy (3D, 3D with channel, 2D without channel, Sequence of all)
# it would be easier with random data + external source
def check_gaussian_blur(batch_size, sigma, window_size, op_type="cpu"):
    decoder_device = "cpu" if op_type == "cpu" else "mixed"
    pipe = Pipeline(batch_size = batch_size, num_threads=4, device_id=0)
    with pipe:
        input, _ = fn.file_reader(file_root=images_dir, shard_id=0, num_shards=1)
        decoded = fn.image_decoder(input, device=decoder_device, output_type=types.RGB)
        blurred = fn.gaussian_blur(decoded, sigma=sigma, window_size=window_size)
        pipe.set_outputs(blurred, decoded)
    pipe.build()

    result, input = pipe.run()
    # if op_type == "gpu":
    #     result = result.as_cpu()
    #     input = input.as_cpu()
    # input = to_batch(input, batch_size)
    # print(input)
    # baseline = reorder(input, shape[0], reorders, persample_reorder)
    # for i in range(batch_size):
    #     np.testing.assert_array_equal(result[i], baseline[i])


def test_sequence_rearrange():
    for dev in ["cpu"]:
        for sigma in [1.0, [1.0, 2.0], None]:
            for window_size in [3, 5, [7, 5], [5, 9], None]:
                yield check_gaussian_blur, 5, sigma, window_size, dev
