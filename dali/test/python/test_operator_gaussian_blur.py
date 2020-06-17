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
from PIL import Image, ImageFilter
from scipy.ndimage import gaussian_filter
import os
from nose.tools import raises

from test_utils import get_dali_extra_path, check_batch

data_root = get_dali_extra_path()
images_dir = os.path.join(data_root, 'db', 'single', 'jpeg')

def to_batch(tl, batch_size):
    return [np.array(tl[i]) for i in range(batch_size)]

def to_cv_win_size(window_size):
    if window_size is None:
        return (0, 0)
    elif isinstance(window_size, int):
        return (window_size, window_size)
    return window_size[0], window_size[1]

def to_cv_sigma(sigma):
    if sigma is None:
        return (0, 0)
    elif isinstance(sigma, float):
        return (sigma, sigma)
    return sigma[1], sigma[0]


def gaussian_scipy(image, sigma):
    dim = len(image.shape)
    channels = image.shape[dim - 1]
    # TODO generic split
    planes = np.split(image, channels, axis=dim-1)
    flat_planes = [plane.squeeze() for plane in planes]
    blurred = [gaussian_filter(np.float32(plane), sigma) for plane in flat_planes]
    x = [np.expand_dims(plane, dim - 1) for plane in blurred]
    return np.uint8(np.concatenate(x, axis=dim-1))


def gaussian_cv(image, sigma, window_size):
    sigma_x, sigma_y = to_cv_sigma(sigma)
    window_size_cv = to_cv_win_size(window_size)
    # compute on floats and round like a sane person (in mathematically complicit way)
    blurred = cv2.GaussianBlur(np.float32(image), window_size_cv, sigmaX=sigma_x, sigmaY=sigma_y)
    return np.uint8(blurred + 0.5)



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
        # pipe.set_outputs(decoded)
    pipe.build()

    result, input = pipe.run()
    if op_type == "gpu":
        result = result.as_cpu()
        input = input.as_cpu()
    input = to_batch(input, batch_size)
    baseline_cv = [gaussian_cv(img, sigma, window_size) for img in input]
    # PIL accuracy is absolutely abysmal, it is iterative BoxFilter not Gaussian blur
    input_pil = [Image.fromarray(img) for img in input]
    baseline_pil = [np.array(img.filter(ImageFilter.GaussianBlur(sigma))) for img in input_pil]
    baseline_scipy = [gaussian_scipy(img, sigma) for img in input]
    # check_batch(result, baseline_cv, batch_size)
    # check_batch(result, baseline_pil, batch_size)
    # for i in range(batch_size):
    #     print("Max diff", np.max(cv2.absdiff(baseline_cv[i], np.array(result[i]))))
    #     print("Total diff", np.sum(cv2.absdiff(baseline_cv[i], np.array(result[i])) != 0))
    # check_batch(result, baseline_cv, batch_size)
    print("PIL")
    check_batch(result, baseline_scipy, batch_size)
    print("CV")
    check_batch(result, baseline_cv, batch_size)
    print("SCIPY")
    check_batch(result, baseline_pil, batch_size)
    # for i in range(batch_size):
    #     diff = np.array(result[i]) - baseline[i]
    #     np.testing.assert_array_equal(result[i], baseline[i])


def test_sequence_rearrange():
    for dev in ["cpu"]:
        for sigma, window_size in [(1.0, 7)]: #, (0.6, 5), (0.7, 5), (1.0, 7), (2.0, 0), (5.0, 31), ]:
        # for sigma in [1.0, [1.0, 2.0], None]:
        #     for window_size in [3, 5, [7, 5], [5, 9], None]:
                if sigma is None and window_size is None:
                    continue
                yield check_gaussian_blur, 10, sigma, window_size, dev
