# Copyright (c) 2019, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nvidia.dali import Pipeline, pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import math
import os
import random
import numpy as np
import glob

from nose import SkipTest
from nose.tools import assert_raises

from test_utils import check_batch
from test_utils import compare_pipelines
from test_utils import RandomDataIterator
from test_utils import get_dali_extra_path
from test_utils import check_output_pattern
from test_utils import to_array

def get_img_files(data_path, subdir='*', ext=None):
    if subdir is None:
        subdir = ''
    if ext:
        if isinstance(ext, (list, tuple)):
            files = []
            for e in ext:
                files += glob.glob(data_path + f"/{subdir}/*.{e}")
        else:
            files = glob.glob(data_path + f"/{subdir}/*.{ext}")
        return files
    else:
        files = glob.glob(data_path  + f"/{subdir}/*.*")
        txt_files = glob.glob(data_path  + f"/{subdir}/*.txt")
        return list(set(files) - set(txt_files))

@pipeline_def
def decoder_pipe(data_path, device, use_fast_idct=False, memory_stats=False):
    inputs, labels = fn.readers.file(file_root = data_path,
                                     shard_id = 0,
                                     num_shards = 1,
                                     name="Reader")
    decoded = fn.decoders.image(inputs, device = device, output_type = types.RGB, use_fast_idct=use_fast_idct,
                                memory_stats=memory_stats)

    return decoded, labels

test_data_root = get_dali_extra_path()
good_path = 'db/single'
missnamed_path = 'db/single/missnamed'
test_good_path = {'jpeg', 'mixed', 'png', 'tiff', 'pnm', 'bmp', 'jpeg2k'}
test_missnamed_path = {'jpeg', 'png', 'tiff', 'pnm', 'bmp'}

def run_decode(data_path, batch, device, threads, memory_stats=False):
    pipe = decoder_pipe(data_path=data_path, batch_size=batch, num_threads=threads, device_id=0, device=device, memory_stats=memory_stats, prefetch_queue_depth=1)
    pipe.build()
    iters = math.ceil(pipe.epoch_size("Reader") / batch)
    for _ in range(iters):
        pipe.run()

def test_image_decoder():
    def log(img_type, size, device, threads):
        pass
    for device in {'cpu', 'mixed'}:
        for batch_size in {1, 10}:
            for img_type in test_good_path:
                for threads in {1, random.choice([2, 3, 4])}:
                    data_path = os.path.join(test_data_root, good_path, img_type)
                    run_decode(data_path, batch_size, device, threads)
                    yield log, img_type, batch_size, device, threads
            for img_type in test_missnamed_path:
                for threads in {1, random.choice([2, 3, 4])}:
                    data_path = os.path.join(test_data_root, missnamed_path, img_type)
                    run_decode(data_path, batch_size, device, threads)
                    yield log, img_type, batch_size, device, threads

# TODO(januszl): check padding behavior
@pipeline_def
def create_decoder_slice_pipeline(data_path, device):
    jpegs, _ = fn.readers.file(file_root = data_path,
                               shard_id = 0,
                               num_shards = 1,
                               name = "Reader")

    anchor = fn.random.uniform(range=[0.05, 0.15], shape=(2,))
    shape = fn.random.uniform(range=[0.5, 0.7], shape=(2,))
    images_sliced_1 = fn.decoders.image_slice(jpegs,
                                              anchor,
                                              shape,
                                              device = device,
                                              hw_decoder_load = 0.7,
                                              output_type = types.RGB,
                                              axes = (0, 1))

    images = fn.decoders.image(jpegs,
                               device = device,
                               hw_decoder_load = 0.7,
                               output_type = types.RGB)
    images_sliced_2 = fn.slice(images,
                               anchor,
                               shape,
                               axes = (0, 1))

    return images_sliced_1, images_sliced_2

# TODO(januszl): check padding behavior
@pipeline_def
def create_decoder_crop_pipeline(data_path, device):
    jpegs, _ = fn.readers.file(file_root = data_path,
                               shard_id = 0,
                               num_shards = 1,
                               name = "Reader")

    crop_pos_x = fn.random.uniform(range=[0.1, 0.9])
    crop_pos_y = fn.random.uniform(range=[0.1, 0.9])
    w = 242
    h = 230

    images_crop_1 = fn.decoders.image_crop(jpegs,
                                           device = device,
                                           output_type = types.RGB,
                                           hw_decoder_load = 0.7,
                                           crop = (w, h),
                                           crop_pos_x = crop_pos_x,
                                           crop_pos_y = crop_pos_y)

    images = fn.decoders.image(jpegs,
                               device = device,
                               hw_decoder_load = 0.7,
                               output_type = types.RGB)

    images_crop_2 = fn.crop(images,
                            crop = (w, h),
                            crop_pos_x = crop_pos_x,
                            crop_pos_y = crop_pos_y)

    return images_crop_1, images_crop_2

@pipeline_def
def create_decoder_random_crop_pipeline(data_path, device):
    seed = 1234
    jpegs, _ = fn.readers.file(file_root = data_path,
                               shard_id = 0,
                               num_shards = 1,
                               name = "Reader")

    w = 242
    h = 230
    images_random_crop_1 = fn.decoders.image_random_crop(jpegs,
                                                         device = device,
                                                         output_type = types.RGB,
                                                         hw_decoder_load = 0.7,
                                                         seed = seed)
    images_random_crop_1 = fn.resize(images_random_crop_1, size = (w, h))

    images = fn.decoders.image(jpegs,
                               device = device,
                               hw_decoder_load = 0.7,
                               output_type = types.RGB)
    images_random_crop_2 = fn.random_resized_crop(images, size = (w, h), seed = seed)

    return images_random_crop_1, images_random_crop_2

def run_decode_fused(test_fun, path, img_type, batch, device, threads, validation_fun):
    data_path = os.path.join(test_data_root, path, img_type)
    pipe = test_fun(data_path=data_path, batch_size=batch, num_threads=threads, device_id=0, device=device, prefetch_queue_depth=1)
    pipe.build()
    iters = math.ceil(pipe.epoch_size("Reader") / batch)
    for _ in range(iters):
        out_1, out_2 = pipe.run()
        for img_1, img_2 in zip(out_1, out_2):
            img_1 = to_array(img_1)
            img_2 = to_array(img_2)
            assert validation_fun(img_1, img_2)

def test_image_decoder_fused():
    threads = 4
    batch_size = 10
    for test_fun in [create_decoder_slice_pipeline, create_decoder_crop_pipeline, create_decoder_random_crop_pipeline]:
        if test_fun == create_decoder_random_crop_pipeline:
            # random_resized_crop can properly handle border as it has pixels that are cropped out, while
            # plain resize folowing image_decoder_random_crop cannot do that and must duplicate the border pixels
            validation_fun = lambda x, y: np.mean(np.abs(x - y) < 0.5)
        else:
            validation_fun = lambda x, y: np.allclose(x, y)
        for device in {'cpu', 'mixed'}:
            for img_type in test_good_path:
                yield run_decode_fused, test_fun, good_path, img_type, batch_size, device, threads, validation_fun

def check_FastDCT_body(batch_size, img_type, device):
    data_path = os.path.join(test_data_root, good_path, img_type)
    compare_pipelines(decoder_pipe(data_path=data_path, batch_size=batch_size, num_threads=3,
                                   device_id=0, device=device, use_fast_idct=False),
                      decoder_pipe(data_path=data_path, batch_size=batch_size, num_threads=3,
                                   device_id=0, device='cpu', use_fast_idct=True),
                      # average difference should be no bigger by off-by-3
                      batch_size=batch_size, N_iterations=3, eps=3)

def test_FastDCT():
    for device in {'cpu', 'mixed'}:
        for batch_size in {1, 8}:
            for img_type in test_good_path:
              yield check_FastDCT_body, batch_size, img_type, device

def test_image_decoder_memory_stats():
    device = 'mixed'
    img_type = 'jpeg'
    def check(img_type, size, device, threads):
        data_path = os.path.join(test_data_root, good_path, img_type)
        # largest allocation should match our (in this case) memory padding settings
        # (assuming no reallocation was needed here as the hint is big enough)
        pattern = 'Device memory: \d+ allocations, largest = 16777216 bytes\n' + \
                  'Host \(pinned\) memory: \d+ allocations, largest = 8388608 bytes\n'
        with check_output_pattern(pattern):
            run_decode(data_path, size, device, threads, memory_stats=True)

    for size in {1, 10}:
        for threads in {1, random.choice([2, 3, 4])}:
            yield check, img_type, size, device, threads

batch_size_test = 16
@pipeline_def(batch_size=batch_size_test, device_id=0, num_threads=4)
def img_decoder_pipe(device, out_type, files):
    encoded, _ = fn.readers.file(files=files)
    decoded = fn.decoders.image(encoded, device=device, output_type=out_type)
    return decoded

def _testimpl_image_decoder_consistency(img_out_type, file_fmt, path, subdir='*', ext=None):
    eps = 1
    if file_fmt == 'jpeg' or file_fmt == 'mixed':
        eps = 4
    if ((file_fmt == 'jpeg2k' or file_fmt == 'mixed') and img_out_type == types.YCbCr):
        eps = 6
    files = get_img_files(os.path.join(test_data_root, path), subdir=subdir, ext=ext)
    compare_pipelines(img_decoder_pipe("cpu", out_type=img_out_type, files=files),
                      img_decoder_pipe("mixed", out_type=img_out_type, files=files),
                      batch_size=batch_size_test, N_iterations=3,
                      eps=eps)

def test_image_decoder_consistency():
    for out_img_type in [types.RGB, types.BGR, types.YCbCr, types.GRAY, types.ANY_DATA]:
        for file_fmt in test_good_path:
            path = os.path.join(good_path, file_fmt)
            yield _testimpl_image_decoder_consistency, out_img_type, file_fmt, path

        for file_fmt, path, ext in [("tiff", "db/single/multichannel/tiff_multichannel", 'tif'),
                                    ("jpeg2k", "db/single/multichannel/with_alpha", 'jp2'),
                                    ("png", "db/single/multichannel/with_alpha", 'png')]:
            subdir = None  # In those paths the images are not organized in subdirs
            yield _testimpl_image_decoder_consistency, out_img_type, file_fmt, path, subdir, ext

def _testimpl_image_decoder_tiff_with_alpha_16bit(device, out_type, path, ext):
    @pipeline_def(batch_size=1, device_id=0, num_threads=1)
    def pipe(device, out_type, files):
        encoded, _ = fn.readers.file(files=files)
        decoded = fn.decoders.image(encoded, device=device, output_type=out_type)
        peeked_shape = fn.peek_image_shape(encoded)
        return decoded, peeked_shape

    files = get_img_files(os.path.join(test_data_root, path), ext=ext, subdir=None)
    pipe = pipe(device, out_type=out_type, files=files)
    pipe.build()
    out, shape = pipe.run()
    if device == 'mixed':
        out = out.as_cpu()
    out = np.array(out[0])
    shape = np.array(shape[0])
    expected_channels = 4 if out_type == types.ANY_DATA else \
                        1 if out_type == types.GRAY else \
                        3
    assert out.shape[2] == expected_channels, \
        f"Expected {expected_channels} but got {out.shape[2]}"

def test_image_decoder_tiff_with_alpha_16bit():
    for device in ['cpu', 'mixed']:
        for out_type in [types.RGB, types.BGR, types.YCbCr, types.ANY_DATA]:
            path = "db/single/multichannel/with_alpha_16bit"
            for ext in [("png", "tiff", "jp2")]:
                yield _testimpl_image_decoder_tiff_with_alpha_16bit, device, out_type, path, ext

@pipeline_def(batch_size=batch_size_test, device_id=0, num_threads=4)
def decoder_pipe_with_name(decoder_op, file_root, device, use_fast_idct):
    encoded, _ = fn.readers.file(file_root=file_root)
    decoded = decoder_op(encoded, device=device, output_type=types.RGB, use_fast_idct=use_fast_idct,
                         seed=42)
    return decoded

def check_image_decoder_alias(new_op, old_op, file_root, device, use_fast_idct):
    new_pipe = decoder_pipe_with_name(new_op, file_root, device, use_fast_idct)
    legacy_pipe = decoder_pipe_with_name(old_op, file_root, device, use_fast_idct)
    compare_pipelines(new_pipe, legacy_pipe, batch_size=batch_size_test, N_iterations=3)


def test_image_decoder_alias():
    data_path = os.path.join(test_data_root, good_path, "jpeg")
    for new_op, old_op in [(fn.decoders.image, fn.image_decoder),
                           (fn.decoders.image_crop, fn.image_decoder_crop),
                           (fn.decoders.image_random_crop, fn.image_decoder_random_crop)]:
        for device in ["cpu", "mixed"]:
            for use_fast_idct in [True, False]:
                yield check_image_decoder_alias, new_op, old_op, data_path, device, use_fast_idct

@pipeline_def(batch_size=batch_size_test, device_id=0, num_threads=4)
def decoder_slice_pipe(decoder_op, file_root, device, use_fast_idct):
    encoded, _ = fn.readers.file(file_root=file_root)
    start = types.Constant(np.array([0., 0.]))
    end = types.Constant(np.array([0.5, 0.5]))
    decoded = decoder_op(encoded, start, end, device=device,
                         output_type=types.RGB, use_fast_idct=use_fast_idct)
    return decoded


def check_image_decoder_slice_alias(new_op, old_op, file_root, device, use_fast_idct):
    new_pipe = decoder_slice_pipe(new_op, file_root, device, use_fast_idct)
    legacy_pipe = decoder_slice_pipe(old_op, file_root, device, use_fast_idct)
    compare_pipelines(new_pipe, legacy_pipe, batch_size=batch_size_test, N_iterations=3)

def test_image_decoder_slice_alias():
    data_path = os.path.join(test_data_root, good_path, "jpeg")
    new_op, old_op = fn.decoders.image_slice, fn.image_decoder_slice
    for device in ["cpu", "mixed"]:
        for use_fast_idct in [True, False]:
            yield check_image_decoder_slice_alias, new_op, old_op, data_path, device, use_fast_idct
