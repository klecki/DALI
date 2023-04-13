# Copyright (c) 2018-2019, NVIDIA CORPORATION
# Copyright (c) 2017-      Facebook, Inc
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import os
import torch
import numpy as np
from PIL import Image
from functools import partial

from image_classification.autoaugment import AutoaugmentImageNetPolicy

DATA_BACKEND_CHOICES = ["pytorch", "synthetic"]
try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator
    import nvidia.dali.types as types

    from image_classification.dali import training_pipe, validation_pipe

    DATA_BACKEND_CHOICES.append("dali")
except ImportError as e:
    print(
        "Please install DALI from https://www.github.com/NVIDIA/DALI to run this example."
    )

import torchvision.datasets as datasets
import torchvision.transforms as transforms

def load_jpeg_from_file(path, cuda=True):
    img_transforms = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]
    )

    img = img_transforms(Image.open(path))
    with torch.no_grad():
        # mean and std are not multiplied by 255 as they are in training script
        # torch dataloader reads data into bytes whereas loading directly
        # through PIL creates a tensor with floats in [0,1] range
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        if cuda:
            mean = mean.cuda()
            std = std.cuda()
            img = img.cuda()
        img = img.float()

        input = img.unsqueeze(0).sub_(mean).div_(std)

    return input


from nvidia.dali.plugin.base_iterator import LastBatchPolicy

from nvidia.dali import types
import math
import logging
import numpy as np
import warnings
from enum import Enum, unique
from collections.abc import Iterable
from nvidia.dali.backend import TensorGPU, TensorListGPU
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
from nvidia.dali import types
from nvidia.dali.plugin.base_iterator import _DaliBaseIterator
from nvidia.dali.plugin.base_iterator import LastBatchPolicy
from nvidia.dali.plugin.pytorch import feed_ndarray, to_torch_type
import torch
import torch.utils.dlpack as torch_dlpack
import ctypes
import numpy as np

class _DaliBaseIterator2(object):
    """
    DALI base iterator class. Shouldn't be used directly.

    Parameters
    ----------
    pipelines : list of nvidia.dali.Pipeline
                List of pipelines to use
    output_map : list of (str, str)
                 List of pairs (output_name, tag) which maps consecutive
                 outputs of DALI pipelines to proper field in MXNet's
                 DataBatch.
                 tag is one of DALIGenericIterator.DATA_TAG
                 and DALIGenericIterator.LABEL_TAG mapping given output
                 for data or label correspondingly.
                 output_names should be distinct.
    size : int, default = -1
                Number of samples in the shard for the wrapped pipeline (if there is more than one
                it is a sum). Providing -1 means that the iterator will work until StopIteration
                is raised from the inside of iter_setup(). The options `last_batch_policy`,
                `last_batch_padded` and `auto_reset` don't work in such case. It works with only
                one pipeline inside the iterator.
                Mutually exclusive with `reader_name` argument
    reader_name : str, default = None
                Name of the reader which will be queried to the shard size, number of shards, and
                all other properties necessary to count properly the number of relevant and padded
                samples that iterator needs to deal with. It allows `last_batch_policy` to be
                PARTIAL or DROP, if FILL is used it is changed to PARTIAL. Sets `last_batch_padded`
                accordingly to the reader's configuration (`pad_last_batch` reader argument)
    auto_reset : string or bool, optional, default = False
                Whether the iterator resets itself for the next epoch or it requires reset() to be
                called explicitly.

                It can be one of the following values:

                * ``"no"``, ``False`` or ``None`` - at the end of epoch StopIteration is raised and
                  reset() needs to be called. Calling ``iter()`` on the iterator would reset
                  it as well.
                * ``"yes"`` or ``"True"``- at the end of epoch StopIteration is raised but reset()
                  is called internally automatically

    fill_last_batch : bool, optional, default = None
                **Deprecated** Please use ``last_batch_policy`` instead

                Whether to fill the last batch with data up to 'self.batch_size'.
                The iterator would return the first integer multiple
                of self._num_gpus * self.batch_size entries which exceeds 'size'.
                Setting this flag to False will cause the iterator to return
                exactly 'size' entries.
    last_batch_policy: optional, default = LastBatchPolicy.FILL
                What to do with the last batch when there are not enough samples in the epoch
                to fully fill it. See :meth:`nvidia.dali.plugin.base_iterator.LastBatchPolicy`
    last_batch_padded : bool, optional, default = False
                Whether the last batch provided by DALI is padded with the last sample
                or it just wraps up. In the conjunction with ``last_batch_policy`` it tells
                if the iterator returning last batch with data only partially filled with
                data from the current epoch is dropping padding samples or samples from
                the next epoch. If set to False next
                epoch will end sooner as data from it was consumed but dropped. If set to
                True next epoch would be the same length as the first one. For this to happen,
                the option `pad_last_batch` in the reader needs to be set to True as well.
                It is overwritten when `reader_name` argument is provided
    prepare_first_batch : bool, optional, default = True
                Whether DALI should buffer the first batch right after the creation of the iterator,
                so one batch is already prepared when the iterator is prompted for the data

    Example
    -------
    With the data set ``[1,2,3,4,5,6,7]`` and the batch size 2:

    last_batch_policy = LastBatchPolicy.PARTIAL, last_batch_padded = True  -> last batch = ``[7]``,
    next iteration will return ``[1, 2]``

    last_batch_policy = LastBatchPolicy.PARTIAL, last_batch_padded = False -> last batch = ``[7]``,
    next iteration will return ``[2, 3]``

    last_batch_policy = LastBatchPolicy.FILL, last_batch_padded = True   -> last batch = ``[7, 7]``,
     next iteration will return ``[1, 2]``

    last_batch_policy = LastBatchPolicy.FILL, last_batch_padded = False  -> last batch = ``[7, 1]``,
    next iteration will return ``[2, 3]``

    last_batch_policy = LastBatchPolicy.DROP, last_batch_padded = True   -> last batch = ``[5, 6]``,
    next iteration will return ``[1, 2]``

    last_batch_policy = LastBatchPolicy.DROP, last_batch_padded = False  -> last batch = ``[5, 6]``,
    next iteration will return ``[2, 3]``
    """

    def __init__(self,
                 pipelines,
                 size=-1,
                 reader_name=None,
                 auto_reset=False,
                 fill_last_batch=None,
                 last_batch_padded=False,
                 last_batch_policy=LastBatchPolicy.FILL,
                 prepare_first_batch=True):
        assert pipelines is not None, "Number of provided pipelines has to be at least 1"
        if not isinstance(pipelines, list):
            pipelines = [pipelines]
        self._num_gpus = len(pipelines)
        # frameworks expect from its data iterators to have batch_size field,
        # so it is not possible to use _batch_size instead
        self.batch_size = pipelines[0].max_batch_size
        assert np.all(np.equal([pipe.max_batch_size for pipe in pipelines], self.batch_size)), \
               "All pipelines should have the same batch size set"

        self._size = int(size)
        if not auto_reset or auto_reset is None or auto_reset == "no":
            self._auto_reset = "no"
        elif auto_reset or auto_reset == "yes":
            self._auto_reset = "yes"
        else:
            raise ValueError(f"Unsupported value for `auto_reset` {auto_reset}")
        self._prepare_first_batch = prepare_first_batch

        if fill_last_batch is not None:
            warnings.warn("Please do not use `fill_last_batch` and use `last_batch_policy` \
                           instead.", Warning, stacklevel=2)
            if fill_last_batch:
                self._last_batch_policy = LastBatchPolicy.FILL
            else:
                self._last_batch_policy = LastBatchPolicy.PARTIAL
        else:
            if type(last_batch_policy) is not LastBatchPolicy:
                raise ValueError("Wrong type for `last_batch_policy`. "
                                 f"Expected {LastBatchPolicy}, got {type(last_batch_policy)}")
            self._last_batch_policy = last_batch_policy

        self._last_batch_padded = last_batch_padded
        assert self._size != 0, "Size cannot be 0"
        assert self._size > 0 or (self._size < 0 and (len(pipelines) == 1 or reader_name)), \
               "Negative size is supported only for a single pipeline"
        assert not reader_name or (reader_name and self._size < 0), \
               "When reader_name is provided, size should not be set"
        assert not reader_name or (reader_name and not last_batch_padded), \
               "When reader_name is provided, last_batch_padded should not be set"
        if self._size < 0 and not reader_name:
            self._last_batch_policy = LastBatchPolicy.FILL
            self._last_batch_padded = False
        # if self.size > 0 and not reader_name:
        #     _iterator_deprecation_warning()
        self._pipes = pipelines
        self._counter = 0

        # Build all pipelines
        for p in self._pipes:
            with p._check_api_type_scope(types.PipelineAPIType.ITERATOR):
                p.build()

        self._reader_name = reader_name
        self._extract_from_reader_and_validate()
        self._ever_scheduled = False
        self._ever_consumed = False

    def _calculate_shard_sizes(self, shard_nums):
        shards_beg = np.floor(shard_nums * self._size_no_pad / self._shards_num)
        shards_end = np.floor((shard_nums + 1) * self._size_no_pad / self._shards_num)
        shards_beg = shards_beg.astype(np.int64)
        shards_end = shards_end.astype(np.int64)
        return shards_end - shards_beg

    def _extract_from_reader_and_validate(self):
        if self._reader_name:
            readers_meta = [p.reader_meta(self._reader_name) for p in self._pipes]

            def err_msg_gen(err_msg):
                return 'Reader Operator should have the same {} in all the pipelines.'.format(
                    err_msg
                )

            def check_equality_and_get(input_meta, name, err_msg):
                assert np.all(np.equal([meta[name] for meta in input_meta], input_meta[0][name])), \
                       err_msg_gen(err_msg)
                return input_meta[0][name]

            def check_all_or_none_and_get(input_meta, name, err_msg):
                assert np.all([meta[name] for meta in readers_meta]) or \
                       not np.any([meta[name] for meta in readers_meta]), \
                       err_msg_gen(err_msg)
                return input_meta[0][name]

            self._size_no_pad = check_equality_and_get(readers_meta,
                                                       "epoch_size", "size value")
            self._shards_num = check_equality_and_get(readers_meta,
                                                      "number_of_shards",
                                                      "`num_shards` argument set")
            self._last_batch_padded = check_all_or_none_and_get(readers_meta, "pad_last_batch",
                                                                "`pad_last_batch` argument set")
            self._is_stick_to_shard = check_all_or_none_and_get(readers_meta, "stick_to_shard",
                                                                "`stick_to_shard` argument set")

            self._shards_id = np.array([meta["shard_id"] for meta in readers_meta], dtype=np.int64)

            if self._last_batch_policy == LastBatchPolicy.DROP:
                # when DROP policy is used round down the shard size
                self._size = self._size_no_pad // self._shards_num
            elif self._last_batch_padded:
                # if padding is enabled all shards are equal
                self._size = readers_meta[0]["epoch_size_padded"] // self._shards_num
            else:
                # get the size as a multiply of the batch size that is bigger or equal
                # than the biggest shard
                self._size = math.ceil(math.ceil(self._size_no_pad / self._shards_num) /
                                       self.batch_size) * self.batch_size

            # count where we starts inside each GPU shard in given epoch,
            # if shards are uneven this will differ epoch2epoch
            self._counter_per_gpu = np.zeros(self._shards_num, dtype=np.int64)
            self._shard_sizes_per_gpu = self._calculate_shard_sizes(np.arange(0, self._shards_num))

            # to avoid recalculation of shard sizes when iterator moves across the shards
            # memorize the initial shard sizes and then use chaning self._shards_id to index it
            self._shard_sizes_per_gpu_initial = self._shard_sizes_per_gpu.copy()

    def _remove_padded(self):
        """
        Checks if remove any padded sample and how much.

        Calculates the number of padded samples in the batch for each pipeline
        wrapped up by the iterator. Returns if there is any padded data that
        needs to be dropped and if so how many samples in each GPU
        """
        if_drop = False
        left = -1
        if self._last_batch_policy == LastBatchPolicy.PARTIAL:
            # calculate each shard size for each id, and check how many samples are left
            # by subtracting from iterator counter the shard size, then go though all GPUs
            # and check how much data needs to be dropped
            left = self.batch_size - \
                   (self._counter - self._shard_sizes_per_gpu_initial[self._shards_id])
            if_drop = np.less(left, self.batch_size)
        return if_drop, left

    def _get_outputs(self):
        """
        Checks iterator stop condition, gets DALI outputs and perform reset in case of StopIteration
        """
        # if pipeline was not scheduled ever do it here
        if not self._ever_scheduled:
            self._schedule_runs(False)
        if self._size > 0 and self._counter >= self._size:
            self._end_iteration()

        outputs = []
        try:
            for p in self._pipes:
                with p._check_api_type_scope(types.PipelineAPIType.ITERATOR):
                    outputs.append(p.run())
        except StopIteration as e:
            # in case ExternalSource returns StopIteration
            if self._size < 0 and self._auto_reset == "yes":
                self.reset()
            raise e
        self._check_batch_size(outputs)
        return outputs

    def _check_batch_size(self, outs):
        if not isinstance(outs, Iterable):
            outs = [outs]
        if self._reader_name or self._size != -1:
            for out in outs:
                for o in out:
                    batch_len = len(o)
                    assert self.batch_size == batch_len, \
                        "Variable batch size is not supported by the iterator " + \
                        "when reader_name is provided or iterator size is set explicitly"

    def _end_iteration(self):
        if self._auto_reset == "yes":
            self.reset()
        raise StopIteration

    def _schedule_runs(self, release_outputs=True):
        """
        Schedule DALI runs
        """
        self._ever_scheduled = True
        # for p in self._pipes:
        #     with p._check_api_type_scope(types.PipelineAPIType.ITERATOR):
        #         if release_outputs:
        #             p.release_outputs()
        #         p.schedule_run()

    def _advance_and_check_drop_last(self):
        """
        Checks whether the current batch is not fully filled and whether it should be dropped.
        """
        # check if for given initial count in any GPU with the current value of the samples read
        # if we read one more batch would we overflow
        if self._reader_name:
            self._counter += self.batch_size
            if self._last_batch_policy == LastBatchPolicy.DROP:
                if np.any(self._counter_per_gpu + self._counter > self._shard_sizes_per_gpu):
                    self._end_iteration()
        else:
            self._counter += self._num_gpus * self.batch_size
            if self._last_batch_policy == LastBatchPolicy.DROP:
                if self._counter > self._size:
                    self._end_iteration()

    def reset(self):
        """
        Resets the iterator after the full epoch.
        DALI iterators do not support resetting before the end of the epoch
        and will ignore such request.
        """
        if self._counter >= self._size or self._size < 0:
            if self._last_batch_policy == LastBatchPolicy.FILL and not self._last_batch_padded:
                if self._reader_name:
                    # accurate way
                    # get the number of samples read in this epoch by each GPU
                    # self._counter had initial value of min(self._counter_per_gpu) so subtract
                    # this to get the actual value
                    self._counter -= min(self._counter_per_gpu)
                    self._counter_per_gpu = self._counter_per_gpu + self._counter
                    # check how much each GPU read ahead from next shard, as shards have different
                    # size each epoch GPU may read ahead or not
                    self._counter_per_gpu = self._counter_per_gpu - self._shard_sizes_per_gpu
                    # to make sure that in the next epoch we read the whole shard we need
                    # to set start value to the smallest one
                    self._counter = min(self._counter_per_gpu)
                else:
                    # legacy way
                    self._counter = self._counter % self._size
            else:
                self._counter = 0
            # advance to the next shard
            if self._reader_name:
                if not self._is_stick_to_shard:
                    # move shards id for wrapped pipeliens
                    self._shards_id = (self._shards_id + 1) % self._shards_num
                # revaluate _size
                if self._last_batch_policy == LastBatchPolicy.FILL and not self._last_batch_padded:
                    # move all shards ids GPU ahead
                    if not self._is_stick_to_shard:
                        self._shard_sizes_per_gpu = np.roll(self._shard_sizes_per_gpu, 1)
                    # check how many samples we need to reach from each shard in next epoch
                    # per each GPU taking into account already read
                    read_in_next_epoch = self._shard_sizes_per_gpu - self._counter_per_gpu
                    # get the maximmum number of samples and round it up to full batch sizes
                    self._size = math.ceil(max(read_in_next_epoch) / self.batch_size) * \
                        self.batch_size
                    # in case some epoch is skipped because we have read ahead in this epoch so
                    # much that in the next one we done already
                    if self._size == 0:
                        # it means that self._shard_sizes_per_gpu == self._counter_per_gpu,
                        # so we can jump to the next epoch and zero self._counter_per_gpu
                        self._counter_per_gpu = np.zeros(self._shards_num, dtype=np.int64)
                        # self._counter = min(self._counter_per_gpu), but just set 0
                        # to make it simpler
                        self._counter = 0
                        # roll once again
                        self._shard_sizes_per_gpu = np.roll(self._shard_sizes_per_gpu, 1)
                        # as self._counter_per_gpu is 0 we can just use
                        # read_in_next_epoch = self._shard_sizes_per_gpu
                        self._size = math.ceil(max(self._shard_sizes_per_gpu) / self.batch_size) * \
                            self.batch_size

            for p in self._pipes:
                p.reset()
                if p.empty():
                    with p._check_api_type_scope(types.PipelineAPIType.ITERATOR):
                        p.schedule_run()
        else:
            logging.warning("DALI iterator does not support resetting while epoch is not finished. \
                             Ignoring...")

    def next(self):
        """
        Returns the next batch of data.
        """
        self._ever_consumed = True
        return self.__next__()

    def __next__(self):
        raise NotImplementedError

    def __iter__(self):
        # avoid redundant reset when someone would call `iter()` on a new iterator
        # do not reset if no data was consumed from the iterator - to avoid unintended
        # buffering in the pipeline and the FW iterator
        if self._counter != 0 and self._ever_consumed:
            self.reset()
        return self

    @property
    def size(self):
        return self._size

    def __len__(self):
        if self._reader_name:
            if self._last_batch_policy != LastBatchPolicy.DROP:
                return math.ceil(self.size / self.batch_size)
            else:
                return self.size // self.batch_size
        else:
            if self._last_batch_policy != LastBatchPolicy.DROP:
                return math.ceil(self.size / (self._num_gpus * self.batch_size))
            else:
                return self.size // (self._num_gpus * self.batch_size)

class DALIWrapper(object):

    def gen_wrapper(dalipipeline, num_classes, one_hot, memory_format):
        for data in dalipipeline:
            if memory_format == torch.channels_last:
                # If we requested the data in channels_last form, utilize the fact that DALI
                # can return it as NHWC. The network expects NCHW shape with NHWC internal memory,
                # so we can keep the memory and just create a view with appropriate shape and
                # strides reflacting that memory layouyt
                shape = data[0]["data"].shape
                stride = data[0]["data"].stride()

                # permute shape and stride from NHWC to NCHW
                def nhwc_to_nchw(t):
                    return t[0], t[3], t[1], t[2]

                input = torch.as_strided(data[0]["data"], size=nhwc_to_nchw(shape),
                                         stride=nhwc_to_nchw(stride))
            else:
                input = data[0]["data"].contiguous(memory_format=memory_format)
            target = torch.reshape(data[0]["label"], [-1]).cuda().long()
            if one_hot:
                target = expand(num_classes, torch.float, target)
            yield input, target
        dalipipeline.reset()

    def __init__(self, dalipipeline, num_classes, one_hot, memory_format):
        self.dalipipeline = dalipipeline
        self.num_classes = num_classes
        self.one_hot = one_hot
        self.memory_format = memory_format

    def __iter__(self):
        return DALIWrapper.gen_wrapper(
            self.dalipipeline, self.num_classes, self.one_hot, self.memory_format
        )

class DALIGenericIterator2(_DaliBaseIterator2):
    """
    General DALI iterator for PyTorch. It can return any number of
    outputs from the DALI pipeline in the form of PyTorch's Tensors.

    Parameters
    ----------
    pipelines : list of nvidia.dali.Pipeline
                List of pipelines to use
    output_map : list of str
                List of strings which maps consecutive outputs
                of DALI pipelines to user specified name.
                Outputs will be returned from iterator as dictionary
                of those names.
                Each name should be distinct
    size : int, default = -1
                Number of samples in the shard for the wrapped pipeline (if there is more than
                one it is a sum)
                Providing -1 means that the iterator will work until StopIteration is raised
                from the inside of iter_setup(). The options `last_batch_policy` and
                `last_batch_padded` don't work in such case. It works with only one pipeline inside
                the iterator.
                Mutually exclusive with `reader_name` argument
    reader_name : str, default = None
                Name of the reader which will be queried to the shard size, number of shards and
                all other properties necessary to count properly the number of relevant and padded
                samples that iterator needs to deal with. It automatically sets `last_batch_policy`
                to PARTIAL when the FILL is used, and `last_batch_padded` accordingly to match
                the reader's configuration
    auto_reset : string or bool, optional, default = False
                Whether the iterator resets itself for the next epoch or it requires reset() to be
                called explicitly.

                It can be one of the following values:

                * ``"no"``, ``False`` or ``None`` - at the end of epoch StopIteration is raised
                  and reset() needs to be called
                * ``"yes"`` or ``"True"``- at the end of epoch StopIteration is raised but reset()
                  is called internally automatically

    dynamic_shape : any, optional,
                Parameter used only for backward compatibility.
    fill_last_batch : bool, optional, default = None
                **Deprecated** Please use ``last_batch_policy`` instead

                Whether to fill the last batch with data up to 'self.batch_size'.
                The iterator would return the first integer multiple
                of self._num_gpus * self.batch_size entries which exceeds 'size'.
                Setting this flag to False will cause the iterator to return
                exactly 'size' entries.
    last_batch_policy: optional, default = LastBatchPolicy.FILL
                What to do with the last batch when there are not enough samples in the epoch
                to fully fill it. See :meth:`nvidia.dali.plugin.base_iterator.LastBatchPolicy`
    last_batch_padded : bool, optional, default = False
                Whether the last batch provided by DALI is padded with the last sample
                or it just wraps up. In the conjunction with ``last_batch_policy`` it tells
                if the iterator returning last batch with data only partially filled with
                data from the current epoch is dropping padding samples or samples from
                the next epoch. If set to ``False`` next
                epoch will end sooner as data from it was consumed but dropped. If set to
                True next epoch would be the same length as the first one. For this to happen,
                the option `pad_last_batch` in the reader needs to be set to True as well.
                It is overwritten when `reader_name` argument is provided
    prepare_first_batch : bool, optional, default = True
                Whether DALI should buffer the first batch right after the creation of the iterator,
                so one batch is already prepared when the iterator is prompted for the data

    Example
    -------
    With the data set ``[1,2,3,4,5,6,7]`` and the batch size 2:

    last_batch_policy = LastBatchPolicy.PARTIAL, last_batch_padded = True  -> last batch = ``[7]``,
    next iteration will return ``[1, 2]``

    last_batch_policy = LastBatchPolicy.PARTIAL, last_batch_padded = False -> last batch = ``[7]``,
    next iteration will return ``[2, 3]``

    last_batch_policy = LastBatchPolicy.FILL, last_batch_padded = True   -> last batch = ``[7, 7]``,
    next iteration will return ``[1, 2]``

    last_batch_policy = LastBatchPolicy.FILL, last_batch_padded = False  -> last batch = ``[7, 1]``,
    next iteration will return ``[2, 3]``

    last_batch_policy = LastBatchPolicy.DROP, last_batch_padded = True   -> last batch = ``[5, 6]``,
    next iteration will return ``[1, 2]``

    last_batch_policy = LastBatchPolicy.DROP, last_batch_padded = False  -> last batch = ``[5, 6]``,
    next iteration will return ``[2, 3]``
    """

    def __init__(self,
                 pipelines,
                 output_map,
                 size=-1,
                 reader_name=None,
                 auto_reset=False,
                 fill_last_batch=None,
                 dynamic_shape=False,
                 last_batch_padded=False,
                 last_batch_policy=LastBatchPolicy.FILL,
                 prepare_first_batch=True):

        # check the assert first as _DaliBaseIterator would run the prefetch
        assert len(set(output_map)) == len(output_map), "output_map names should be distinct"
        self._output_categories = set(output_map)
        self.output_map = output_map

        _DaliBaseIterator2.__init__(self,
                                   pipelines,
                                   size,
                                   reader_name,
                                   auto_reset,
                                   fill_last_batch,
                                   last_batch_padded,
                                   last_batch_policy,
                                   prepare_first_batch=prepare_first_batch)

        self._first_batch = None
        if self._prepare_first_batch:
            try:
                self._first_batch = DALIGenericIterator2.__next__(self)
                # call to `next` sets _ever_consumed to True but if we are just calling it from
                # here we should set if to False again
                self._ever_consumed = False
            except StopIteration:
                assert False, "It seems that there is no data in the pipeline. This may happen " \
                       "if `last_batch_policy` is set to PARTIAL and the requested batch size is " \
                       "greater than the shard size."

    def __next__(self):
        self._ever_consumed = True
        if self._first_batch is not None:
            batch = self._first_batch
            self._first_batch = None
            return batch

        # Gather outputs
        outputs = self._get_outputs()

        data_batches = [None for i in range(self._num_gpus)]
        for i in range(self._num_gpus):
            dev_id = self._pipes[i].device_id
            # initialize dict for all output categories
            category_outputs = dict()
            # segregate outputs into categories
            for j, out in enumerate(outputs[i]):
                category_outputs[self.output_map[j]] = out

            # Change DALI TensorLists into Tensors
            category_tensors = dict()
            category_shapes = dict()
            for category, out in category_outputs.items():
                category_tensors[category] = out.as_tensor()
                category_shapes[category] = category_tensors[category].shape()

            category_torch_type = dict()
            category_device = dict()
            torch_gpu_device = None
            torch_cpu_device = torch.device('cpu')
            # check category and device
            for category in self._output_categories:
                category_torch_type[category] = to_torch_type[category_tensors[category].dtype]
                if type(category_tensors[category]) is TensorGPU:
                    if not torch_gpu_device:
                        torch_gpu_device = torch.device('cuda', dev_id)
                    category_device[category] = torch_gpu_device
                else:
                    category_device[category] = torch_cpu_device

            pyt_tensors = dict()
            for category in self._output_categories:
                pyt_tensors[category] = torch.empty(category_shapes[category],
                                                    dtype=category_torch_type[category],
                                                    device=category_device[category])

            data_batches[i] = pyt_tensors

            # Copy data from DALI Tensors to torch tensors
            for category, tensor in category_tensors.items():
                if isinstance(tensor, (TensorGPU, TensorListGPU)):
                    # Using same cuda_stream used by torch.zeros to set the memory
                    stream = torch.cuda.current_stream(device=pyt_tensors[category].device)
                    feed_ndarray(tensor, pyt_tensors[category], cuda_stream=stream)
                else:
                    feed_ndarray(tensor, pyt_tensors[category])

        self._schedule_runs()

        self._advance_and_check_drop_last()

        if self._reader_name:
            if_drop, left = self._remove_padded()
            if np.any(if_drop):
                output = []
                for batch, to_copy in zip(data_batches, left):
                    batch = batch.copy()
                    for category in self._output_categories:
                        batch[category] = batch[category][0:to_copy]
                    output.append(batch)
                return output

        else:
            if self._last_batch_policy == LastBatchPolicy.PARTIAL and (
                                          self._counter > self._size) and self._size > 0:
                # First calculate how much data is required to return exactly self._size entries.
                diff = self._num_gpus * self.batch_size - (self._counter - self._size)
                # Figure out how many GPUs to grab from.
                numGPUs_tograb = int(np.ceil(diff / self.batch_size))
                # Figure out how many results to grab from the last GPU
                # (as a fractional GPU batch may be required to bring us
                # right up to self._size).
                mod_diff = diff % self.batch_size
                data_fromlastGPU = mod_diff if mod_diff else self.batch_size

                # Grab the relevant data.
                # 1) Grab everything from the relevant GPUs.
                # 2) Grab the right data from the last GPU.
                # 3) Append data together correctly and return.
                output = data_batches[0:numGPUs_tograb]
                output[-1] = output[-1].copy()
                for category in self._output_categories:
                    output[-1][category] = output[-1][category][0:data_fromlastGPU]
                return output

        return data_batches

class DALIClassificationIterator2(DALIGenericIterator2):
    """
    DALI iterator for classification tasks for PyTorch. It returns 2 outputs
    (data and label) in the form of PyTorch's Tensor.

    Calling

    .. code-block:: python

       DALIClassificationIterator(pipelines, reader_name)

    is equivalent to calling

    .. code-block:: python

       DALIGenericIterator(pipelines, ["data", "label"], reader_name)

    Parameters
    ----------
    pipelines : list of nvidia.dali.Pipeline
                List of pipelines to use
    size : int, default = -1
                Number of samples in the shard for the wrapped pipeline (if there is more than
                one it is a sum)
                Providing -1 means that the iterator will work until StopIteration is raised
                from the inside of iter_setup(). The options `last_batch_policy` and
                `last_batch_padded` don't work in such case. It works with only one pipeline inside
                the iterator.
                Mutually exclusive with `reader_name` argument
    reader_name : str, default = None
                Name of the reader which will be queried to the shard size, number of shards and
                all other properties necessary to count properly the number of relevant and padded
                samples that iterator needs to deal with. It automatically sets `last_batch_policy`
                to PARTIAL when the FILL is used, and `last_batch_padded` accordingly to match
                the reader's configuration
    auto_reset : string or bool, optional, default = False
                Whether the iterator resets itself for the next epoch or it requires reset() to be
                called explicitly.

                It can be one of the following values:

                * ``"no"``, ``False`` or ``None`` - at the end of epoch StopIteration is raised
                  and reset() needs to be called
                * ``"yes"`` or ``"True"``- at the end of epoch StopIteration is raised but reset()
                  is called internally automatically

    dynamic_shape : any, optional,
                Parameter used only for backward compatibility.
    fill_last_batch : bool, optional, default = None
                **Deprecated** Please use ``last_batch_policy`` instead

                Whether to fill the last batch with data up to 'self.batch_size'.
                The iterator would return the first integer multiple
                of self._num_gpus * self.batch_size entries which exceeds 'size'.
                Setting this flag to False will cause the iterator to return
                exactly 'size' entries.
    last_batch_policy: optional, default = LastBatchPolicy.FILL
                What to do with the last batch when there are not enough samples in the epoch
                to fully fill it. See :meth:`nvidia.dali.plugin.base_iterator.LastBatchPolicy`
    last_batch_padded : bool, optional, default = False
                Whether the last batch provided by DALI is padded with the last sample
                or it just wraps up. In the conjunction with ``last_batch_policy`` it tells
                if the iterator returning last batch with data only partially filled with
                data from the current epoch is dropping padding samples or samples from
                the next epoch. If set to ``False`` next
                epoch will end sooner as data from it was consumed but dropped. If set to
                True next epoch would be the same length as the first one. For this to happen,
                the option `pad_last_batch` in the reader needs to be set to True as well.
                It is overwritten when `reader_name` argument is provided
    prepare_first_batch : bool, optional, default = True
                Whether DALI should buffer the first batch right after the creation of the iterator,
                so one batch is already prepared when the iterator is prompted for the data

    Example
    -------
    With the data set ``[1,2,3,4,5,6,7]`` and the batch size 2:

    last_batch_policy = LastBatchPolicy.PARTIAL, last_batch_padded = True  -> last batch = ``[7]``,
    next iteration will return ``[1, 2]``

    last_batch_policy = LastBatchPolicy.PARTIAL, last_batch_padded = False -> last batch = ``[7]``,
    next iteration will return ``[2, 3]``

    last_batch_policy = LastBatchPolicy.FILL, last_batch_padded = True   -> last batch = ``[7, 7]``,
    next iteration will return ``[1, 2]``

    last_batch_policy = LastBatchPolicy.FILL, last_batch_padded = False  -> last batch = ``[7, 1]``,
    next iteration will return ``[2, 3]``

    last_batch_policy = LastBatchPolicy.DROP, last_batch_padded = True   -> last batch = ``[5, 6]``,
    next iteration will return ``[1, 2]``

    last_batch_policy = LastBatchPolicy.DROP, last_batch_padded = False  -> last batch = ``[5, 6]``,
    next iteration will return ``[2, 3]``
    """

    def __init__(self,
                 pipelines,
                 size=-1,
                 reader_name=None,
                 auto_reset=False,
                 fill_last_batch=None,
                 dynamic_shape=False,
                 last_batch_padded=False,
                 last_batch_policy=LastBatchPolicy.FILL,
                 prepare_first_batch=True):
        super(DALIClassificationIterator2, self).__init__(pipelines, ["data", "label"],
                                                         size,
                                                         reader_name=reader_name,
                                                         auto_reset=auto_reset,
                                                         fill_last_batch=fill_last_batch,
                                                         dynamic_shape=dynamic_shape,
                                                         last_batch_padded=last_batch_padded,
                                                         last_batch_policy=last_batch_policy,
                                                         prepare_first_batch=prepare_first_batch)



def get_dali_train_loader(dali_device="gpu"):
    def gdtl(
        data_path,
        image_size,
        batch_size,
        num_classes,
        one_hot,
        interpolation="bilinear",
        augmentation="disabled",
        start_epoch=0,
        workers=5,
        _worker_init_fn=None,
        memory_format=torch.contiguous_format,
        **kwargs,
    ):
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1

        interpolation = {
            "bicubic": types.INTERP_CUBIC,
            "bilinear": types.INTERP_LINEAR,
            "triangular": types.INTERP_TRIANGULAR,
        }[interpolation]

        output_layout = "HWC" if memory_format == torch.channels_last else "CHW"

        traindir = os.path.join(data_path, "train")

        pipeline_kwargs = {
            "batch_size" : batch_size,
            "num_threads" : workers,
            "device_id" : rank % torch.cuda.device_count(),
            "seed": 12 + rank % torch.cuda.device_count(),
        }

        pipe = training_pipe(data_dir=traindir, interpolation=interpolation, image_size=image_size,
                             output_layout=output_layout, automatic_augmentation=augmentation,
                             dali_device=dali_device, rank=rank, world_size=world_size,
                             **pipeline_kwargs)

        pipe.build()
        train_loader = DALIClassificationIterator2(
            pipe, size=100000, fill_last_batch=False
        )

        return (
            DALIWrapper(train_loader, num_classes, one_hot, memory_format),
            int(100000 / (world_size * batch_size)),
        )

    return gdtl


def get_dali_val_loader():
    def gdvl(
        data_path,
        image_size,
        batch_size,
        num_classes,
        one_hot,
        interpolation="bilinear",
        crop_padding=32,
        workers=5,
        _worker_init_fn=None,
        memory_format=torch.contiguous_format,
        **kwargs,
    ):
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1

        interpolation = {
            "bicubic": types.INTERP_CUBIC,
            "bilinear": types.INTERP_LINEAR,
            "triangular": types.INTERP_TRIANGULAR,
        }[interpolation]

        output_layout = "HWC" if memory_format == torch.channels_last else "CHW"

        valdir = os.path.join(data_path, "val")

        pipeline_kwargs = {
            "batch_size" : batch_size,
            "num_threads" : workers,
            "device_id" : rank % torch.cuda.device_count(),
            "seed": 12 + rank % torch.cuda.device_count(),
        }

        pipe = validation_pipe(data_dir=valdir, interpolation=interpolation,
                               image_size=image_size + crop_padding, image_crop=image_size,
                               output_layout=output_layout, **pipeline_kwargs)

        pipe.build()
        val_loader = DALIClassificationIterator(
            pipe, reader_name="Reader", fill_last_batch=False
        )

        return (
            DALIWrapper(val_loader, num_classes, one_hot, memory_format),
            int(pipe.epoch_size("Reader") / (world_size * batch_size)),
        )

    return gdvl


def fast_collate(memory_format, batch):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8).contiguous(
        memory_format=memory_format
    )
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if nump_array.ndim < 3:
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)

        tensor[i] += torch.from_numpy(nump_array.copy())

    return tensor, targets


def expand(num_classes, dtype, tensor):
    e = torch.zeros(
        tensor.size(0), num_classes, dtype=dtype, device=torch.device("cuda")
    )
    e = e.scatter(1, tensor.unsqueeze(1), 1.0)
    return e


class PrefetchedWrapper(object):
    def prefetched_loader(loader, num_classes, one_hot):
        mean = (
            torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255])
            .cuda()
            .view(1, 3, 1, 1)
        )
        std = (
            torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255])
            .cuda()
            .view(1, 3, 1, 1)
        )

        stream = torch.cuda.Stream()
        first = True

        for next_input, next_target in loader:
            with torch.cuda.stream(stream):
                next_input = next_input.cuda(non_blocking=True)
                next_target = next_target.cuda(non_blocking=True)
                next_input = next_input.float()
                if one_hot:
                    next_target = expand(num_classes, torch.float, next_target)

                next_input = next_input.sub_(mean).div_(std)

            if not first:
                yield input, target
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            input = next_input
            target = next_target

        yield input, target

    def __init__(self, dataloader, start_epoch, num_classes, one_hot):
        self.dataloader = dataloader
        self.epoch = start_epoch
        self.one_hot = one_hot
        self.num_classes = num_classes

    def __iter__(self):
        if self.dataloader.sampler is not None and isinstance(
            self.dataloader.sampler, torch.utils.data.distributed.DistributedSampler
        ):

            self.dataloader.sampler.set_epoch(self.epoch)
        self.epoch += 1
        return PrefetchedWrapper.prefetched_loader(
            self.dataloader, self.num_classes, self.one_hot
        )

    def __len__(self):
        return len(self.dataloader)


def get_pytorch_train_loader(
    data_path,
    image_size,
    batch_size,
    num_classes,
    one_hot,
    interpolation="bilinear",
    augmentation=None,
    start_epoch=0,
    workers=5,
    _worker_init_fn=None,
    prefetch_factor=2,
    memory_format=torch.contiguous_format,
):
    interpolation = {"bicubic": Image.BICUBIC, "bilinear": Image.BILINEAR}[
        interpolation
    ]
    traindir = os.path.join(data_path, "train")
    transforms_list = [
        transforms.RandomResizedCrop(image_size, interpolation=interpolation),
        transforms.RandomHorizontalFlip(),
    ]
    if augmentation == "disabled":
        pass
    elif augmentation == "autoaugment":
        transforms_list.append(AutoaugmentImageNetPolicy())
    else:
        raise NotImplementedError(f"Automatic augmentation: '{augmentation}' is not supported"
                                  " for PyTorch data loader.")
    train_dataset = datasets.ImageFolder(traindir, transforms.Compose(transforms_list))

    if torch.distributed.is_initialized():
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, shuffle=True
        )
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=workers,
        worker_init_fn=_worker_init_fn,
        pin_memory=True,
        collate_fn=partial(fast_collate, memory_format),
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
    )

    return (
        PrefetchedWrapper(train_loader, start_epoch, num_classes, one_hot),
        len(train_loader),
    )


def get_pytorch_val_loader(
    data_path,
    image_size,
    batch_size,
    num_classes,
    one_hot,
    interpolation="bilinear",
    workers=5,
    _worker_init_fn=None,
    crop_padding=32,
    memory_format=torch.contiguous_format,
    prefetch_factor=2,
):
    interpolation = {"bicubic": Image.BICUBIC, "bilinear": Image.BILINEAR}[
        interpolation
    ]
    valdir = os.path.join(data_path, "val")
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose(
            [
                transforms.Resize(
                    image_size + crop_padding, interpolation=interpolation
                ),
                transforms.CenterCrop(image_size),
            ]
        ),
    )

    if torch.distributed.is_initialized():
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False
        )
    else:
        val_sampler = None

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        sampler=val_sampler,
        batch_size=batch_size,
        shuffle=(val_sampler is None),
        num_workers=workers,
        worker_init_fn=_worker_init_fn,
        pin_memory=True,
        collate_fn=partial(fast_collate, memory_format),
        drop_last=False,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
    )

    return PrefetchedWrapper(val_loader, 0, num_classes, one_hot), len(val_loader)


class SynteticDataLoader(object):
    def __init__(
        self,
        batch_size,
        num_classes,
        num_channels,
        height,
        width,
        one_hot,
        memory_format=torch.contiguous_format,
    ):
        input_data = (
            torch.randn(batch_size, num_channels, height, width)
            .contiguous(memory_format=memory_format)
            .cuda()
            .normal_(0, 1.0)
        )
        if one_hot:
            input_target = torch.empty(batch_size, num_classes).cuda()
            input_target[:, 0] = 1.0
        else:
            input_target = torch.randint(0, num_classes, (batch_size,))
        input_target = input_target.cuda()

        self.input_data = input_data
        self.input_target = input_target

    def __iter__(self):
        while True:
            yield self.input_data, self.input_target


def get_synthetic_loader(
    data_path,
    image_size,
    batch_size,
    num_classes,
    one_hot,
    interpolation=None,
    augmentation=None,
    start_epoch=0,
    workers=None,
    _worker_init_fn=None,
    memory_format=torch.contiguous_format,
    **kwargs,
):
    return (
        SynteticDataLoader(
            batch_size,
            num_classes,
            3,
            image_size,
            image_size,
            one_hot,
            memory_format=memory_format,
        ),
        -1,
    )
