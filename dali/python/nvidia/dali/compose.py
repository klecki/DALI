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


from nvidia.dali.pipeline import Pipeline as _Pipeline
from nvidia.dali.ops import FileReader as _FileReader, \
        CaffeReader as _CaffeReader, \
        Caffe2Reader as _Caffe2Reader, \
        MXNetReader as _MXNetReader, \
        TFRecordReader as _TFRecordReader


_allowed_regular_readers = (_FileReader, _CaffeReader, _Caffe2Reader, _MXNetReader)
_allowed_readers = _allowed_regular_readers + (_TFRecordReader,)

class ComposePipeline(_Pipeline):
    def __init__(self, reader, transforms = [], *, tfrecord_input = None, tfrecord_label = None, **kwargs):
                #  batch_size = -1, num_threads = -1, device_id = -1, seed = -1,
                #  exec_pipelined=True, prefetch_queue_depth=2,
                #  exec_async=True, bytes_per_sample=0,
                #  set_affinity=False, max_streams=-1, default_cuda_stream_priority = 0):
        super(ComposePipeline, self).__init__(**kwargs)
        self.reader = reader
        if isinstance(self.reader, _TFRecordReader):
            if not isinstance(tfrecord_input, str):
                # TODO
                raise ValueError("`tfrecord_input` is expected to represent label, got {}".format(tfrecord_input))
            if not isinstance(tfrecord_label, str):
                # TODO
                raise ValueError("`tfrecord_label` is expected to represent label, got {}".format(tfrecord_input))
            self.tf = True
            self.tfrecord_input = tfrecord_input
            self.tfrecord_label = tfrecord_label
        else if not isinstance(self.reader, _allowed_regular_readers):
            self.tf = False
            raise TypeError(("The `reader` argument is not one of allowed reader types. " + \
                "Expected one of {}, got: {}.").format(str(_allowed_readers), type(self.reader)))
        if not isinstance(transforms, list):
            raise TypeError(("`transforms` is expected to be a list of DALI Operators. Got {}".format(type(transforms))))
        for t in transforms:
            # if not isinstance ... DALI OPERATOR INSTANCE
        self.transforms = transforms

    def define_graph(self):
        if self.tf:
            tf_inputs = self.reader()
            data = tf_inputs[self.tfrecord_input]
            labels = tf_inputs[self.tfrecord_labels]
        else:
            data, labels = self.reader()
        for t in self.transforms:
            data = t(data)
        return data, labels



