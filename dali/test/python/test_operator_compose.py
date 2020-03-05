# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

from __future__ import print_function
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import nvidia.dali.tfrecord as tfrec
from nvidia.dali.compose import ComposePipeline
import glob
import argparse
import time

import os.path

from test_utils import compare_pipelines, get_dali_extra_path

test_data_root = get_dali_extra_path()
caffe_lmdb_folder = os.path.join(test_data_root, 'db', 'lmdb')
caffe2_lmdb_folder = os.path.join(test_data_root, 'db', 'c2lmdb')
recordio_db_folder = os.path.join(test_data_root, 'db', 'recordio')
tfrecord_db_folder = os.path.join(test_data_root, 'db', 'tfrecord')
jpeg_folder = os.path.join(test_data_root, 'db', 'single', 'jpeg')


def get_transforms(img_decoder_random_crop = False, nhwc = True, fp16 = False, **decoder_kwargs):
    layout = types.NHWC if nhwc else types.NCHW
    out_type = types.FLOAT16 if fp16 else types.FLOAT

    if img_decoder_random_crop:
        transforms = [
            ops.ImageDecoderRandomCrop(device = "mixed", output_type = types.RGB, **decoder_kwargs),
            ops.Resize(device = "gpu", resize_x = 224, resize_y = 224)
        ]
    else:
        transforms = [
            ops.ImageDecoder(device = "mixed", output_type = types.RGB, **decoder_kwargs),
            ops.RandomResizedCrop(device = "gpu", size = (224, 224))
        ]
    cmn = ops.CropMirrorNormalize(device="gpu", output_dtype=out_type,
                                  output_layout=layout, crop=(224, 224),
                                  image_type=types.RGB,
                                  mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                  std=[0.229 * 255,0.224 * 255,0.225 * 255])
    transforms.append(cmn)
    return transforms


class CommonPipeline(Pipeline):
    def __init__(self, data_paths, num_gpus, batch_size, num_threads, device_id, prefetch, fp16, nhwc,
                 decoder_type, reader_queue_depth):
        super(CommonPipeline, self).__init__(batch_size, num_threads, device_id, prefetch_queue_depth=prefetch)
        if decoder_type == 'roi':
            print('Using nvJPEG with ROI decoding')
            self.decode_gpu = ops.ImageDecoderRandomCrop(device = "mixed", output_type = types.RGB)
            self.res = ops.Resize(device="gpu", resize_x=224, resize_y=224)
        else:
            print('Using nvJPEG')
            self.decode_gpu = ops.ImageDecoder(device = "mixed", output_type = types.RGB)
            self.res = ops.RandomResizedCrop(device="gpu", size =(224,224))

        layout = types.NHWC if nhwc else types.NCHW
        out_type = types.FLOAT16 if fp16 else types.FLOAT

        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=out_type,
                                            output_layout=layout,
                                            crop=(224, 224),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                            std=[0.229 * 255,0.224 * 255,0.225 * 255])
        self.coin = ops.CoinFlip(probability=0.5)

    def base_define_graph(self, inputs, labels):
        rng = self.coin()
        images = self.decode_gpu(inputs)
        images = self.res(images)
        # TODO(klecki): no mirroring as of now
        # output = self.cmnp(images.gpu(), mirror=rng)
        output = self.cmnp(images.gpu())
        return (output, labels)


def get_reader(name):
    reader_factory = {
        'MXNetReader':
            lambda kwargs: ops.MXNetReader(path = kwargs['data_paths'][0],
                                           index_path = kwargs['data_paths'][1],
                                           shard_id = kwargs['device_id'],
                                           num_shards = kwargs['num_gpus'],
                                           prefetch_queue_depth = kwargs['reader_queue_depth']),
        'CaffeReader':
            lambda kwargs: ops.CaffeReader(path = kwargs['data_paths'][0],
                                           shard_id = kwargs['device_id'],
                                           num_shards = kwargs['num_gpus'],
                                           prefetch_queue_depth = kwargs['reader_queue_depth']),
        'Caffe2Reader':
            lambda kwargs: ops.Caffe2Reader(path = kwargs['data_paths'][0],
                                      shard_id = kwargs['device_id'],
                                      num_shards = kwargs['num_gpus'],
                                      prefetch_queue_depth = kwargs['reader_queue_depth']),
        'FileReader':
            lambda kwargs: ops.FileReader(file_root = kwargs['data_paths'][0],
                                    shard_id = kwargs['device_id'],
                                    num_shards = kwargs['num_gpus'],
                                    prefetch_queue_depth = kwargs['reader_queue_depth']),
        'TFRecordReader':
            lambda kwargs: ops.TFRecordReader(path = kwargs['path'],
                                        index_path = kwargs['index_path'],
                                        shard_id = kwargs['device_id'],
                                        num_shards = kwargs['num_gpus'],
                                        features = {"image/encoded" : tfrec.FixedLenFeature((), tfrec.string, ""),
                                                    "image/class/label": tfrec.FixedLenFeature([1], tfrec.int64,  -1)
                                        })
    }
    return reader_factory[name]

class MXNetReaderPipeline(CommonPipeline):
    def __init__(self, **kwargs):
        super(MXNetReaderPipeline, self).__init__(**kwargs)
        self.input = get_reader('MXNetReader')(kwargs)

    def define_graph(self):
        images, labels = self.input(name="Reader")
        return self.base_define_graph(images, labels)

class CaffeReadPipeline(CommonPipeline):
    def __init__(self, **kwargs):
        super(CaffeReadPipeline, self).__init__(**kwargs)
        self.input = get_reader('CaffeReader')(kwargs)

    def define_graph(self):
        images, labels = self.input(name="Reader")
        return self.base_define_graph(images, labels)

class Caffe2ReadPipeline(CommonPipeline):
    def __init__(self, **kwargs):
        super(Caffe2ReadPipeline, self).__init__(**kwargs)
        self.input = get_reader('Caffe2Reader')(kwargs)

    def define_graph(self):
        images, labels = self.input(name="Reader")
        return self.base_define_graph(images, labels)

class FileReadPipeline(CommonPipeline):
    def __init__(self, **kwargs):
        super(FileReadPipeline, self).__init__(**kwargs)
        self.input = get_reader('FileReader')(kwargs)

    def define_graph(self):
        images, labels = self.input(name="Reader")
        return self.base_define_graph(images, labels)

class TFRecordPipeline(CommonPipeline):
    def __init__(self, **kwargs):
        super(TFRecordPipeline, self).__init__(**kwargs)
        kwargs['path'] = sorted(glob.glob(kwargs['data_paths'][0]))
        kwargs['index_path'] = sorted(glob.glob(kwargs['data_paths'][1]))
        self.input = get_reader('TFRecordReader')(kwargs)

    def define_graph(self):
        inputs = self.input(name="Reader")
        images = inputs["image/encoded"]
        labels = inputs["image/class/label"]
        return self.base_define_graph(images, labels)

# get_dali_extra_path()


            # tfrecord = sorted(glob.glob(os.path.join(tfrecord_db_folder, '*[!i][!d][!x]')))
            # tfrecord_idx = sorted(glob.glob(os.path.join(tfrecord_db_folder, '*idx')))

test_data = {
            FileReadPipeline: [[jpeg_folder]],
            MXNetReaderPipeline: [[os.path.join(recordio_db_folder, "train.rec"), os.path.join(recordio_db_folder, "train.idx")]],
            CaffeReadPipeline: [[caffe_lmdb_folder]],
            Caffe2ReadPipeline: [[caffe2_lmdb_folder]],
            # TFRecordPipeline: [["/data/imagenet/train-val-tfrecord-480/train-*", "/data/imagenet/train-val-tfrecord-480.idx/train-*"]],
            }

parser = argparse.ArgumentParser(description='Test nvJPEG based RN50 augmentation pipeline with different datasets')
parser.add_argument('-g', '--gpus', default=1, type=int, metavar='N',
                    help='number of GPUs (default: 1)')
parser.add_argument('-b', '--batch', default=2048, type=int, metavar='N',
                    help='batch size (default: 2048)')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-j', '--workers', default=3, type=int, metavar='N',
                    help='number of data loading workers (default: 3)')
parser.add_argument('--prefetch', default=2, type=int, metavar='N',
                    help='prefetch queue depth (default: 2)')
parser.add_argument('--separate_queue', action='store_true',
                    help='Use separate queues executor')
parser.add_argument('--cpu_size', default=2, type=int, metavar='N',
                    help='cpu prefetch queue depth (default: 2)')
parser.add_argument('--gpu_size', default=2, type=int, metavar='N',
                    help='gpu prefetch queue depth (default: 2)')
parser.add_argument('--fp16', action='store_true',
                    help='Run fp16 pipeline')
parser.add_argument('--nhwc', action='store_true',
                    help='Use NHWC data instead of default NCHW')
parser.add_argument('-i', '--iters', default=-1, type=int, metavar='N',
                    help='Number of iterations to run (default: -1 - whole data set)')
parser.add_argument('--epochs', default=2, type=int, metavar='N',
                    help='Number of epochs to run')
parser.add_argument('--decoder_type', default='', type=str, metavar='N',
                    help='allowed: `roi` or for empty default: regular nvjpeg')
parser.add_argument('--reader_queue_depth', default=1, type=int, metavar='N',
                    help='prefetch queue depth (default: 1)')
parser.add_argument('--simulate_N_gpus', default=None, type=int, metavar='N',
                    help='Used to simulate small shard as it would be in a multi gpu setup with this number of gpus. If provided, each gpu will see a shard size as if we were in a multi gpu setup with this number of gpus')
args = parser.parse_args()

N = args.gpus             # number of GPUs
BATCH_SIZE = args.batch   # batch size
LOG_INTERVAL = args.print_freq
WORKERS = args.workers
PREFETCH = args.prefetch
if args.separate_queue:
    PREFETCH = {'cpu_size': args.cpu_size , 'gpu_size': args.gpu_size}
FP16 = args.fp16
NHWC = args.nhwc

DECODER_TYPE = args.decoder_type
READER_QUEUE_DEPTH = args.reader_queue_depth
SIMULATE_N_GPUS = N if args.simulate_N_gpus == None else args.simulate_N_gpus

print("GPUs: {}, batch: {}, workers: {}, prefetch depth: {}, loging interval: {}, fp16: {}, NHWC: {}"
      .format(N, BATCH_SIZE, WORKERS, PREFETCH, LOG_INTERVAL, FP16, NHWC))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.avg_last_n = 0
        self.max_val = 0

    def update(self, val, n=1):
        self.val = val
        self.max_val = max(self.max_val, val)
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

for pipe_name in test_data.keys():
    data_set_len = len(test_data[pipe_name])
    for i, data_set in enumerate(test_data[pipe_name]):
        pipes = [pipe_name(batch_size=BATCH_SIZE, num_threads=WORKERS, device_id=n,
                           num_gpus=SIMULATE_N_GPUS, data_paths=data_set, prefetch=PREFETCH, fp16=FP16,
                           nhwc=NHWC, decoder_type=DECODER_TYPE,
                           reader_queue_depth=READER_QUEUE_DEPTH) for n in range(N)]
        [pipe.build() for pipe in pipes]

        if args.iters < 0:
            iters = pipes[0].epoch_size("Reader")
            assert(all(pipe.epoch_size("Reader") == iters for pipe in pipes))
            iters_tmp = iters
            iters = iters // BATCH_SIZE
            if iters_tmp != iters * BATCH_SIZE:
                iters += 1
            iters_tmp = iters

            iters = iters // SIMULATE_N_GPUS
            if iters_tmp != iters * SIMULATE_N_GPUS:
                iters += 1
        else:
            iters = args.iters

        print ("RUN {0}/{1}: {2}".format(i + 1, data_set_len, pipe_name.__name__))
        print (data_set)
        end = time.time()
        for i in range(args.epochs):
          if i == 0:
              print("Warm up")
          else:
              print("Test run " + str(i))
          data_time = AverageMeter()
          for j in range(iters):
              for pipe in pipes:
                  pipe.run()
              data_time.update(time.time() - end)
              if j % LOG_INTERVAL == 0:
                  print("{} {}/ {}, avg time: {} [s], worst time: {} [s], speed: {} [img/s]"
                  .format(pipe_name.__name__, j + 1, iters, data_time.avg, data_time.max_val, N * BATCH_SIZE / data_time.avg))
              end = time.time()

        print("OK {0}/{1}: {2}".format(i + 1, data_set_len, pipe_name.__name__))
