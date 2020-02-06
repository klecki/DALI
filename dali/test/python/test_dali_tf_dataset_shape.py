import tensorflow as tf
import nvidia.dali.ops
import nvidia.dali.pipeline
import nvidia.dali.plugin.tf

import numpy as np

import os
from nose.tools import assert_equals, raises

data_path = os.path.join(os.environ['DALI_EXTRA_PATH'], 'db/single/jpeg/')
file_list_path = os.path.join(data_path, 'image_list.txt')


def _tf_pipe(shape):
    def generator():
        for file in [data_path + '/241/cute-4074304_1280.jpg', data_path + '/241/dog-1461239_1280.jpg',
                        data_path + '/241/dog-4366295_1920.jpg', data_path + '/695/padlock-406986_640.jpg']:
            data = tf.io.read_file(file)
            image = tf.io.decode_jpeg(data)
            # resized = tf.image.resize(image, (200, 200))
            # yield resized
            yield image
            # yield np.zeros([2, 10, 10, 3])

    ds = tf.data.Dataset.from_generator(generator, (tf.uint8,), shape) \
            # .unbatch().unbatch()
            # .batch(2, output_shapes = )
    for image in ds:
        print(image.shape)


def dali_pipe_batch_1(shapes, types):
    class TestPipeline(nvidia.dali.pipeline.Pipeline):
        def __init__(self, **kwargs):
            super(TestPipeline, self).__init__(**kwargs)
            self.reader = nvidia.dali.ops.FileReader(file_root=data_path, file_list=file_list_path)
            self.decoder = nvidia.dali.ops.ImageDecoder(device='mixed')

        def define_graph(self):
            data, label = self.reader()
            image = self.decoder(data)
            return image

    pipe = TestPipeline(batch_size=1)
    ds = nvidia.dali.plugin.tf.DALIDataset(pipe, batch_size=1, output_dtypes=types, output_shapes=shapes)
    ds_iter = iter(ds)
    # See if the iteration over different images works
    for i in range(10):
        image, = ds_iter.next()
        print(image.shape)


# def test_something():
#     _tf_pipe((None,))
#     # for shape in [None, (None, None, None), (None, None, 3)]:
#     #     yield _tf_pipe, shape

def test_batch_1_different_shapes():
    for shape in [None, (None, None, None, None), (None, None, None),
                  (1, None, None, None), (1, None, None, 3), (None, None, 3)]:
        yield dali_pipe_batch_1, shape, tf.uint8
        yield dali_pipe_batch_1, (shape,), (tf.uint8,)

# Dummy wrapper expecting mix of tuple/not-tuple in arguments
@raises(ValueError, TypeError, tf.errors.InvalidArgumentError)
def dali_pipe_batch_1_raises(shapes, types):
    dali_pipe_batch_1(shapes, types)

# @raises(tf.errors.InvalidArgumentError)
# def dali_pipe_1_raises_InvalidArgumentError(shapes, types):
#     _dali_pipe_1(shapes, types)

def test_batch_1_mixed_tuple():
    for shape in [(None, None, None, None), (None, None, None), (1, None, None, None),
                  (1, None, None, 3), (None, None, 3)]:
        yield dali_pipe_batch_1_raises, shape, (tf.uint8,)
        yield dali_pipe_batch_1_raises, (shape,), tf.uint8

def test_batch_1_wrong_shape():
    for shape in [(2, None, None, None), (None, None, 4), (2, None, None, 4), (None, 0, None, 3)]:
        yield dali_pipe_batch_1_raises, shape, tf.uint8

