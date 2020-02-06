import tensorflow as tf
import nvidia.dali.ops as ops
import nvidia.dali.pipeline as pipeline
import nvidia.dali.plugin.tf as dali_tf
import nvidia.dali.types as dali_types

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
    class TestPipeline(pipeline.Pipeline):
        def __init__(self, **kwargs):
            super(TestPipeline, self).__init__(**kwargs)
            self.reader = ops.FileReader(file_root=data_path, file_list=file_list_path)
            self.decoder = ops.ImageDecoder(device='mixed')

        def define_graph(self):
            data, label = self.reader()
            image = self.decoder(data)
            return image

    pipe = TestPipeline(batch_size=1)
    ds = dali_tf.DALIDataset(pipe, batch_size=1, output_dtypes=types, output_shapes=shapes)
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

def dali_pipe_batch_N(shapes, types, batch):
    class TestPipeline(pipeline.Pipeline):
        def __init__(self, **kwargs):
            super(TestPipeline, self).__init__(**kwargs)
            self.reader = ops.FileReader(file_root=data_path, file_list=file_list_path)
            self.decoder = ops.ImageDecoder(device='mixed')
            self.resize = ops.Resize(device="gpu", resize_x = 200, resize_y = 200)

        def define_graph(self):
            data, label = self.reader()
            image = self.decoder(data)
            resized = self.resize(image)
            return resized

    pipe = TestPipeline(batch_size=batch)
    ds = dali_tf.DALIDataset(pipe, batch_size=batch, output_dtypes=types, output_shapes=shapes)
    ds_iter = iter(ds)
    for i in range(10):
        image, = ds_iter.next()
        if shapes is None or len(shapes) == 4:
            assert image.shape == (batch, 200, 200, 3)
        else:
            assert image.shape == (200, 200, 3)


def test_batch_N_valid_shapes():
    for batch in [1, 10]:
        # No shape
        yield dali_pipe_batch_N, None, tf.uint8, batch
        # Full shape
        output_shape = (batch, 200, 200, 3)
        for i in range(2 ** len(output_shape)):
            noned_shape = tuple([(dim if i & (2 ** idx) else None) for idx, dim in enumerate(output_shape)])
            yield dali_pipe_batch_N, noned_shape, tf.uint8, batch
    # Omitted batch = 1
    output_shape = (200, 200, 3)
    for i in range(2 ** len(output_shape)):
        noned_shape = tuple([(dim if i & (2 ** idx) else None) for idx, dim in enumerate(output_shape)])
        yield dali_pipe_batch_N, noned_shape, tf.uint8, 1


def dali_pipe_multiple_out(shapes, types, batch):
    class TestPipeline(pipeline.Pipeline):
        def __init__(self, **kwargs):
            super(TestPipeline, self).__init__(**kwargs)
            self.reader = ops.FileReader(file_root=data_path, file_list=file_list_path)
            self.decoder = ops.ImageDecoder(device='mixed')
            self.resize = ops.Resize(device="gpu", resize_x = 200, resize_y = 200)

        def define_graph(self):
            data, label = self.reader()
            image = self.decoder(data)
            resized = self.resize(image)
            return resized, label

    pipe = TestPipeline(batch_size=batch)
    ds = dali_tf.DALIDataset(pipe, batch_size=batch, output_dtypes=types, output_shapes=shapes)
    ds_iter = iter(ds)
    for i in range(10):
        image, label = ds_iter.next()
        if shapes is None or shapes[0] is None or len(shapes[0]) == 4:
            assert image.shape == (batch, 200, 200, 3)
        else:
            assert image.shape == (200, 200, 3)
        if shapes is None or shapes[1] is None or len(shapes[1]) == 2:
            assert label.shape == (batch, 1)
        else:
            assert label.shape == (batch,)

def test_multiple_input_valid_shapes():

    for batch in [1, 10]:
        for shapes in [None, (None, None), ((batch, 200, 200, 3), None), (None, (batch, 1)), (None, (batch,))]:
            yield dali_pipe_multiple_out, shapes, (tf.uint8, tf.int32), batch
    # for i in range(2 ** len(output_shape)):
    #     noned_shape = tuple([(dim if i & (2 ** idx) else None) for idx, dim in enumerate(output_shape)])
    #     yield dali_pipe_batch_N, noned_shape, tf.uint8, batch
    # yield dali_pipe_batch_N, None, tf.uint8, batch
    # yield dali_pipe_batch_N, (None,), (tf.uint8,), batch
    # batch = 1
    # output_shape = (200, 200, 3)
    # for i in range(2 ** len(output_shape)):
    #     noned_shape = tuple([(dim if i & (2 ** idx) else None) for idx, dim in enumerate(output_shape)])
    #     yield dali_pipe_batch_N, noned_shape, tf.uint8, batch


def dali_pipe_artificial_shape(shapes, types, batch):
    class TestPipeline(pipeline.Pipeline):
        def __init__(self, **kwargs):
            super(TestPipeline, self).__init__(**kwargs)
            self.constant = ops.Constant(dtype=dali_types.UINT8, idata=[1,1], shape=[1, 2, 1])

        def define_graph(self):
            return self.constant()

    pipe = TestPipeline(batch_size=batch)
    ds = dali_tf.DALIDataset(pipe, batch_size=batch, output_dtypes=types, output_shapes=shapes)
    ds_iter = iter(ds)
    for i in range(10):
        out, = ds_iter.next()
        print(out.shape())

def test_aa():
    for batch in [1, 10]:
        for shape in [(None, None, None, None), (None, None, 2), (batch, None, None, None),
                    (batch, None, 2), (batch, 2)]:
            dali_pipe_artificial_shape(shape, tf.uint8, batch)