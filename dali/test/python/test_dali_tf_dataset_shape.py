import tensorflow as tf
import nvidia.dali.ops as ops
import nvidia.dali.pipeline as pipeline
import nvidia.dali.plugin.tf as dali_tf
import nvidia.dali.types as dali_types

import numpy as np

import os
from nose.tools import assert_equals, raises
import itertools
import warnings

data_path = os.path.join(os.environ['DALI_EXTRA_PATH'], 'db/single/jpeg/')
file_list_path = os.path.join(data_path, 'image_list.txt')

def dali_pipe_batch_1(shapes, types, as_single_tuple = False):
    class TestPipeline(pipeline.Pipeline):
        def __init__(self, **kwargs):
            super(TestPipeline, self).__init__(**kwargs)
            self.reader = ops.FileReader(file_root=data_path, file_list=file_list_path)
            self.decoder = ops.ImageDecoder(device='mixed')

        def define_graph(self):
            data, label = self.reader()
            image = self.decoder(data)
            return image

    pipe = TestPipeline(batch_size=1, seed=0)
    ds = dali_tf.DALIDataset(pipe, batch_size=1, output_dtypes=types, output_shapes=shapes)
    # for clarity, we could have used the previous `pipe`
    pipe_ref = TestPipeline(batch_size=1, seed=0, device_id=0, num_threads=4)
    pipe_ref.build()

    ds_iter = iter(ds)
    # See if the iteration over different images works
    if as_single_tuple:
        shapes = shapes[0]
    for i in range(10):
        image, = ds_iter.next()
        image_ref, = pipe_ref.run()
        if shapes is None or len(shapes) == 4:
            assert_equals(image.shape, ([1,] + image_ref[0].shape()))
        else:
            assert_equals(image.shape, image_ref[0].shape())


def test_batch_1_different_shapes():
    for shape in [None, (None, None, None, None), (None, None, None),
                  (1, None, None, None), (1, None, None, 3), (None, None, 3)]:
        yield dali_pipe_batch_1, shape, tf.uint8
        yield dali_pipe_batch_1, (shape,), (tf.uint8,), True

# Dummy wrapper expecting mix of tuple/not-tuple in arguments
@raises(ValueError, TypeError, tf.errors.InvalidArgumentError)
def dali_pipe_batch_1_raises(shapes, types):
    dali_pipe_batch_1(shapes, types)


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

    pipe = TestPipeline(batch_size=batch, seed=0)
    ds = dali_tf.DALIDataset(pipe, batch_size=batch, output_dtypes=types, output_shapes=shapes)
    ds_iter = iter(ds)
    for i in range(10):
        image, = ds_iter.next()
        if shapes is None or len(shapes) == 4:
            assert_equals(image.shape, (batch, 200, 200, 3))
        else:
            assert_equals(image.shape, (200, 200, 3))


def test_batch_N_valid_shapes():
    for batch in [1, 10]:
        # No shape
        yield dali_pipe_batch_N, None, tf.uint8, batch
        # Full shape
        output_shape = (batch, 200, 200, 3)
        for i in range(2 ** len(output_shape)):
            noned_shape = tuple([(dim if i & (2 ** idx) else None) for idx, dim in enumerate(output_shape)])
            yield dali_pipe_batch_N, noned_shape, tf.uint8, batch
    # Omitted batch of size `1`
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

    pipe = TestPipeline(batch_size=batch, seed=0)
    ds = dali_tf.DALIDataset(pipe, batch_size=batch, output_dtypes=types, output_shapes=shapes)
    ds_iter = iter(ds)
    for i in range(10):
        image, label = ds_iter.next()
        if shapes is None or shapes[0] is None or len(shapes[0]) == 4:
            assert_equals(image.shape, (batch, 200, 200, 3))
        else:
            assert_equals(image.shape, (200, 200, 3))
        if shapes is None or shapes[1] is None or len(shapes[1]) == 2:
            assert_equals(label.shape, (batch, 1))
        else:
            assert_equals(label.shape, (batch,))

def test_multiple_input_valid_shapes():
    for batch in [1, 10]:
        for shapes in [None, (None, None), ((batch, 200, 200, 3), None), (None, (batch, 1)), (None, (batch,))]:
            yield dali_pipe_multiple_out, shapes, (tf.uint8, tf.int32), batch

@raises(ValueError, TypeError, tf.errors.InvalidArgumentError)
def dali_pipe_multiple_out_raises(shapes, types, batch):
    dali_pipe_multiple_out(shapes, types, batch)

def test_multiple_input_invalid():
    for batch in [1, 10]:
        for shapes in [(None,), (batch, 200, 200, 3, None), (None, None, None)]:
            yield dali_pipe_multiple_out_raises, shapes, (tf.uint8, tf.uint8), batch


def dali_pipe_artificial_shape(shapes, tf_type, dali_type, batch):
    class TestPipeline(pipeline.Pipeline):
        def __init__(self, **kwargs):
            super(TestPipeline, self).__init__(**kwargs)
            self.constant = ops.Constant(dtype=dali_type, idata=[1,1], shape=[1, 2, 1])

        def define_graph(self):
            return self.constant()

    pipe = TestPipeline(batch_size=batch, seed=0)
    ds = dali_tf.DALIDataset(pipe, batch_size=batch, output_dtypes=tf_type, output_shapes=shapes)
    ds_iter = iter(ds)
    for i in range(10):
        out, = ds_iter.next()
        if len(shapes) == 4:
            assert_equals(out.shape, (batch, 1, 2, 1))
        if len(shapes) == 3:
            assert_equals(out.shape, (batch, 1, 2))
        if len(shapes) == 2:
            assert_equals(out.shape, (batch, 2,))
        if len(shapes) == 1:
            assert_equals(out.shape, (2,))

def test_artificial_match():
    for batch in [1, 10]:
        for shape in [(None, None, None, None), (None, None, 2), (batch, None, None, None),
                    (batch, None, 2)]:
            yield dali_pipe_artificial_shape, shape, tf.uint8, dali_types.UINT8, batch
    yield dali_pipe_artificial_shape, (10, 2), tf.uint8, dali_types.UINT8, 10
    yield dali_pipe_artificial_shape, (2,), tf.uint8, dali_types.UINT8, 1

# Dummy wrapper expecting mix of tuple/not-tuple in arguments
@raises(ValueError, TypeError, tf.errors.InvalidArgumentError)
def dali_pipe_artificial_shape_raises(shapes, tf_type, dali_type, batch):
    dali_pipe_artificial_shape(shapes, tf_type, dali_type, batch)

def test_artificial_no_match():
    batch = 10
    for shape in [(batch + 1, None, None, None), (None, None, 3), (batch, 2, 1, 1)]:
        yield dali_pipe_artificial_shape_raises, shape, tf.uint8, dali_types.UINT8, batch


def dali_pipe_types(tf_type, dali_type):
    class TestPipeline(pipeline.Pipeline):
        def __init__(self, **kwargs):
            super(TestPipeline, self).__init__(**kwargs)
            self.constant = ops.Constant(dtype=dali_type, idata=[1,1], shape=[2])

        def define_graph(self):
            return self.constant()

    pipe = TestPipeline(batch_size=1, seed=0)
    ds = dali_tf.DALIDataset(pipe, batch_size=1, output_dtypes=tf_type)
    ds_iter = iter(ds)
    out, = ds_iter.next()
    assert_equals(out.dtype, tf_type)

# Dummy wrapper expecting mix of tuple/not-tuple in arguments
@raises(ValueError, TypeError, tf.errors.InvalidArgumentError)
def dali_pipe_types_raises(tf_type, dali_type):
    dali_pipe_types(tf_type, dali_type)


# float64 not tested because constand doesn't support it
tf_type_list = [tf.uint8, tf.uint16, tf.uint32, tf.uint64,
                tf.int8, tf.int16, tf.int32, tf.int64,
                tf.bool,
                tf.float16, tf.float32]
dali_type_list = [dali_types.UINT8, dali_types.UINT16, dali_types.UINT32, dali_types.UINT64,
                    dali_types.INT8, dali_types.INT16, dali_types.INT32, dali_types.INT64,
                    dali_types.BOOL,
                    dali_types.FLOAT16, dali_types.FLOAT]
matching_types = list(zip(tf_type_list, dali_type_list))
all_types = itertools.product(tf_type_list, dali_type_list)
not_matching_types = list(set(all_types).difference(set(matching_types)))

def test_type_returns():
    for tf_t, dali_t in matching_types:
        yield dali_pipe_types, tf_t, dali_t
    for tf_t, dali_t in not_matching_types:
        yield dali_pipe_types_raises, tf_t, dali_t


def dali_pipe_deprecated(dataset_kwargs, shapes, tf_type, dali_type, batch, expected_warnings_count):
    class TestPipeline(pipeline.Pipeline):
        def __init__(self, **kwargs):
            super(TestPipeline, self).__init__(**kwargs)
            self.constant = ops.Constant(dtype=dali_type, idata=[1,1], shape=[2])

        def define_graph(self):
            return self.constant()

    pipe = TestPipeline(batch_size=batch, seed=0)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ds = dali_tf.DALIDataset(pipe, batch_size=batch, **dataset_kwargs)
        assert_equals(len(w), expected_warnings_count)
        ds_iter = iter(ds)
        for i in range(10):
            out, = ds_iter.next()
            if isinstance(shapes, int) or len(shapes) == 1:
                assert_equals(out.shape, (2,))
            else:
                assert_equals(out.shape, (batch, 2))
            assert_equals(out.dtype, tf_type)

@raises(ValueError, TypeError, tf.errors.InvalidArgumentError)
def dali_pipe_deprecated_raises(*args):
    dali_pipe_deprecated(*args)

def test_deprecated():
    yield dali_pipe_deprecated, \
            { "shapes": 2, "dtypes": tf.uint8 }, 2, tf.uint8, dali_types.UINT8, 1, 2
    yield dali_pipe_deprecated, \
            { "shapes": [4, 2], "dtypes": tf.uint8 }, [4, 2], tf.uint8, dali_types.UINT8, 4, 2
    yield dali_pipe_deprecated, \
            { "shapes": [[4, 2]], "dtypes": [tf.uint8] }, [4, 2], tf.uint8, dali_types.UINT8, 4, 2
    yield dali_pipe_deprecated, \
            { "output_shapes": 2, "dtypes": tf.uint8 }, 2, tf.uint8, dali_types.UINT8, 1, 1
    yield dali_pipe_deprecated, \
            { "output_shapes": (4, 2), "dtypes": tf.uint8 }, [4, 2], tf.uint8, dali_types.UINT8, 4, 1
    yield dali_pipe_deprecated, \
            { "output_shapes": ((4, 2),), "dtypes": [tf.uint8] }, [4, 2], tf.uint8, dali_types.UINT8, 4, 1
    yield dali_pipe_deprecated, \
            { "shapes": 2, "output_dtypes": tf.uint8 }, 2, tf.uint8, dali_types.UINT8, 1, 1
    yield dali_pipe_deprecated, \
            { "shapes": [4, 2], "output_dtypes": tf.uint8 }, [4, 2], tf.uint8, dali_types.UINT8, 4, 1
    yield dali_pipe_deprecated, \
            { "shapes": [[4, 2]], "output_dtypes": (tf.uint8,) }, [4, 2], tf.uint8, dali_types.UINT8, 4, 1


def test_deprecated_double_def():
    yield dali_pipe_deprecated_raises, \
            { "shapes": 2, "output_shapes": 2, "dtypes": tf.uint8 }, 2, tf.uint8, dali_types.UINT8, 1, 2
    yield dali_pipe_deprecated_raises, \
            { "shapes": 2, "dtypes": tf.uint8, "output_dtypes": tf.uint8 }, 2, tf.uint8, dali_types.UINT8, 1, 2

def test_no_output_dtypes():
    yield dali_pipe_deprecated_raises, \
            { "shapes": 2, }, 2, tf.uint8, dali_types.UINT8, 1, 2