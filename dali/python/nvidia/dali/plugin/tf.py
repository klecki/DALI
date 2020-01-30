# Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

import tensorflow as tf
from tensorflow.python.data.util import nest
from tensorflow.python.framework import tensor_shape

import os
import glob
from collections import Iterable
import re
from distutils.version import StrictVersion
import warnings


_tf_plugins = glob.glob(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'libdali_tf*.so'))
_dali_tf_module = None
# Order: 'current', prebuilt for current TF version, prebuilt for other TF versions
_tf_version = re.search("(\d+.\d+).\d+", tf.__version__).group(1)
_tf_version_underscore = _tf_version.replace('.', '_')
_dali_tf_current = list(filter(lambda x: 'current' in x, _tf_plugins))
_dali_tf_prebuilt_tf_ver = list(filter(lambda x: _tf_version_underscore in x, _tf_plugins))
_dali_tf_prebuilt_others = list(filter(lambda x: 'current' not in x and _tf_version_underscore not in x, _tf_plugins))
_processed_tf_plugins = _dali_tf_current + _dali_tf_prebuilt_tf_ver + _dali_tf_prebuilt_others

first_error = None
for _libdali_tf in _processed_tf_plugins:
  try:
    _dali_tf_module = tf.load_op_library(_libdali_tf)
    break
  # if plugin is not compatible skip it
  except tf.errors.NotFoundError as error:
    if first_error == None:
      first_error = error
else:
  raise first_error or Exception('No matching DALI plugin found for installed TensorFlow version')

_dali_tf = _dali_tf_module.dali

_dali_tf.__doc__ = _dali_tf.__doc__ + """

    Please keep in mind that TensorFlow allocates almost all available device memory by default. This might cause errors in
    DALI due to insufficient memory. On how to change this behaviour please look into the TensorFlow documentation, as it may
    differ based on your use case.
"""

def serialize_pipeline(pipeline):
  try:
    return pipeline.serialize()
  except RuntimeError as e:
    raise RuntimeError("Error during pipeline initialization. Note that some operators "
                       "(e.g. Python Operators) cannot be used with "
                       "tensorflow data set API and DALIIterator.") from e


def DALIIteratorWrapper(pipeline = None, serialized_pipeline = None, sparse = [],
                        shapes = [], dtypes = [], batch_size = -1, prefetch_queue_depth = 2, **kwargs):
  """
TF Plugin Wrapper

This operator works in the same way as DALI TensorFlow plugin, with the exception that is also accepts Pipeline objects as the input and serializes it internally. For more information, please look **TensorFlow Plugin API reference** in the documentation.
  """
  if type(prefetch_queue_depth) is dict:
      exec_separated = True
      cpu_prefetch_queue_depth = prefetch_queue_depth["cpu_size"]
      gpu_prefetch_queue_depth = prefetch_queue_depth["gpu_size"]
  elif type(prefetch_queue_depth) is int:
      exec_separated = False
      cpu_prefetch_queue_depth = -1 # dummy: wont' be used
      gpu_prefetch_queue_depth = prefetch_queue_depth

  if serialized_pipeline is None:
    serialized_pipeline = serialize_pipeline(pipeline)

  # if batch_size is not provided we need to extract if from the shape arg
  if (not isinstance(shapes, Iterable) or len(shapes) == 0) and batch_size == -1:
    raise Exception('shapes and batch_size arguments cannot be empty, '
                    'please provide at leas one shape argument element with the BATCH size or set batch_size')

  if len(sparse) > 0 and sparse[0] and batch_size == -1:
    if isinstance(shapes[0], Iterable) and len(shapes[0]) == 1:
      shapes[0] = (shapes[0][0], 1)
    else:
      shapes[0] = (shapes[0], 1)

  # shapes and dtypes need to take into account that sparse tensor will produce 3 output tensors
  new_dtypes = []
  new_shapes = []
  for i in range(len(dtypes)):
    if i < len(sparse) and sparse[i]:
      # indices type of sparse tensor is tf.int64
      new_dtypes.append(tf.int64)
      new_dtypes.append(dtypes[i])
      # dense shape type of sparse tensor is tf.int64
      new_dtypes.append(tf.int64)
      if len(shapes) > i and len(shapes[i]) > 0:
        new_shapes.append((shapes[i][0], 1))
        new_shapes.append((shapes[i][0]))
      else:
        new_shapes.append(())
        new_shapes.append(())
      new_shapes.append(())
    else:
      new_dtypes.append(dtypes[i])
      if len(shapes) > i:
        new_shapes.append(shapes[i])

  # gpu_prefetch_queue_depth correspond to the global queue depth in the uniform case
  out = _dali_tf(serialized_pipeline=serialized_pipeline, shapes=new_shapes, dtypes=new_dtypes, sparse=sparse, batch_size=batch_size,
                 exec_separated=exec_separated, gpu_prefetch_queue_depth=gpu_prefetch_queue_depth, cpu_prefetch_queue_depth=cpu_prefetch_queue_depth, **kwargs)
  new_out = []
  j = 0
  for i in range(len(dtypes)):
    if i < len(sparse) and sparse[i]:
      new_out.append(tf.SparseTensor(indices=out[j], values=out[j + 1], dense_shape=out[j + 2]))
      j += 3
    else:
      new_out.append(out[j])
      j += 1
  return new_out

def DALIIterator():
    return DALIIteratorWrapper

# Vanilla raw operator legacy
def DALIRawIterator():
    return _dali_tf


def _get_tf_version():
  return StrictVersion(tf.__version__)


MIN_TENSORFLOW_VERSION = StrictVersion('1.15')
def dataset_compatible_tensorflow():
    return StrictVersion(tf.__version__) >= MIN_TENSORFLOW_VERSION


if dataset_compatible_tensorflow():
  from tensorflow.python.framework import ops
  from tensorflow.python.data.ops import dataset_ops
  from tensorflow.python.data.util import structure
  import functools


  class _DALIDatasetV2(dataset_ops.DatasetSource):
    def __init__(
      self,
      pipeline,
      output_dtypes,
      output_shapes = None,
      *,
      batch_size = 1,
      num_threads = 4,
      device_id = 0,
      exec_separated = False,
      prefetch_queue_depth = 2,
      cpu_prefetch_queue_depth = 2,
      gpu_prefetch_queue_depth = 2,
      dtypes=None,
      shapes=None):

      # Python will handle the case of mandatory `pipeline` and `output_dtypes` arguments
      if dtypes is not None:
        warnings.warn(("Use of argument `dtypes` is deprecated. Please use `output_dtypes` instead. " \
            + "`output_dtypes` should be provided as a tuple."), Warning, stacklevel=2)

      if shapes is not None:
        if output_shapes is not None:
          raise ValueError("Usage of `shapes` is deprecated in favor of `output_shapes`. " +
            "Both arguments were provided.")
        # show only this warning
        warnings.warn(("Use of argument `shapes` is deprecated. Please use `output_shapes` instead. " \
            + "`output_shapes` should be provided as a tuple."), Warning, stacklevel=2)
        if isinstance(shapes, list):
          output_shapes = tuple(list_arg)

      if not self._check_output_dtypes(output_dtypes):
        raise TypeError(("`output_dtypes` should be provided as single tf.DType value " +
            "or a tuple of tf.DType values. Got value `{}` of type `{}`.") \
                .format(output_dtypes, type(output_dtypes)))

      if output_shapes is None:
        output_shapes = nest.map_structure(lambda _: tensor_shape.TensorShape(None), output_dtypes)
      else:
        output_shapes = nest.map_structure_up_to(output_dtypes, tensor_shape.as_shape, output_shapes)

      if not isinstance(output_dtypes, tuple):
        output_dtypes = (output_dtypes,)
        output_shapes = (output_shapes,)

      output_classes = nest.map_structure(lambda _: ops.Tensor, output_dtypes)

      self._pipeline = serialize_pipeline(pipeline)
      self._batch_size = batch_size
      self._num_threads = num_threads
      self._device_id = device_id
      self._exec_separated = exec_separated
      self._prefetch_queue_depth = prefetch_queue_depth
      self._cpu_prefetch_queue_depth = cpu_prefetch_queue_depth
      self._gpu_prefetch_queue_depth = gpu_prefetch_queue_depth
      self._output_shapes = output_shapes
      self._output_dtypes = output_dtypes

      self._structure = structure.convert_legacy_structure(
        self._output_dtypes, self._output_shapes, output_classes)

      super(_DALIDatasetV2, self).__init__(self._as_variant_tensor())

    def _check_output_dtypes(self, output_dtypes):
      """Check whether output_dtypes is instance of tf.DType or tuple of tf.DType
      """
      if not isinstance(output_dtypes, tf.DType):
        if isinstance(output_dtypes, tuple) \
            and all(isinstance(dtype, tf.DType) for dtype in output_dtypes):
          return True
        else:
          return False
      else:
        return True

    def _handle_deprecation(self, tuple_arg, list_arg, arg_name):
        """
        Check usage of old `shapes` and `dtypes` (the list_arg) argument
        instead of `output_shapes` and `output_dtypes` (the tuple_arg).
        Convert list to tuple if the list_arg was provided.
        """





    @property
    def element_spec(self):
      return self._structure


    @property
    def _element_structure(self):
      return self._structure


    def _as_variant_tensor(self):
      return _dali_tf_module.dali_dataset(
        pipeline = self._pipeline,
        batch_size = self._batch_size,
        num_threads = self._num_threads,
        device_id = self._device_id,
        exec_separated = self._exec_separated,
        prefetch_queue_depth = self._prefetch_queue_depth,
        cpu_prefetch_queue_depth = self._cpu_prefetch_queue_depth,
        gpu_prefetch_queue_depth = self._gpu_prefetch_queue_depth,
        output_shapes = self._output_shapes,
        output_dtypes = self._output_dtypes)


  if _get_tf_version() < StrictVersion('2.0'):
    class DALIDataset(dataset_ops.DatasetV1Adapter):
      @functools.wraps(_DALIDatasetV2.__init__)
      def __init__(self, **kwargs):
        wrapped = _DALIDatasetV2(**kwargs)
        super(DALIDataset, self).__init__(wrapped)
  else:
    DALIDataset = _DALIDatasetV2

else:
  class DALIDataset:
    def __init__(
      self,
      pipeline,
      output_dtypes,
      output_shapes = None,
      *,
      batch_size = 1,
      num_threads = 4,
      device_id = 0,
      exec_separated = False,
      prefetch_queue_depth = 2,
      cpu_prefetch_queue_depth = 2,
      gpu_prefetch_queue_depth = 2,
      dtypes=None,
      shapes=None):
      raise RuntimeError('DALIDataset is not supported for detected version of TensorFlow.  DALIDataset supports versions: 1.15, 2.0')

DALIDataset.__doc__ =  """Creates a `DALIDataset` compatible with tf.data.Dataset from a DALI pipeline. It supports TensorFlow 1.15 and 2.0


    Please keep in mind that TensorFlow allocates almost all available device memory by default. This might cause errors in
    DALI due to insufficient memory. On how to change this behaviour please look into the TensorFlow documentation, as it may
    differ based on your use case.


    Parameters
    ----------
    `pipeline` : `nvidia.dali.Pipeline`
        defining the augmentations to be performed.
    `output_dtypes`: `tf.DType` or `tuple` of `tf.DType`
        expected output types
    `output_shapes`: tuple of shapes, optional
        expected output shapes. If provided, must match arity of the `output_dtypes`.
        When set to None, DALI will infer the shapes on its own.
        Individual shapes can be also set to None or contain None to indicate unknown dimensions.
        If specified must be compatible with shape returned from DALI Pipeline
        and with `batch_size` argument which will be the outermost dimension of returned tensors.
        In case of `batch_size = 1` it can be omitted in the shape.
        DALI Dataset will try to match requested shape by squeezing 1-sized dimensions
        from shape obtained from Pipeline.
    `batch_size` : int, optional
        batch size of the pipeline.
    `num_threads` : int, optional
        number of CPU threads used by the pipeline.
    `device_id` : int, optional
        id of GPU used by the pipeline.
    `exec_separated` : bool, optional
        Whether to execute the pipeline in a way that enables
        overlapping CPU and GPU computation, typically resulting
        in faster execution speed, but larger memory consumption.
    `prefetch_queue_depth` : int, optional
        depth of the executor queue. Deeper queue makes DALI more
        resistant to uneven execution time of each batch, but it also
        consumes more memory for internal buffers.
        Value will be used with `exec_separated` set to False.
    `cpu_prefetch_queue_depth` : int, optional
        depth of the executor cpu queue. Deeper queue makes DALI more
        resistant to uneven execution time of each batch, but it also
        consumes more memory for internal buffers.
        Value will be used with `exec_separated` set to True.
    `gpu_prefetch_queue_depth` : int, optional
        depth of the executor gpu queue. Deeper queue makes DALI more
        resistant to uneven execution time of each batch, but it also
        consumes more memory for internal buffers.
        Value will be used with `exec_separated` set to True.

    Returns
    -------
    `DALIDataset` object based on DALI pipeline and compatible with `tf.data.Dataset` API.

    """

DALIIterator.__doc__ = DALIIteratorWrapper.__doc__
DALIRawIterator.__doc__ = _dali_tf.__doc__
