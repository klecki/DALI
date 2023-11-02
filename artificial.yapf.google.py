# Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import List, Tuple, Any, Optional
import os
import socket
import threading
import warnings
import multiprocessing
import copy
from collections import deque
from nvidia.dali._multiproc.shared_batch import deserialize_batch, import_numpy, read_shm_message, \
    BufShmChunk, SharedBatchWriter, write_shm_message, _align_up as align_up
from nvidia.dali._multiproc.shared_queue import ShmQueue


def signature_test0(argument1, argument2):
    pass

def signature_test1(argument1, argument2,):
    pass


def signature_test2(argument1, argument2, argument3, argument4, argument5, argument6, argument7,
                    argument8):
    pass


def signature_test3(argument1, argument2, argument3, argument4, argument5,
                    argument6, argument7,
                    argument8):
    pass

def signature_test4(argument1, argument2, argument3, argument4, argument5, argument6, argument7,
                    argument8, argument9, argument10, argument11):
    pass


def signature_test5(argument1, argument2, argument3, argument4, argument5,
                    argument6, argument7,
                    argument8, argument9, argument10, argument11):
    pass

def signature_test6(argument1, argument2, argument3, argument4, argument5,
                    argument6, argument7,
                    argument8, argument9, argument10, argument11,):
    pass

class ShmChunkManager:
    """Two dimensional buffer of shared memory chunks (queue_depth X num_minibatches),
       chunks can be accessed either by providing two coordinates or via shm chunk's unique id.
       Each ExternalSource callback gets its own buffer, first dimension is cycled
       over when scheduling and receiving consecutive batches, second dimension is
       used to separate minibatches."""

    def __init__(self, shm_pool: List[BufShmChunk], queue_depth, initial_chunk_capacity,
                 num_minibatches):
        if queue_depth < 1:
            raise RuntimeError("Prefetch queue must have at least one element")
        if initial_chunk_capacity < 1:
            raise RuntimeError("Buffer chunk capacity must be a positive integer")
        self.shm_pool = shm_pool
        self.queue_depth = queue_depth
        self.initial_chunk_capacity = align_up(initial_chunk_capacity,
                                               SharedBatchWriter.BUFFER_ALIGNMENT)
        self.num_minibatches = num_minibatches
        self.chunks_ids_by_pos = []
        for _ in range(self.queue_depth):
            self.chunks_ids_by_pos.append([
                self.allocate_chunk(self.initial_chunk_capacity)
                for _ in range(self.num_minibatches)
            ])
        self.chunks_ids = [chunk_id for dest_buf in self.chunks_ids_by_pos for chunk_id in dest_buf]


def get_required_kwargs(fun, skip_positional=0):
    """
    Returns the list of names of args/kwargs without defaults from
    `fun` signature.
    """
    sig = inspect.signature(fun)
    # the params from signature with up to skip_positional filtered out
    # (less only if there is not enough of positional args)
    params = [(name, param) for i, (name, param) in enumerate(sig.parameters.items())
              if i >= skip_positional or param.kind not in
              [inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.POSITIONAL_ONLY]]
    return [
        name for name, param in params if param.default is inspect.Parameter.empty
        and param.kind in [inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY]
    ]



def foo_error():
    raise TypeError(f"Argument to arithmetic operation not supported."
                    f"Got {str(type(input))}, expected a return value from other"
                    f"DALI Operator  or a constant value of type 'bool', 'int', "
                    f"'float' or 'nvidia.dali.types.Constant'.")
    raise TypeError(
        f"Argument to arithmetic operation not supported. Got {str(type(input))}, expected a return value from other"
        "DALI Operator  or a constant value of type 'bool', 'int', 'float' or 'nvidia.dali.types.Constant'."
    )


def DALIIteratorWrapper(pipeline=None, serialized_pipeline=None, sparse=[], shapes=[], dtypes=[],
                        batch_size=-1, prefetch_queue_depth=2, **kwargs):
    """
  TF Plugin Wrapper

  This operator works in the same way as DALI TensorFlow plugin, with the exception that it also
  accepts Pipeline objects as an input, which are serialized internally. For more information,
  see :meth:`nvidia.dali.plugin.tf.DALIRawIterator`.
  """
    if type(prefetch_queue_depth) is dict:
        exec_separated = True
        cpu_prefetch_queue_depth = prefetch_queue_depth["cpu_size"]
        gpu_prefetch_queue_depth = prefetch_queue_depth["gpu_size"]
    elif type(prefetch_queue_depth) is int:
        exec_separated = False
        cpu_prefetch_queue_depth = -1  # dummy: wont' be used
        gpu_prefetch_queue_depth = prefetch_queue_depth


_known_types = {
    DALIDataType.INT8: ("int", int),
    DALIDataType.INT16: ("int", int),
    DALIDataType.INT32: ("int", int),
    DALIDataType.INT64: ("int", int),
    DALIDataType.UINT8: ("int", int),
    DALIDataType.UINT16: ("int", int),
    DALIDataType.UINT32: ("int", int),
    # DALIDataType.UINT64: ("int", int), # everything else fits into the Python int
    DALIDataType.FLOAT: ("float", float),
    DALIDataType.BOOL: ("bool", bool),
    DALIDataType.STRING: ("str", str),
    DALIDataType._BOOL_VEC: ("bool", _to_list(bool)),
    DALIDataType._INT32_VEC: ("int", _to_list(int)),
    DALIDataType._STRING_VEC: ("str", _to_list(str)),
    DALIDataType._FLOAT_VEC: ("float", _to_list(float)),
    DALIDataType.IMAGE_TYPE: ("nvidia.dali.types.DALIImageType", lambda x: DALIImageType(int(x))),
    DALIDataType.DATA_TYPE: ("nvidia.dali.types.DALIDataType", lambda x: DALIDataType(int(x))),
    DALIDataType.INTERP_TYPE:
    ("nvidia.dali.types.DALIInterpType", lambda x: DALIInterpType(int(x))),
    DALIDataType.TENSOR_LAYOUT: (":ref:`layout str<layout_str_doc>`", lambda x: str(x)),
    DALIDataType.PYTHON_OBJECT: ("object", lambda x: x),
    DALIDataType._TENSOR_LAYOUT_VEC:
    (":ref:`layout str<layout_str_doc>`", _to_list(lambda x: str(x))),
    DALIDataType._DATA_TYPE_VEC: ("nvidia.dali.types.DALIDataType",
                                  _to_list(lambda x: DALIDataType(int(x))))
}
# common type names used by numpy, torch and possibly
_type_name_to_dali_type = {
    'bool': DALIDataType.BOOL,
    'boolean': DALIDataType.BOOL,
    'int8': DALIDataType.INT8,
    'sbyte': DALIDataType.INT8,
    'uint8': DALIDataType.UINT8,
    'byte': DALIDataType.UINT8,
    'ubyte': DALIDataType.UINT8,
    'int16': DALIDataType.INT16,
    'short': DALIDataType.INT16,
    'uint16': DALIDataType.UINT16,
    'ushort': DALIDataType.UINT16,
    'int32': DALIDataType.INT32,
    'uint32': DALIDataType.UINT32,
    'int64': DALIDataType.INT64,
    'long': DALIDataType.INT64,
    'uint64': DALIDataType.UINT64,
    'ulong': DALIDataType.UINT64,
    'half': DALIDataType.FLOAT16,
    'float16': DALIDataType.FLOAT16,
    'float': DALIDataType.FLOAT,
    'float32': DALIDataType.FLOAT,
    'float64': DALIDataType.FLOAT64,
    'double': DALIDataType.FLOAT64,
}


_bool_types = [DALIDataType.BOOL]
_int_types = [
    DALIDataType.INT8, DALIDataType.INT16, DALIDataType.INT32, DALIDataType.INT64,
    DALIDataType.UINT8, DALIDataType.UINT16, DALIDataType.UINT32, DALIDataType.UINT64
]
_int_types2 = [
    DALIDataType.INT8, DALIDataType.INT16, DALIDataType.INT32, DALIDataType.INT64,
    DALIDataType.UINT8, DALIDataType.UINT16, DALIDataType.UINT32, DALIDataType.UINT64,
]
_float_types = [DALIDataType.FLOAT16, DALIDataType.FLOAT, DALIDataType.FLOAT64]
_float_types2 = [
    DALIDataType.FLOAT16,
    DALIDataType.FLOAT,
    DALIDataType.FLOAT64,
]
