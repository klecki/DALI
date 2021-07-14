# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from enum import Enum
from nvidia.dali import types

# TODO(klecki): Remove this dependency somehow?
# Now it is used only for the dummy type
import tensorflow as tf

class _SourceKind(Enum):
    CALLABLE       = 0
    ITERABLE       = 1
    GENERATOR_FUNC = 2

class _SourceDescription:
    """Keep the metadata about the source parameter that was originally passed
    """
    def __init__(self, source, kind: _SourceKind, has_inputs: bool, cycle: str):
        self.source = source
        self.kind = kind
        self.has_inputs = has_inputs
        self.cycle = cycle

    def __str__(self) -> str:
        if self.kind == _SourceKind.CALLABLE:
            return "Callable source " + ("with" if self.has_inputs else "without") + " inputs: `{}`".format(self.source)
        elif self.kind == _SourceKind.ITERABLE:
            return "Iterable (or iterator) source: `{}` with cycle: `{}`.".format(self.source, self.cycle)
        else:
            return "Generator function source: `{}` with cycle: `{}`.".format(self.source, self.cycle)




def _inspect_data(data):
    # TODO(klecki): return actual type and shape
    return tf.int32, None


def get_batch_iterable_from_callback(source_desc):
    """Transform batch callback accepting one argument into an Iterable
    """
    first = source_desc.source(0)
    dtype, shape = _inspect_data(first)

    class CallableBatchIterator:
        first_value = first

        def __init__(self):
            self.iteration = 0
            self.source = source_desc.source

        def __iter__(self):
            self.iteration = 0
            return self

        def __next__(self):
            if self.iteration == 0 and CallableBatchIterator.first_value is not None:
                result = CallableBatchIterator.first_value
                CallableBatchIterator.first_value = None
            else:
                result = self.source(self.iteration)
            self.iteration += 1
            return result

    return CallableBatchIterator, dtype, shape

def get_sample_iterable_from_callback(source_desc, batch_size):
    """Transform sample callback accepting one argument into an Iterable
    """
    first = source_desc.source(types.SampleInfo(0, 0, 0))
    dtype, shape = _inspect_data(first)

    class CallableSampleIterator:
        first_value = first
        def __init__(self):
            self.idx_in_epoch = 0
            self.idx_in_batch = 0
            self.iteration = 0
            self.source = source_desc.source

        def __iter__(self):
            self.idx_in_epoch = 0
            self.idx_in_batch = 0
            self.iteration = 0
            return self

        def __next__(self):
            if self.idx_in_epoch == 0 and CallableSampleIterator.first_value is not None:
                result = CallableSampleIterator.first_value
                CallableSampleIterator.first_value = None
            else:
                idx = types.SampleInfo(self.idx_in_epoch, self.idx_in_batch, self.iteration)
                result = self.source(idx)
            self.idx_in_epoch += 1
            self.idx_in_batch += 1
            if self.idx_in_batch == batch_size:
                self.idx_in_batch = 0
                self.iteration += 1
            return result

    return CallableSampleIterator, dtype, shape

def get_iterable_from_callback(source_desc):
    """Transform callback that doesn't accept arguments into iterable
    """
    first = source_desc.source()
    dtype, shape = _inspect_data(first)

    class CallableIterator:
        first_value = first
        def __init__(self):
            self.source = source_desc.source

        def __iter__(self):
            return self

        def __next__(self):
            if CallableIterator.first_value is not None:
                result = CallableIterator.first_value
                CallableIterator.first_value = None
            else:
                result = self.source()
            return result

    return CallableIterator, dtype, shape


def get_iterable_from_iterable(source_desc):
    """Wrap iterable into another iterable while peeking the first element
    """
    first_iter = iter(source_desc.source)
    first =  next(first_iter)
    dtype, shape = _inspect_data(first)

    class PeekFirstGenerator:
        first_iterator = first_iter
        first_value = first
        def __init__(self):
            self.source = source_desc.source

        def __iter__(self):
            if PeekFirstGenerator.first_iterator is not None:
                self.it = PeekFirstGenerator.first_iterator
                PeekFirstGenerator.first_iterator = None
            else:
                self.it = iter(self.source)
            return self

        def __next__(self):
            if PeekFirstGenerator.first_value is not None:
                result = PeekFirstGenerator.first_value
                PeekFirstGenerator.first_value = None
                return result
            else:
                return next(self.it)

    return PeekFirstGenerator, dtype, shape

def get_iterable_from_generator(source_desc):
    """Wrap iterable into another iterable while peeking the first element
    """
    # TODO(klecki): difference from the get_iterable_from_iterable is we also need to call the source
    first_iter = iter(source_desc.source())
    first =  next(first_iter)
    dtype, shape = _inspect_data(first)

    class PeekFirstGenerator:
        first_iterator = first_iter
        first_value = first
        def __init__(self):
            self.source = source_desc.source

        def __iter__(self):
            if PeekFirstGenerator.first_iterator is not None:
                self.it = PeekFirstGenerator.first_iterator
                PeekFirstGenerator.first_iterator = None
            else:
                self.it = iter(self.source())
            return self

        def __next__(self):
            if PeekFirstGenerator.first_value is not None:
                result = PeekFirstGenerator.first_value
                PeekFirstGenerator.first_value = None
                return result
            else:
                return next(self.it)

    return PeekFirstGenerator, dtype, shape

def _get_generator_from_source_desc(source_desc, batch_size, is_batched):
    """Based on DALI source description create a generator function, type and shape specification
    compatible with TF Generator Dataset.

    Cycling is delegated to the dataset as some control of some cycling behaviour cannot be
    realized in TF.
    """
    if source_desc.kind == _SourceKind.CALLABLE:
        if source_desc.has_inputs:
            if is_batched:
                return get_batch_iterable_from_callback(source_desc)
            else:
                return get_sample_iterable_from_callback(source_desc, batch_size)
        else:
            # No inputs, plain iteration
            return get_iterable_from_callback(source_desc)
    elif source_desc.kind == _SourceKind.ITERABLE:
        return get_iterable_from_iterable(source_desc)
    else:
        # Generator Func
        return get_iterable_from_generator(source_desc.source)