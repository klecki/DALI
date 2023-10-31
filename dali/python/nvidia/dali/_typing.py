# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Union, Optional, overload, TypeAlias, Protocol
from typing import Any, Sequence, List, Callable, Iterable

from nvidia.dali.tensors import TensorCPU, TensorGPU, TensorListCPU, TensorListGPU

TensorLike : TypeAlias = Union[Any, TensorCPU, TensorGPU]
"""Type alias representing n-dim tensor that can be consumed by DALI.

Must be in one of the compatible array types:

        * NumPy ndarray (CPU)
        * MXNet ndarray (CPU)
        * PyTorch tensor (CPU or GPU)
        * CuPy array (GPU)
        * objects implementing ``__cuda_array_interface__``
        * DALI `Tensor` object
"""

BatchLike : TypeAlias = Union[TensorLike, List[TensorLike], TensorListCPU, TensorListGPU]
"""Type alias representing a batch of n-dim tensors that can be consumed by DALI.

Batch of n-dim tensors can be represented by:
    * List of compatible TensorLike objects
    * (n+1)-dim TensorLike object
    * DALI `TensorList` objects
"""