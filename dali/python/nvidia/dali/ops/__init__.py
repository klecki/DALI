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


import sys

from nvidia.dali import backend as _b
from nvidia.dali.ops._operator_registration import (cpu_ops, gpu_ops, mixed_ops, register_cpu_op,
                                                    register_gpu_op, Reload)

from nvidia.dali.ops import _operator_registration

from nvidia.dali.ops._copmpose_op import Compose
from nvidia.dali.ops._tf_record_op import _load_readers_tfrecord

def _load_ops():
    _operator_registration._discover_ops()
    _all_ops = _operator_registration._all_registered_ops()
    ops_module = sys.modules[__name__]

    for op_reg_name in _all_ops:
        schema = _b.TryGetSchema(op_reg_name)
        make_hidden = schema.IsDocHidden() if schema else False
        _, submodule, op_name = _process_op_name(op_reg_name, make_hidden)
        module = _internal.get_submodule(ops_module, submodule)
        if not hasattr(module, op_name):
            op_class = python_op_factory(op_name, op_reg_name)
            op_class.__module__ = module.__name__
            setattr(module, op_name, op_class)

            if op_name not in ["ExternalSource"]:
                _wrap_op(op_class, submodule)

            # The operator was inserted into nvidia.dali.ops.hidden module, let's import it here
            # so it would be usable, but not documented as coming from other module
            if make_hidden:
                parent_module = _internal.get_submodule(ops_module, submodule[:-1])
                setattr(parent_module, op_name, op_class)


def Reload():
    _load_ops()


_wrap_op(PythonFunction)
_wrap_op(DLTensorPythonFunction)






# This must go at the end - the purpose of these imports is to expose the operators in
# nvidia.dali.ops module
from nvidia.dali.external_source import ExternalSource  # noqa: E402


ExternalSource.__module__ = __name__


register_cpu_op("Compose")
register_gpu_op("Compose")
_load_ops()

try:
    _load_readers_tfrecord()
except RuntimeError:
    # TFRecord can be disabled (custom build). No need to fail
    pass

# Load _arithm_op wrapper, that requires ArithmGenericOp to be already present
from nvidia.dali.ops._arithm_op import _arithm_op
