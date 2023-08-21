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


def _instantiate_constant_node(device, constant):
    return _Constant(device=device, value=constant.value, dtype=constant.dtype,
                     shape=constant.shape)


def _choose_device(inputs):
    for input in inputs:
        if isinstance(input, (tuple, list)):
            if any(getattr(inp, "device", None) == "gpu" for inp in input):
                return "gpu"
        elif getattr(input, "device", None) == "gpu":
            return "gpu"
    return "cpu"


def _preprocess_inputs(inputs, op_name, device, schema=None):
    if isinstance(inputs, tuple):
        inputs = list(inputs)

    def is_input(x):
        if isinstance(x, (_DataNode, nvidia.dali.types.ScalarConstant)):
            return True
        return (isinstance(x, (list))
                and any(isinstance(y, _DataNode) for y in x)
                and all(isinstance(y, (_DataNode, nvidia.dali.types.ScalarConstant)) for y in x))

    default_input_device = "gpu" if device == "gpu" else "cpu"

    for idx, inp in enumerate(inputs):
        if not is_input(inp):
            if schema:
                input_device = schema.GetInputDevice(idx) or default_input_device
            else:
                input_device = default_input_device
            if not isinstance(inp, nvidia.dali.types.ScalarConstant):
                try:
                    inp = _Constant(inp, device=input_device)
                except Exception as ex:
                    raise TypeError(f"""when calling operator {op_name}:
Input {idx} is neither a DALI `DataNode` nor a list of data nodes but `{type(inp).__name__}`.
Attempt to convert it to a constant node failed.""") from ex

            if not isinstance(inp, _DataNode):
                inp = nvidia.dali.ops._instantiate_constant_node(input_device, inp)

        inputs[idx] = inp
    return inputs
