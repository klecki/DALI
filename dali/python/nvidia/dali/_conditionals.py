# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nvidia.dali import _autograph
from nvidia.dali._autograph.operators import Undefined as _Undefined
from nvidia.dali import data_node
from nvidia.dali import fn

class DaliOperatorOverload(_autograph.OperatorBase):

    def detect_overload_if_stmt(self, cond):
        return isinstance(cond, data_node.DataNode)

    def if_stmt(self, cond, body, orelse, get_state, set_state, symbol_names, nouts):
        # Initial checkpoint before if
        init_state = get_state()

        # split all values, as we may have variables that are both inputs and outputs
        # TODO(klecki): what if something is undefined
        # TODO(klecki): what if something is not DALI-related?
        body_inputs = []
        orelse_inputs = []

        for input_value in init_state:
            if isinstance(input_value, _Undefined):
                body_val, orelse_val = input_value, input_value
            else:
                # TODO we need fn in ag__
                body_val, orelse_val = fn._conditional.split(input_value, predicate=cond)
            body_inputs.append(body_val)
            orelse_inputs.append(orelse_val)

        # Set the state for the body inputs, execute the body and collect the outputs.
        # Get only the true outputs, we don't want to merge what is not used after the if.
        set_state(body_inputs)
        body()
        body_outputs = get_state()[:nouts]

        # Do the same for else block.
        set_state(orelse_inputs)
        orelse()
        orelse_outputs = get_state()[:nouts]

        # Build the state that is the combination of both branches. Only the actual outputs
        # should be affected by the if/else blocks, the rest can be reused from-before split.
        output_values = []
        for new_body_val, new_orelse_val in zip(body_outputs, orelse_outputs):
            output_values.append(fn._conditional.merge(new_body_val, new_orelse_val, predicate=cond))

        # No point in propagating the split/merged values for pure inputs
        output_values += init_state[nouts:]
        set_state(output_values)


_autograph.initialize_autograph(
    DaliOperatorOverload(), filtered_library_modules=["nvidia.dali._autograph", "nvidia.dali.fn", "nvidia.dali"])
