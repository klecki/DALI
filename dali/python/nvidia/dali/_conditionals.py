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

from contextlib import contextmanager

from collections import namedtuple
from enum import Enum


class _Branch(Enum):
    TrueBrach = 0
    FalseBranch = 1
    Undefined = 2


class _StackEntry:
    def __init__(self, predicate):
        self.predicate = predicate
        self.branch = _Branch.Undefined
        self.nodes = {}
        self.produced = set()

    def __str__(self):
        return f"StackEntry: pred={self.predicate}, branch={self.branch}, nodes={self.nodes}, produced={self.produced}"

    def has(self, dn):
        if dn in self.produced:
            return True
        elif dn in self.nodes:
            return True
        else:
            return False

    def get(self, dn):
        assert self.has(dn)
        if dn in self.produced:
            return dn
        else:
            return self.nodes[dn][self.branch.value]



# TODO - this needs to be stack + path True/False
_GLOBAL_CONDITION=[_StackEntry(None)]

def _condition_stack_top():
    return _GLOBAL_CONDITION[-1]

@contextmanager
def _cond_manager(predicate):
    new_entry = _StackEntry(predicate=predicate)
    print("> > Starting if > ")
    _GLOBAL_CONDITION.append(new_entry)
    try:
        yield
    finally:
        assert _condition_stack_top().branch == _Branch.FalseBranch
        _GLOBAL_CONDITION.pop()

@contextmanager
def _cond_true():
    assert _condition_stack_top().branch == _Branch.Undefined
    _condition_stack_top().branch = _Branch.TrueBrach
    print("> > Starting True > ")
    yield
    # clear what we produced, we do not cross contaminate branches
    _condition_stack_top().produced = set()


@contextmanager
def _cond_false():
    assert _condition_stack_top().branch == _Branch.TrueBrach
    _condition_stack_top().branch = _Branch.FalseBranch
    print("> > Starting False > ")
    yield

def _current_branch():
    return _condition_stack_top().branch

def _register_data_nodes(dn):
    if isinstance(dn, data_node.DataNode):
        _condition_stack_top().produced |= {dn}
    else:
        _condition_stack_top().produced |= set(dn)

def _process_input(dn):
    print(f"Looking for {dn}")
    # return dn
    stack_depth = len(_GLOBAL_CONDITION)
    found_at = stack_depth - 1
    print(_GLOBAL_CONDITION[found_at])
    if _condition_stack_top().has(dn):
        return _condition_stack_top().get(dn)
    while not _GLOBAL_CONDITION[found_at].has(dn):
        print(f"found_at: {found_at}, {_GLOBAL_CONDITION[found_at]}")
        found_at -= 1
    produced = _GLOBAL_CONDITION[found_at].get(dn)
    for i in range(found_at + 1, stack_depth):
        pred = _GLOBAL_CONDITION[i].predicate
        _GLOBAL_CONDITION[i].nodes[dn] = fn._conditional.split(produced, predicate=pred)
        produced = _GLOBAL_CONDITION[i].get(dn)
    return produced


class DaliOperatorOverload(_autograph.OperatorBase):

    def detect_overload_if_stmt(self, cond):
        return isinstance(cond, data_node.DataNode)

    def if_stmt(self, cond, body, orelse, get_state, set_state, symbol_names, nouts):
        # Initial checkpoint before if
        init_state = get_state()
        with _cond_manager(cond):
            # split all values, as we may have variables that are both inputs and outputs
            # TODO(klecki): what if something is undefined
            # TODO(klecki): what if something is not DALI-related?
            body_inputs = []
            orelse_inputs = []

            # for input_value in init_state:
            #     if isinstance(input_value, _Undefined):
            #         body_val, orelse_val = input_value, input_value
            #     else:
            #         # TODO we need fn in ag__
            #         body_val, orelse_val = fn._conditional.split(input_value, predicate=cond)
            #     body_inputs.append(body_val)
            #     orelse_inputs.append(orelse_val)

            # Set the state for the body inputs, execute the body and collect the outputs.
            # Get only the true outputs, we don't want to merge what is not used after the if.
            set_state(init_state)
            with _cond_true():
                body()
            body_outputs = get_state()[:nouts]

            # Do the same for else block.
            set_state(init_state)
            with _cond_false():
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


_OVERLOADS = DaliOperatorOverload()

_autograph.initialize_autograph(
    _OVERLOADS, do_not_convert_modules=["nvidia.dali._autograph", "nvidia.dali"])
