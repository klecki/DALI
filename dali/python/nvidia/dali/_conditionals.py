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

from nvidia.dali._autograph.utils import ag_logging as logging

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
        self.splits = {}
        self.produced_true = set()
        self.produced_false = set()
        self.produced_special = set()

    @property
    def produced(self):
        if self.branch == _Branch.TrueBrach:
            return self.produced_true
        elif self.branch == _Branch.FalseBranch:
            return self.produced_false
        else:
            return self.produced_special | self.produced_true | self.produced_false

    @produced.setter
    def produced(self, value):
        if self.branch == _Branch.TrueBrach:
            self.produced_true = value
        elif self.branch == _Branch.FalseBranch:
            self.produced_false = value
        else:
            self.produced_special = value


    def __str__(self):
        return f"StackEntry: pred={self.predicate}, branch={self.branch}, splits={self.splits}, produced={self.produced}"

    def has(self, dn):
        if dn in self.produced:
            return True
        elif dn in self.splits:
            return True
        else:
            return False

    def get(self, dn):
        assert self.has(dn)
        if dn in self.produced:
            return dn
        else:
            return self.splits[dn][self.branch.value]


class _ConditionStack:
    def __init__(self):
        self._stack = [_StackEntry(None)]

    def push_predicate(self, predicate):
        new_entry = _StackEntry(predicate)
        self._stack.append(new_entry)

    def top(self):
        return self._stack[-1]

    def pop(self):
        result = self._stack.pop()
        return result

    def stack_depth(self):
        return len(self._stack)

    def _find_closest(self, data_node):
        for level in range(self.stack_depth()-1, -1, -1):
            if self._stack[level].has(data_node):
                return level
        raise ValueError(f"{data_node} was not produced within this trace.")

    def _realize_split(self, data_node, stack_level):
        assert 0 <= stack_level and stack_level < self.stack_depth() - 1
        logging.log(8, f"{'  ' * _CONDITION_STACK.stack_depth()}[Input] {data_node} requires splitting from {stack_level}")
        produced_data_node = self._stack[stack_level].get(data_node)
        bottom = self._stack[: stack_level+1]
        top = self._stack[stack_level+1 :]
        # print(f"{bottom}:{top}")
        # We will be inserting new split nodes at this level. They are above the branches.
        self._stack = bottom
        level = stack_level+1
        while top:
            current_entry = top.pop(0)
            predicate = current_entry.predicate

            logging.log(8, f"{'  ' * _CONDITION_STACK.stack_depth()}[Input] Inserting split for {data_node} at {level}: split({produced_data_node}, predicate={predicate}) ...")
            # TODO(klecki): Make split not register the produced DataNodes automatically
            true, false = fn._conditional.split(produced_data_node, predicate=predicate)
            # logging.log(8, f"New split inserted at [{level}], {produced_data_node} -> if {predicate} -> ({true}, {false})")

            logging.log(8, f"{'  ' * _CONDITION_STACK.stack_depth()}[Input] Inserted split({produced_data_node}): if {predicate} -> ({true}, {false}) at {level}")
            # Record the result of splitting the `data_node` that we are trying to look up (short-cut)
            current_entry.splits[data_node] = (true, false)
            # Record the direct preceding node as the producer:
            current_entry.splits[produced_data_node] = (true, false)
            current_entry.produced_true |= {true}
            current_entry.produced_false |= {false}
            # current_entry.produced |= {true, false}
            produced_data_node = true if current_entry.branch == _Branch.TrueBrach else false
            self._stack.append(current_entry)
            level += 1
        # print(self._stack)
        return produced_data_node

    def preprocess_input(self, data_node):
        """Process the DataNode that is an input to an operator call. Detect if the DataNode was
        produced on the same nesting level. If not, split accordingly to the stack of the previous
        conditions. Caches the previously processed DataNodes to not do repeated splitting.

        Parameters
        ----------
        data_node : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """

        logging.log(8, f"{'  ' * _CONDITION_STACK.stack_depth()}[Input] Looking up {data_node} from {_CONDITION_STACK.stack_depth() - 1}")

        stack_level = self._find_closest(data_node)

        logging.log(8, f"{'  ' * _CONDITION_STACK.stack_depth()}[Input] {data_node} found at {stack_level}")

        # We already have it cached or produced in this scope.
        if stack_level == self.stack_depth() - 1:
            return self.top().get(data_node)

        return self._realize_split(data_node, stack_level)

_CONDITION_STACK = _ConditionStack()


@contextmanager
def _cond_manager(predicate):
    logging.log(7, (f"{'  ' * _CONDITION_STACK.stack_depth()}[IF]: {predicate}"
                    f" at {_CONDITION_STACK.stack_depth()}"))
    # Split it for current level if needed, this would happen automatically either way
    new_pred = _CONDITION_STACK.preprocess_input(predicate)
    # Use that condition directly on this stack level
    _CONDITION_STACK.push_predicate(new_pred)

    logging.log(7, (f"{'  ' * _CONDITION_STACK.stack_depth()}[IF/sliced]: {new_pred}"
                    f" at {_CONDITION_STACK.stack_depth() - 1}"))
    # Return it so we can use it in merge
    yield new_pred
    # finally:
    # assert _CONDITION_STACK.top().branch == _Branch.FalseBranch
    _CONDITION_STACK.pop()

@contextmanager
def _cond_true():
    logging.log(7, (f"{'  ' * _CONDITION_STACK.stack_depth()}[TRUE BRANCH]"
                    f" at {_CONDITION_STACK.stack_depth() - 1}"))
    assert _CONDITION_STACK.top().branch == _Branch.Undefined
    _CONDITION_STACK.top().branch = _Branch.TrueBrach
    yield
    # clear what we produced, we do not cross contaminate branches
    # _CONDITION_STACK.top().produced_bkp = _CONDITION_STACK.top().produced
    # _CONDITION_STACK.top().produced = set()


@contextmanager
def _cond_false():
    logging.log(7, (f"{'  ' * _CONDITION_STACK.stack_depth()}[FALSE BRANCH]"
                    f" at {_CONDITION_STACK.stack_depth() - 1}"))
    assert _CONDITION_STACK.top().branch == _Branch.TrueBrach
    _CONDITION_STACK.top().branch = _Branch.FalseBranch
    yield
    # For the validation of merge
    # _CONDITION_STACK.top().produced |= _CONDITION_STACK.top().produced_bkp

@contextmanager
def _cond_merge():
    logging.log(7, (f"{'  ' * _CONDITION_STACK.stack_depth()}[MERGE OUTPUTS]"
                    f" at {_CONDITION_STACK.stack_depth() - 1}"))
    assert _CONDITION_STACK.top().branch == _Branch.FalseBranch
    _CONDITION_STACK.top().branch = _Branch.Undefined
    # We merge everything that was produced using the predicate, so we treat it as "known"
    # The correct merge predicate needs to be taken from one level above.
    # TODO(klecki): encapsulate this in the CONDITION_STACK code.
    tmp = _CONDITION_STACK.pop()
    tmp.produced |= {_CONDITION_STACK.preprocess_input(tmp.predicate)}
    _CONDITION_STACK._stack.append(tmp)
    yield

def _current_branch():
    return _CONDITION_STACK.top().branch

def _register_data_nodes(dn):
    logging.log(7, (f"{'  ' * _CONDITION_STACK.stack_depth()}[Register nodes] {dn}"
                    f" at {_CONDITION_STACK.stack_depth() -1}"))
    if isinstance(dn, data_node.DataNode):
        _CONDITION_STACK.top().produced |= {dn}
    else:
        _CONDITION_STACK.top().produced |= set(dn)

# def _process_input(dn):
#     print(f"Looking for {dn}")
#     # return dn
#     stack_depth = len(_CONDITION_STACK)
#     found_at = stack_depth - 1
#     print(_CONDITION_STACK[found_at])
#     if _CONDITION_STACK.top().has(dn):
#         return _CONDITION_STACK.top().get(dn)
#     while not _CONDITION_STACK[found_at].has(dn):
#         print(f"found_at: {found_at}, {_CONDITION_STACK[found_at]}")
#         found_at -= 1
#     produced = _CONDITION_STACK[found_at].get(dn)
#     for i in range(found_at + 1, stack_depth):
#         pred = _CONDITION_STACK[i].predicate
#         _CONDITION_STACK[i].splits[dn] = fn._conditional.split(produced, predicate=pred)
#         produced = _CONDITION_STACK[i].get(dn)
#     return produced

def _apply_conditional_split(inputs, kwargs):
    inputs_bkp = list(inputs)
    for i, input in enumerate(inputs):
        if isinstance(input, data_node.DataNode):
            inputs_bkp[i] = _CONDITION_STACK.preprocess_input(input)
    inputs = tuple(inputs_bkp)
    for key, arg in kwargs.items():
        if isinstance(arg, data_node.DataNode):
            kwargs[key] = _CONDITION_STACK.preprocess_input(arg)
    return inputs, kwargs

class DaliOperatorOverload(_autograph.OperatorBase):

    def detect_overload_if_stmt(self, cond):
        return isinstance(cond, data_node.DataNode)

    def if_stmt(self, cond, body, orelse, get_state, set_state, symbol_names, nouts):
        # Initial checkpoint before if
        init_state = get_state()
        with _cond_manager(cond) as merge_cond:
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
            # Merge is tricky. The new values are produced in some "child" scopes (if/else block),
            # but the predicate is one level above.
            # We execute the merge _after_ both branches, and pretend for a moment, that it
            # can see those values produced in child scopes.
            with _cond_merge():
                for new_body_val, new_orelse_val in zip(body_outputs, orelse_outputs):
                    logging.log(8, (f"{'  ' * _CONDITION_STACK.stack_depth()}[Output] Inserting merge"
                                    f" at {_CONDITION_STACK.stack_depth() -1}:"
                                    f"merge({new_body_val}, {new_orelse_val}, predicate={cond}"))
                    output_values.append(fn._conditional.merge(new_body_val, new_orelse_val, predicate=merge_cond))

        # Register as produced the new nodes, they will be used in subsequent calls.
        _register_data_nodes(output_values)
        # No point in propagating the split/merged values for pure inputs
        output_values += init_state[nouts:]
        set_state(output_values)


_OVERLOADS = DaliOperatorOverload()

_autograph.initialize_autograph(
    _OVERLOADS, do_not_convert_modules=["nvidia.dali._autograph", "nvidia.dali"])
