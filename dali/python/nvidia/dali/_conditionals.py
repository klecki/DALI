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
"""This module contains the implementation of DALI if statement.

It initializes AutoGraph with the DaliOperatorOverload that provides the overload for the if_stmt
and adjust the filtered modules so DALI code is not converted.

The if_stmt provides access to both branches as callables and the set_state/get_state functions
that allows to capture and adjust all symbols modified within those branches. This allows to
checkpoint the state and visit the code of both branches.

if_stmt highlights which state variables are considered the outputs of the if/else pair - we can
use the state captured after visiting if and else branches and produce fn._conditional.merge
nodes for all of them.

When visiting the if/else scopes, we are tracking tha path that we took and the predicates that
were used via the _ConditionStack. As it is not easy to detect which state variables would be
consumed as inputs to DALI operators, we inject additional code to the operator function.
Every time a DataNode is consumed, we look up in which scope it was produced and travel the
path from that point to the current scope in the _ConditionStack, applying necessary splits.
All the return values are registered to the current scope for further lookups.
"""

from nvidia.dali import _autograph
from nvidia.dali.data_node import DataNode as _DataNode
from nvidia.dali import fn

from nvidia.dali._autograph.utils import ag_logging as logging
from nvidia.dali._autograph.operators import variables

from contextlib import contextmanager

from enum import Enum


class _Branch(Enum):
    TrueBranch = 0
    FalseBranch = 1
    Undefined = 2


class _StackEntry:
    """Information about 1 nesting level of if/else statement.

    Keeps the current branch (if we entered if/else branch) and the data nodes that were
    produced in their scopes. Keeps the mapping of DataNodes produced in higher scopes that
    were already split for use in this scope.
    """

    def __init__(self, predicate):
        self.predicate = predicate
        self.branch = _Branch.Undefined
        self.splits = {}
        self.produced_true = set()
        self.produced_false = set()
        # The produced_special handles the case of producing something visible on the same nesting
        # level, but not in one of the branches and is used by merge code.
        self.produced_special = set()

    @property
    def produced(self):
        """Access the set of DataNodes produced in the scope of currently selected branch."""
        if self.branch == _Branch.TrueBranch:
            return self.produced_true
        elif self.branch == _Branch.FalseBranch:
            return self.produced_false
        else:
            return self.produced_special | self.produced_true | self.produced_false

    @produced.setter
    def produced(self, value):
        """Access the set of DataNodes produced in the scope of currently selected branch."""
        if self.branch == _Branch.TrueBranch:
            self.produced_true = value
        elif self.branch == _Branch.FalseBranch:
            self.produced_false = value
        else:
            self.produced_special = value


    def __str__(self):
        return (f"StackEntry: pred={self.predicate}, branch={self.branch}, splits={self.splits},"
                f" produced={self.produced}")

    def has(self, data_node):
        """Check if this DataNode was either produced in this scope or already split for this scope.
        """
        if data_node in self.produced:
            return True
        elif data_node in self.splits:
            return True
        else:
            return False

    def get(self, data_node):
        """Return the `data_node` if it was produced in this scope, or the appropriate split node
        that was created for accessing the `data_node` in this scope.
        """
        assert self.has(data_node)
        if data_node in self.produced:
            return data_node
        else:
            assert self.branch in {_Branch.TrueBranch, _Branch.FalseBranch}
            return self.splits[data_node][self.branch.value]


class _ConditionStack:
    """Tracks the current if/else scope with the path that we took. Captures the used and produced
    data nodes, applying the necessary splits based on the scope level where they were produced
    and where they are used.
    """
    def __init__(self):
        self._stack = [_StackEntry(None)]

    def push_predicate(self, predicate):
        """Add next level of if/else scope that is predicated with the `predicate`.
        The user might have provided a predicate from a scope of higher level, which means
        that `predicate` might be subject to additional slicing. Apply that slicing and return
        the actual predicate that will be used for slicing when entering this scope.

        The situation will happen for example in a case like this, where both predicates are
        produced in global scope:

        pred_0 = ...
        pred_1 = ...

        if pred_0:  # push_pred(pred_0) -> returns pred_0
            if pred_1:  # push_pred(pred_1) ->
                        # -> returns fn._conditional.slice(pred_1, predicate=pred_0)

        Parameters
        ----------
        predicate : DataNode
            Predicate guarding this scope.

        Returns
        -------
        DataNode
            Actual predicate after applying necessary slices to use it in this scope.
        """
        new_pred = _CONDITION_STACK.preprocess_input(predicate)
        new_entry = _StackEntry(new_pred)
        self._stack.append(new_entry)
        return new_pred

    def top(self):
        return self._stack[-1]

    def pop(self):
        result = self._stack.pop()
        return result

    def stack_depth(self):
        return len(self._stack)

    def _find_closest(self, data_node):
        """Find the closest scope level in the stack where we can access this node as produced
        (or the split of this node closest to us).
        """
        for level in range(self.stack_depth()-1, -1, -1):
            if self._stack[level].has(data_node):
                return level
        raise ValueError(f"{data_node} was not produced within this trace.")

    def _realize_split(self, data_node, stack_level):
        """The data_node was produced (or last accessed as via split) in scope earlier than the
        current one, traverse the scopes between that level and current one, and insert split nodes.

        Parameters
        ----------
        data_node : DataNode
            The data node that we want to use in the current scope.
        stack_level : int
            Stack level where the data_node was last "seen".

        Returns
        -------
        DataNode
            New node that can be used in current branch and scope.
        """
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
            # TODO(klecki): Do not register the outputs in the current scope, track them only
            # in the desired branches.
            true, false = fn._conditional.split(produced_data_node, predicate=predicate)

            logging.log(8, f"{'  ' * _CONDITION_STACK.stack_depth()}[Input] Inserted split({produced_data_node}): if {predicate} -> ({true}, {false}) at {level}")
            # Record the result of splitting the `data_node` that we are trying to look up
            # (short-cut for consecutive lookups)
            current_entry.splits[data_node] = (true, false)
            # Record the direct preceding node as the producer:
            current_entry.splits[produced_data_node] = (true, false)
            current_entry.produced_true |= {true}
            current_entry.produced_false |= {false}
            produced_data_node = true if current_entry.branch == _Branch.TrueBranch else false
            self._stack.append(current_entry)
            level += 1
        # print(self._stack)
        return produced_data_node

    def preprocess_input(self, data_node):
        """Process the DataNode that is an input to an operator call. Detect if the DataNode was
        produced on the same nesting level. If not, split accordingly to the stack of the previous
        conditions. Caches the previously processed DataNodes to not do repeated splitting.
        """

        logging.log(8, f"{'  ' * _CONDITION_STACK.stack_depth()}[Input] Looking up {data_node} from {_CONDITION_STACK.stack_depth() - 1}")

        stack_level = self._find_closest(data_node)

        logging.log(8, f"{'  ' * _CONDITION_STACK.stack_depth()}[Input] {data_node} found at {stack_level}")

        # We already have it cached or produced in this scope.
        if stack_level == self.stack_depth() - 1:
            return self.top().get(data_node)

        return self._realize_split(data_node, stack_level)

    def track_true_branch(self):
        self.top().branch = _Branch.TrueBranch

    def track_false_branch(self):
        self.top().branch = _Branch.FalseBranch

    def no_branch(self):
        self.top().branch = _Branch.Undefined

    def track_merge(self, split_predicate):
        self.no_branch()
        self.top().produced |= {split_predicate}

_CONDITION_STACK = _ConditionStack()

@contextmanager
def _cond_manager(predicate):
    logging.log(7, (f"{'  ' * _CONDITION_STACK.stack_depth()}[IF]: {predicate}"
                    f" at {_CONDITION_STACK.stack_depth()}"))
    actual_predicate = _CONDITION_STACK.push_predicate(predicate)

    logging.log(7, (f"{'  ' * _CONDITION_STACK.stack_depth()}[IF/sliced]: {actual_predicate}"
                    f" at {_CONDITION_STACK.stack_depth() - 1}"))
    # Return it so we can use it in merge
    yield actual_predicate
    _CONDITION_STACK.pop()

@contextmanager
def _cond_true():
    _CONDITION_STACK.track_true_branch()
    yield
    _CONDITION_STACK.no_branch()


@contextmanager
def _cond_false():
    _CONDITION_STACK.track_false_branch()
    yield
    _CONDITION_STACK.no_branch()

@contextmanager
def _cond_merge(split_predicate):
    _CONDITION_STACK.no_branch()
    bkp = _CONDITION_STACK.top().produced
    _CONDITION_STACK.top().produced |= {split_predicate}
    yield
    _CONDITION_STACK.top().produced = bkp
    _CONDITION_STACK.no_branch()

def _register_data_nodes(data_node):
    logging.log(7, (f"{'  ' * _CONDITION_STACK.stack_depth()}[Register nodes] {data_node}"
                    f" at {_CONDITION_STACK.stack_depth() -1}"))
    if isinstance(data_node, _DataNode):
        _CONDITION_STACK.top().produced |= {data_node}
    else:
        _CONDITION_STACK.top().produced |= set(data_node)

# def _process_input(data_node):
#     print(f"Looking for {data_node}")
#     # return data_node
#     stack_depth = len(_CONDITION_STACK)
#     found_at = stack_depth - 1
#     print(_CONDITION_STACK[found_at])
#     if _CONDITION_STACK.top().has(data_node):
#         return _CONDITION_STACK.top().get(data_node)
#     while not _CONDITION_STACK[found_at].has(data_node):
#         print(f"found_at: {found_at}, {_CONDITION_STACK[found_at]}")
#         found_at -= 1
#     produced = _CONDITION_STACK[found_at].get(data_node)
#     for i in range(found_at + 1, stack_depth):
#         pred = _CONDITION_STACK[i].predicate
#         _CONDITION_STACK[i].splits[data_node] = fn._conditional.split(produced, predicate=pred)
#         produced = _CONDITION_STACK[i].get(data_node)
#     return produced

def _apply_conditional_split(inputs, kwargs):
    inputs_bkp = list(inputs)
    for i, input in enumerate(inputs):
        if isinstance(input, _DataNode):
            inputs_bkp[i] = _CONDITION_STACK.preprocess_input(input)
    inputs = tuple(inputs_bkp)
    for key, arg in kwargs.items():
        if isinstance(arg, _DataNode):
            kwargs[key] = _CONDITION_STACK.preprocess_input(arg)
    return inputs, kwargs


def _verify_branch_outputs(outputs, symbol_names, branch_name):
    """Verifies variables output by a conditional branch for consistency."""
    common_explanation = (
        "Encountered inconsistent outputs out of the `if/else` control flow statement."
        " Variables need to be initialized in every code path (both `if` branches).")
    for name, output in zip(symbol_names, outputs):
        if isinstance(output, variables.Undefined):
            raise ValueError(f"{common_explanation} Variable '{name}' must also be initialized"
                             f" in the `{branch_name}` branch.")
        if isinstance(output, variables.UndefinedReturnValue):
            raise ValueError(f"{common_explanation} The `{branch_name}` branch must also have"
                             " a return statement.")

class DaliOperatorOverload(_autograph.OperatorBase):

    def detect_overload_if_stmt(self, cond):
        return isinstance(cond, _DataNode)

    def if_stmt(self, cond, body, orelse, get_state, set_state, symbol_names, nouts):
        # Initial checkpoint before if
        init_state = get_state()
        with _cond_manager(cond) as split_predicate:
            # Set the state for the body inputs, execute the body and collect the outputs.
            # Verify if all outputs are initialized within the branch.
            set_state(init_state)
            with _cond_true():
                body()
            body_state = get_state()
            _verify_branch_outputs(body_state, symbol_names, "if")
            body_outputs = body_state[:nouts]

            # Do the same for else block.
            set_state(init_state)
            with _cond_false():
                orelse()
            orelse_state = get_state()
            _verify_branch_outputs(orelse_state, symbol_names, "else")
            orelse_outputs = orelse_state[:nouts]


            # Build the state that is the combination of both branches. Only the actual outputs
            # should be affected by the if/else blocks, the rest can be reused from-before split.
            output_values = []
            # We execute the merge _after_ both branches, and pretend for a moment, that it
            # can see those values produced in child scopes.
            with _cond_merge(split_predicate):
                for new_body_val, new_orelse_val in zip(body_outputs, orelse_outputs):
                    logging.log(8, (f"{'  ' * _CONDITION_STACK.stack_depth()}[Output] Inserting merge"
                                    f" at {_CONDITION_STACK.stack_depth() -1}:"
                                    f"merge({new_body_val}, {new_orelse_val}, predicate={cond}"))
                    output_values.append(fn._conditional.merge(new_body_val, new_orelse_val, predicate=split_predicate))

        # Register the new nodes outside of the conditional scope, they will be used in subsequent
        # calls.
        _register_data_nodes(output_values)
        # No point in propagating the split/merged values that won't be read later.
        output_values += init_state[nouts:]
        set_state(output_values)

_OVERLOADS = DaliOperatorOverload()

_autograph.initialize_autograph(_OVERLOADS,
                                do_not_convert_modules=["nvidia.dali._autograph", "nvidia.dali"])
