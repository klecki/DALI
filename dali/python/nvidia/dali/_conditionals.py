# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
        self._is_registration_allowed = True

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
        """Get the top scope in the stack"""
        return self._stack[-1]

    def pop(self):
        """Remove the top scope from the stack"""
        result = self._stack.pop()
        return result

    def stack_depth(self):
        """Get the depth of the stack. Note, that by default there is at least one element
        - the global scope."""
        return len(self._stack)

    def _find_closest(self, data_node):
        """Find the closest scope level in the stack where we can access this node as produced
        (or the split of this node closest to us).
        """
        for level in range(self.stack_depth() - 1, -1, -1):
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
        produced_data_node = self._stack[stack_level].get(data_node)
        bottom = self._stack[:stack_level + 1]
        top = self._stack[stack_level + 1:]
        self._stack = bottom
        while top:
            current_entry = top.pop(0)
            predicate = current_entry.predicate

            # Do not automatically register the outputs in the current scope, we track them below
            # in their respective branches.
            logging.log(9, (f"{_indent()}[IF] Inserting split"
                            f" at {_CONDITION_STACK.stack_depth() -1}:"
                            f" split({produced_data_node}, predicate={predicate}."))
            self._is_registration_allowed = False
            true, false = fn._conditional.split(produced_data_node, predicate=predicate)
            self._is_registration_allowed = True

            # Record the result of splitting the `data_node` that we are trying to look up
            # (short-cut for consecutive lookups)
            current_entry.splits[data_node] = (true, false)
            # Record the direct preceding node as the producer:
            current_entry.splits[produced_data_node] = (true, false)
            current_entry.produced_true |= {true}
            current_entry.produced_false |= {false}
            produced_data_node = true if current_entry.branch == _Branch.TrueBranch else false
            self._stack.append(current_entry)
        return produced_data_node

    def preprocess_input(self, data_node):
        """Process the DataNode that is an input to an operator call. Detect if the DataNode was
        produced on the same nesting level. If not, split accordingly to the stack of the previous
        conditions. Caches the previously processed DataNodes to not do repeated splitting.
        """
        stack_level = self._find_closest(data_node)
        logging.log(8, (f"{_indent()}[IF/Input] {data_node} accessed at level"
                        f" {self.stack_depth() - 1} found at {stack_level}."))
        # We already have it cached or produced in this scope.
        if stack_level == self.stack_depth() - 1:
            return self.top().get(data_node)
        # otherwise, we need to fill in the splits.
        return self._realize_split(data_node, stack_level)

    def register_data_nodes(self, data_node, global_scope=False):
        """Register the data nodes as produced in current scope, otherwise if `global_scope` is True
        put them in the outermost scope.
        """
        if not self._is_registration_allowed:
            return
        logging.log(8, (f"{_indent()}[IF/Register] {data_node} at {self.stack_depth() -1}"))
        scope = self._stack[0] if global_scope else self.top()
        if isinstance(data_node, _DataNode):
            scope.produced |= {data_node}
        else:
            scope.produced |= set(data_node)

    def track_true_branch(self):
        """Mark `if` (true) branch as current scope."""
        self.top().branch = _Branch.TrueBranch

    def track_false_branch(self):
        """Mark `else` (false) branch as current scope."""
        self.top().branch = _Branch.FalseBranch

    def no_branch(self):
        """Mark no branch being tracked, the scope "level" stays related to the same if/else
        statement."""
        self.top().branch = _Branch.Undefined

    def track_merge(self, split_predicate):
        """Enter the merge section of the if/else statement. It adds the corresponding
        split_predicate to the nodes visible as produced in the current scope, so all data nodes
        are directly accessible in this scope when looked up by the merge operator.
        We don't care about removing it as it's the last thing happening in that statement.
        """
        self.no_branch()
        self.top().produced |= {split_predicate}


_CONDITION_STACK = _ConditionStack()


def _indent():
    """Helper for indenting the log messages to resemble visited scopes"""
    return '  ' * (_CONDITION_STACK.stack_depth() - 1)


@contextmanager
def _cond_manager(predicate):
    actual_predicate = _CONDITION_STACK.push_predicate(predicate)
    logging.log(7, (f"{_indent()}[IF]: {predicate} at {_CONDITION_STACK.stack_depth() - 1}"))
    # Return it so we can use it in merge
    yield actual_predicate
    _CONDITION_STACK.pop()


@contextmanager
def _cond_true():
    _CONDITION_STACK.track_true_branch()
    logging.log(7, (f"{_indent()}[IF]: `if` branch at {_CONDITION_STACK.stack_depth() - 1}"))
    yield
    _CONDITION_STACK.no_branch()


@contextmanager
def _cond_false():
    _CONDITION_STACK.track_false_branch()
    logging.log(7, (f"{_indent()}[IF]: `else` branch at {_CONDITION_STACK.stack_depth() - 1}"))
    yield
    _CONDITION_STACK.no_branch()


@contextmanager
def _cond_merge(split_predicate):
    _CONDITION_STACK.track_merge(split_predicate)
    yield
    _CONDITION_STACK.no_branch()


def register_data_nodes(data_node, inputs=[]):
    """Register the outputs of the operator as produced in the scope of the current conditional
    branch.

    Parameters
    ----------
    data_node : DataNode or a list/tuple of DataNode
        The output of the operator to be registered.
    inputs : List of DataNode
        Optional list of inputs of the operator whose outputs we are registering.
        If there were no inputs, the outputs are considered as produced in global scope.
    """

    any_input = any(isinstance(input, _DataNode) for input in inputs)
    # TODO(klecki): In theory we have two approaches for inputless operators. Here we insert their
    # outputs to top level and let the automatic splitting handle the situation. Otherwise we could
    # pass the scope information and batch_size within that scope to all operators that are invoked
    # within that scope.
    _CONDITION_STACK.register_data_nodes(data_node, global_scope=not any_input)

def apply_conditional_split(input):
    """Preprocess the DataNode to obtain correctly split batch for the current if scope."""
    return _CONDITION_STACK.preprocess_input(input)

def apply_conditional_split_to_args(inputs, kwargs):
    """Preprocess the inputs and kwargs of the operator to obtain correctly split inputs for the
    current if scope."""
    inputs_bkp = list(inputs)
    for i, input in enumerate(inputs):
        if isinstance(input, _DataNode):
            inputs_bkp[i] = apply_conditional_split(input)
    inputs = tuple(inputs_bkp)
    for key, arg in kwargs.items():
        if isinstance(arg, _DataNode):
            kwargs[key] = apply_conditional_split(arg)
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

    def detect_overload_ld(self, v):
        return isinstance(v, _DataNode)

    def ld(self, v):
        branch_v = apply_conditional_split(v)
        return branch_v

    def detect_overload_if_stmt(self, cond):
        return isinstance(cond, _DataNode)

    def if_stmt(self, cond, body, orelse, get_state, set_state, symbol_names, nouts):
        # Initial checkpoint before if
        init_state = get_state()
        with _cond_manager(cond) as split_predicate:
            # Set the state for the body inputs, execute the body and collect the outputs.
            # Verify if all outputs are initialized within the branch.
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
                    logging.log(9, (f"{_indent()}[IF] Inserting merge"
                                    f" at {_CONDITION_STACK.stack_depth() -1}:"
                                    f" merge({new_body_val}, {new_orelse_val}, predicate="
                                    f"{split_predicate}."))
                    merged = fn._conditional.merge(new_body_val, new_orelse_val,
                                                   predicate=split_predicate)
                    output_values.append(merged)

        # Register the new nodes outside of the conditional scope, they will be used in subsequent
        # calls.
        register_data_nodes(output_values)
        # No point in propagating the split/merged values that won't be read later.
        output_values += init_state[nouts:]
        set_state(output_values)


_OVERLOADS = DaliOperatorOverload()

_autograph.initialize_autograph(_OVERLOADS,
                                do_not_convert_modules=["nvidia.dali._autograph", "nvidia.dali"])
