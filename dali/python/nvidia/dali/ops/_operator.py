# Copyright (c) 2017-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# pylint: disable=no-member
import sys
import threading
import warnings
from itertools import count

import nvidia.dali.python_function_plugin
from nvidia.dali import backend as _b
from nvidia.dali.backend import OpSchema as _OpSchema, OpSpec as _OpSpec
from nvidia.dali import fn as _functional
from nvidia.dali import internal as _internal
from nvidia.dali.data_node import DataNode as _DataNode
from nvidia.dali.pipeline import Pipeline as _Pipeline
from nvidia.dali.types import (_type_name_convert_to_string, _type_convert_value,
                               _default_converter, _vector_element_type, DALIDataType, CUDAStream as
                               _CUDAStream, Constant as _Constant, ScalarConstant as
                               _ScalarConstant)
from nvidia.dali import _conditionals

from nvidia.dali.ops._operator_docs import (_docstring_generator, _docstring_generator_call,
                                            _docstring_generator_fn)
from nvidia.dali.ops._operator_utils import (_instantiate_constant_node, _preprocess_inputs, _build_input_sets)


cupy = None


def _setup_cupy():
    global cupy
    if cupy is None:
        import cupy as cupy


def _schema_name(cls):
    return getattr(cls, 'schema_name', cls.__name__)


class _OpCounter(object):
    """Atomic counter used for auto-naming new operator instances throughout the lifetime of the
    process.
    """
    # pylint: disable=too-few-public-methods
    _lock = threading.Lock()
    _op_count = count(0)

    def __init__(self):
        with self._lock:
            self._id = next(self._op_count)

    @property
    def id(self):
        return self._id




def _separate_kwargs(kwargs, arg_input_type=_DataNode):
    """Separates arguments into ones that should go to operator's __init__ and to __call__,
    the former are the scalar arguments passed via AddArg to the OpSpec,
    the latter are the argument inputs represented by DataNodes/TensorLists.

    Returns a pair of dictionaries of kwargs - the first for __init__, the second for __call__.

    Args:
        kwargs: Keyword arguments.
        arg_input_type: operator's argument input type, DataNode for pipeline mode, TensorListCPU
            for eager mode.
    """

    def is_arg_input_type(x):
        return isinstance(x, arg_input_type)

    def is_call_arg(name, value):
        if name == "device":
            return False
        if name == "ndim":
            return False
        if name == "name" or is_arg_input_type(value):
            return True
        if isinstance(value, (str, list, tuple, nvidia.dali.types.ScalarConstant)):
            return False
        return not nvidia.dali.types._is_scalar_value(value)

    def to_scalar(scalar):
        return scalar.value if isinstance(scalar, nvidia.dali.types.ScalarConstant) else scalar

    init_args = {}
    call_args = {}
    for name, value in kwargs.items():
        if value is None:
            continue
        if is_call_arg(name, value):
            call_args[name] = value
        else:
            init_args[name] = to_scalar(value)

    return init_args, call_args


def _check_arg_input(schema, op_name, name):
    if name == "name":
        return
    if not schema.IsTensorArgument(name):
        expected_type_name = _type_name_convert_to_string(schema.GetArgumentType(name), False)
        raise TypeError(
            f"The argument `{name}` for operator `{op_name}` should not be a `DataNode` but a "
            f"{expected_type_name}")


def _add_spec_args(schema, spec, kwargs):
    for key, value in kwargs.items():
        if value is None:
            # None is not a valid value for any argument type, so treat it
            # as if the argument was not supplied at all
            continue

        dtype = schema.GetArgumentType(key)
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                spec.AddArgEmptyList(key, _vector_element_type(dtype))
                continue
        converted_value = _type_convert_value(dtype, value)
        spec.AddArg(key, converted_value)

def _handle_argument_deprecation(schema, op_name, kwargs):
    # For whatever reason, we do it only for scalar arguments
    # TODO(klecki): Extend this to handle other kinds of arguments as well
    # Check for any deprecated arguments that should be replaced or removed
    arg_names = list(kwargs.keys())
    for arg_name in arg_names:
        if not schema.IsDeprecatedArg(arg_name):
            continue
        meta = schema.DeprecatedArgMeta(arg_name)
        new_name = meta['renamed_to']
        removed = meta['removed']
        msg = meta['msg']
        if new_name:
            if new_name in kwargs:
                raise TypeError(f"Operator {op_name} got an unexpected"
                                f" '{arg_name}' deprecated argument when '{new_name}'"
                                f" was already provided")
            kwargs[new_name] = kwargs[arg_name]
            del kwargs[arg_name]
        elif removed:
            del kwargs[arg_name]

        with warnings.catch_warnings():
            warnings.simplefilter("default")
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
    return kwargs

# TODO(klecki): Why do we have:
# * _OperatorInstance
# python_op_factory with class Operator
# metaclass _DaliOperatorMeta


class _OperatorInstance(object):
    """Operator Instance is created when the __call__ function of respective DALI API is invoked.
    It is responsible for preparing the output DataNodes of that __call__ function.

    The `__init__` is part of the API's operator `__call__` that generates the name of the operator
    instance for the backend and fills the OpSpec with Argument Inputs.

    Parameters
    ----------
    object : _type_
        _description_
    """

    def __init__(self, inputs, op, **kwargs):
        self._counter = _OpCounter()
        self._outputs = []
        self._op = op
        self._default_call_args = op._call_args
        self._spec = op.spec.copy()
        self._relation_id = self._counter.id

        # TODO(klecki): We already did _preprocess_inputs, so we should have already replaced ScalarConstants.
        if inputs is not None:
            default_input_device = "gpu" if op.device == "gpu" else "cpu"
            inputs = list(inputs)
            for i in range(len(inputs)):
                inp = inputs[i]
                if isinstance(inp, _ScalarConstant):
                    inputs[i] = _instantiate_constant_node(default_input_device, inp)
            inputs = tuple(inputs)

        # TODO(klecki): We didn't include the ones from __init__ in class API, tough luck,
        # go to the fn API.
        if _conditionals.conditionals_enabled():
            inputs, kwargs = _conditionals.apply_conditional_split_to_args(inputs, kwargs)

        self._inputs = inputs

        spec_args, kwargs = _separate_kwargs(kwargs)
        # TODO(klecki): HANDLE DEPRECATION FOR ARGUMENTS AGAIN :V
        spec_args = _handle_argument_deprecation(op._schema, type(op).__name__, spec_args)
        _add_spec_args(op._schema, self._spec, spec_args)

        # TODO(klecki): Extract it as a specific ops API call job to merge both stages
        call_args = {**self._default_call_args}
        for k, v in kwargs.items():
            if v is None:
                # if an argument was specified in __init__ and in __call__ it is None, ignore it
                continue
            if k in self._default_call_args:
                raise ValueError("The argument `{}` was already specified in __init__.".format(k))
            call_args[k] = v

        # TODO(klecki): Handle the name of this instance once
        name = call_args.get("name", None)
        if name is not None:
            self._name = name
        else:
            self._name = '__' + type(op).__name__ + "_" + str(self._counter.id)
        # TODO(klecki): MIS will work with the same name if it was provided in init xDDD

        # Add inputs
        if inputs:
            for inp in inputs:
                if not isinstance(inp, _DataNode):
                    raise TypeError(
                        f"Expected inputs of type `DataNode`. Received input of type '{inp}'.")
                self._spec.AddInput(inp.name, inp.device)
        # Argument inputs
        for k in sorted(call_args.keys()):
            if k not in ["name"]:
                arg_inp = call_args[k]
                # TODO(klecki): Extract none filtering once!
                if arg_inp is None:
                    continue
                # TODO(klecki): _preprocess_inputs here as well?
                if isinstance(arg_inp, _ScalarConstant):
                    arg_inp = _instantiate_constant_node("cpu", arg_inp)
                if not isinstance(arg_inp, _DataNode):
                    try:
                        arg_inp = _Constant(arg_inp, device="cpu")
                    except Exception as e:
                        raise TypeError(
                            f"Expected inputs of type "
                            f"`DataNode` or convertible to constant nodes. Received "
                            f"input `{k}` of type '{type(arg_inp).__name__}'.") from e

                _check_arg_input(op._schema, type(self._op).__name__, k)

                self._spec.AddArgumentInput(k, arg_inp.name)
                self._inputs = list(self._inputs) + [arg_inp]

        if self._op.schema.IsDeprecated():
            # TODO(klecki): how to know if this is fn or ops?
            msg = "WARNING: `{}` is now deprecated".format(_op_name(type(self._op).__name__, "fn"))
            use_instead = _op_name(self._op.schema.DeprecatedInFavorOf(), "fn")
            if use_instead:
                msg += ". Use `" + use_instead + "` instead."
            explanation = self._op.schema.DeprecationMessage()
            if explanation:
                msg += "\n" + explanation
            with warnings.catch_warnings():
                warnings.simplefilter("default")
                warnings.warn(msg, DeprecationWarning, stacklevel=2)

    def check_args(self):
        self._op.schema.CheckArgs(self._spec)

    def generate_outputs(self):
        pipeline = _Pipeline.current()
        if pipeline is None and self._op.preserve:
            _Pipeline._raise_pipeline_required("Operators with side-effects ")
        # TODO(klecki): THIS IS THE MOST XD PART OF THE LIBRARY
        # Add outputs
        if self._op.device == "gpu" or self._op.device == "mixed":
            output_device = "gpu"
        else:
            output_device = "cpu"

        num_output = (self._op.schema.CalculateOutputs(self._spec)
                      + self._op.schema.CalculateAdditionalOutputs(self._spec))

        if num_output == 0 and self._op.preserve:
            t_name = type(self._op).__name__ + "_id_" + str(self.id) + "_sink"
            pipeline.add_sink(_DataNode(t_name, output_device, self))
            return

        for i in range(num_output):
            t_name = self._name
            if num_output > 1:
                t_name += "[{}]".format(i)
            t = _DataNode(t_name, output_device, self)
            self._spec.AddOutput(t.name, t.device)
            if self._op.preserve:
                pipeline.add_sink(t)
            self.append_output(t)

    @property
    def id(self):
        return self._counter.id

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    @property
    def unwrapped_outputs(self):
        if len(self._outputs) == 1:
            return self._outputs[0]
        else:
            return self._outputs

    @property
    def spec(self):
        return self._spec

    @property
    def name(self):
        return self._name

    @property
    def relation_id(self):
        return self._relation_id

    @relation_id.setter
    def relation_id(self, value):
        self._relation_id = value

    def append_output(self, output):
        self._outputs.append(output)


class _DaliOperatorMeta(type):

    @property
    def __doc__(self):
        return _docstring_generator(self)




def python_op_factory(name, schema_name=None):

    class Operator(metaclass=_DaliOperatorMeta):

        def __init__(self, *, device="cpu", **kwargs):
            schema_name = _schema_name(type(self))
            self._spec = _b.OpSpec(schema_name)
            self._schema = _b.GetSchema(schema_name)

            # Get the device argument. We will need this to determine
            # the device that our outputs will be stored on
            self._device = device
            self._spec.AddArg("device", self._device)

            kwargs, self._call_args = _separate_kwargs(kwargs)

            for k in self._call_args.keys():
                _check_arg_input(self._schema, type(self).__name__, k)

            if "preserve" in kwargs.keys():
                self._preserve = kwargs["preserve"]
                # we don't want to set "preserve" arg twice
                del kwargs["preserve"]
            else:
                self._preserve = False
            self._spec.AddArg("preserve", self._preserve)
            self._preserve = self._preserve or self._schema.IsNoPrune()

            kwargs = _handle_argument_deprecation(self._schema, type(self).__name__, kwargs)

            # Store the specified arguments
            _add_spec_args(self._schema, self._spec, kwargs)

        @property
        def spec(self):
            return self._spec

        @property
        def schema(self):
            return self._schema

        @property
        def device(self):
            return self._device

        @property
        def preserve(self):
            return self._preserve

        def __call__(self, *inputs, **kwargs):
            self._check_schema_num_inputs(inputs)

            inputs = _preprocess_inputs(inputs, self.__class__.__name__, self._device, self._schema)

            input_sets = _build_input_sets(inputs, self.__class__.__name__)

            # Create OperatorInstance for every input set
            op_instances = []
            for input_set in input_sets:
                op_instances.append(_OperatorInstance(input_set, self, **kwargs))
                op_instances[-1].generate_outputs()

            # Tie the instances together
            relation_id = op_instances[0].id
            for op in op_instances:
                op.relation_id = relation_id

            # If we don't have multiple input sets, flatten the result
            if len(op_instances) == 1:
                result = op_instances[0].unwrapped_outputs
            else:
                outputs = []
                for op in op_instances:
                    outputs.append(op.outputs)
                result = self._repack_output_sets(outputs)
            if _conditionals.conditionals_enabled():
                if len(op_instances) != 1:
                    raise ValueError("Multiple input sets are not supported with conditional"
                                     " execution (when `enable_conditionals=True`)")
                _conditionals.register_data_nodes(result, input_sets[0], kwargs)
            return result

        def _check_schema_num_inputs(self, inputs):
            if len(inputs) < self._schema.MinNumInput() or len(inputs) > self._schema.MaxNumInput():
                raise ValueError(
                    f"Operator {type(self).__name__} expects "
                    f"from {self._schema.MinNumInput()} to {self._schema.MaxNumInput()} inputs, "
                    f"but received {len(inputs)}.")

    Operator.__name__ = str(name)
    Operator.schema_name = schema_name or Operator.__name__
    Operator.__call__.__doc__ = _docstring_generator_call(Operator.schema_name)
    return Operator


def _process_op_name(op_schema_name, make_hidden=False):
    # Two underscores (reasoning: we might want to have single underscores in the namespace itself)
    namespace_delim = "__"
    op_full_name = op_schema_name.replace(namespace_delim, '.')
    *submodule, op_name = op_full_name.split('.')
    if make_hidden:
        submodule = [*submodule, 'hidden']
    return op_full_name, submodule, op_name


def _op_name(op_schema_name, api="fn"):
    full_name, submodule, op_name = _process_op_name(op_schema_name)
    if api == "fn":
        return ".".join([*submodule, _functional._to_snake_case(op_name)])
    elif api == "ops":
        return full_name
    else:
        raise ValueError(f'{api} is not a valid DALI api name, try one of {"fn", "ops"}')


def _wrap_op(op_class, submodule=[], parent_module=None):
    return _functional._wrap_op(op_class, submodule, parent_module,
                                _docstring_generator_fn(op_class))
