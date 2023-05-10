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


import ast
import sys


from inspect import Signature, Parameter

from nvidia.dali import backend as _b
from nvidia.dali import internal as _internal
import nvidia.dali.ops as _ops

_special_case_mapping = {
    "b_box": "bbox",
    "mx_net": "mxnet",
    "tf_record": "tfrecord"
}


def _handle_special_case(s):
    for artifact, desired in _special_case_mapping.items():
        s = s.replace(artifact, desired)
    return s


def _to_snake_case(pascal):
    out = ""
    nupper = 0
    start = 0
    for i, c in enumerate(pascal):
        if c.isupper():
            if nupper == 0:
                start = i
            nupper += 1
        elif c.islower():
            if nupper == 0:
                out += c
            else:
                # do not add another leading underscore
                if len(out) > 0 and out[-1] != '_':
                    out += '_'
                if nupper > 1:
                    out += pascal[start:i - 1].lower() + '_'
                out += pascal[i - 1].lower()
                out += c
                nupper = 0
            start = i + 1
        else:
            out += pascal[start:i + 1].lower()
            start = i + 1
            nupper = 0

    if nupper > 0:
        if len(out) and out[-1].islower():
            out += '_'
        out += pascal[start:].lower()
    out = _handle_special_case(out)
    return out


def _wrap_op_fn(op_class, wrapper_name, wrapper_doc):
    def op_wrapper(*inputs, **kwargs):
        import nvidia.dali.ops
        init_args, call_args = nvidia.dali.ops._separate_kwargs(kwargs)

        default_dev = nvidia.dali.ops._choose_device(inputs)
        if default_dev == "gpu" and init_args.get("device") == "cpu":
            raise ValueError("An operator with device='cpu' cannot accept GPU inputs.")

        if "device" not in init_args:
            init_args["device"] = default_dev

        return op_class(**init_args)(*inputs, **call_args)

    def fn_wrapper(*inputs, **kwargs):
        from nvidia.dali._debug_mode import _PipelineDebug
        current_pipeline = _PipelineDebug.current()
        if getattr(current_pipeline, '_debug_on', False):
            return current_pipeline._wrap_op_call(op_class, wrapper_name, *inputs, **kwargs)
        else:
            return op_wrapper(*inputs, **kwargs)

    fn_wrapper.__name__ = wrapper_name
    fn_wrapper.__qualname__ = wrapper_name
    fn_wrapper.__doc__ = wrapper_doc
    fn_wrapper._schema_name = op_class.schema_name
    return fn_wrapper


def _wrap_op(op_class, submodule, parent_module, wrapper_doc):
    """Wrap the DALI Operator with fn API and insert the function into appropriate module.

    Args:
        op_class: Op class to wrap
        submodule: Additional submodule (scope)
        parent_module (str): If set to None, the wrapper is placed in nvidia.dali.fn module,
            otherwise in a specified parent module.
        wrapper_doc (str): Documentation of the wrapper function
    """
    from nvidia.dali._utils import eager_utils

    schema = _b.TryGetSchema(op_class.schema_name)
    make_hidden = schema.IsDocHidden() if schema else False
    wrapper_name = _to_snake_case(op_class.__name__)

    # Add operator to eager API.
    eager_utils._wrap_eager_op(op_class, submodule, parent_module,
                               wrapper_name, wrapper_doc, make_hidden)

    if parent_module is None:
        fn_module = sys.modules[__name__]
    else:
        fn_module = sys.modules[parent_module]
    module = _internal.get_submodule(fn_module, submodule)
    if not hasattr(module, wrapper_name):
        wrap_func = _wrap_op_fn(op_class, wrapper_name, wrapper_doc)
        setattr(module, wrapper_name, wrap_func)
        if submodule:
            wrap_func.__module__ = module.__name__
        if make_hidden:
            parent_module = _internal.get_submodule(fn_module, submodule[:-1])
            setattr(parent_module, wrapper_name, wrap_func)


# based on https://peps.python.org/pep-0362/
def _generate_prototype(registered_op_name):
    schema = _b.TryGetSchema(registered_op_name)
    make_hidden = schema.IsDocHidden() if schema else False
    _, submodule, op_name = _ops._process_op_name(registered_op_name, make_hidden)
    wrapper_name = _to_snake_case(op_name)
    parameters = []
    if schema.MaxNumInput() > 0:
        variadic_input_param = Parameter(name="input", kind=Parameter.VAR_POSITIONAL)
        parameters.append(variadic_input_param)
    for kwarg in schema.GetArgumentNames():
        kwarg_name = kwarg
        default_value = Parameter.empty

        # clunky deprecation handling
        if schema.IsDeprecatedArg(kwarg):
            meta = schema.DeprecatedArgMeta(kwarg)
            renamed_arg = meta['renamed_to']
            removed = meta['removed']
            if renamed_arg:
                kwarg = renamed_arg
            if removed:
                continue

        # Try to obtain the default value
        if schema.HasArgumentDefaultValue(kwarg):
            default_value_string = schema.GetArgumentDefaultValueString(kwarg)
            default_value = ast.literal_eval(default_value_string)
        elif schema.IsArgumentOptional(kwarg):
            default_value = None

        kwarg_param = Parameter(name=kwarg_name, kind=Parameter.KEYWORD_ONLY, default=default_value)
        parameters.append(kwarg_param)
    call_sig = Signature(parameters)
    stub = f"def {wrapper_name}{str(call_sig)}:\n    ...\n"
    print(f"\n\nmodule {submodule}\n")
    print(f"def {wrapper_name}{str(call_sig)}:\n    ...\n")
    return submodule, stub

def _generate_prototypes(target_dir):
    import os
    all_registered_ops = _ops._registered_ops()
    # Maybe it would be nicer to have this for all ops in a given module
    files = {}
    # writing a stub file: https://peps.python.org/pep-0484/#stub-files
    for registered_op_name in all_registered_ops:
        schema = _b.TryGetSchema(registered_op_name)
        if not schema:
            print(f"{registered_op_name} doesn't have schema")
            continue
        make_hidden = schema.IsDocHidden() if schema else False
        if make_hidden:
            continue
        submodule, stub = _generate_prototype(registered_op_name)
        submodule = ["fn"] + submodule
        module_path = target_dir + "/" + "/".join(submodule)
        module_stub_file = module_path + "/__init__.pyi"
        if module_path not in files:
            os.makedirs(module_path, exist_ok=True)
            with open(module_stub_file, "w") as f:
                f.write("# <License header>")
                f.write("\n\n")
            files[module_path] = True
        with open(module_stub_file, "a") as f:
            f.write(stub)
            f.write("\n\n")

