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


class _TFRecordReaderImpl():
    """ custom wrappers around ops """

    def __init__(self, path, index_path, features, **kwargs):
        if isinstance(path, list):
            self._path = path
        else:
            self._path = [path]
        if isinstance(index_path, list):
            self._index_path = index_path
        else:
            self._index_path = [index_path]
        self._schema = _b.GetSchema(self._internal_schema_name)
        self._spec = _b.OpSpec(self._internal_schema_name)
        self._device = "cpu"

        self._spec.AddArg("path", self._path)
        self._spec.AddArg("index_path", self._index_path)

        kwargs, self._call_args = _separate_kwargs(kwargs)

        for key, value in kwargs.items():
            self._spec.AddArg(key, value)

        self._features = features

    @property
    def spec(self):
        return self._spec

    @property
    def schema(self):
        return self._schema

    @property
    def device(self):
        return self._device

    def __call__(self, *inputs, **kwargs):
        # We do not handle multiple input sets for Reader as they do not have inputs
        if (len(inputs) > self._schema.MaxNumInput() or len(inputs) < self._schema.MinNumInput()):
            raise ValueError(
                f"Operator {type(self).__name__} expects "
                f"from {self._schema.MinNumInput()} to {self._schema.MaxNumInput()} inputs, "
                f"but received {len(inputs)}.")

        op_instance = _OperatorInstance(inputs, self, **kwargs)
        outputs = {}
        feature_names = []
        features = []
        for i, (feature_name, feature) in enumerate(self._features.items()):
            t_name = op_instance._name
            if len(self._features.items()) > 1:
                t_name += "[{}]".format(i)

            t = _DataNode(t_name, self._device, op_instance)
            op_instance.spec.AddOutput(t.name, t.device)
            op_instance.append_output(t)
            outputs[feature_name] = t
            feature_names.append(feature_name)
            features.append(feature)

        # We know this reader doesn't have any inputs
        if _conditionals.conditionals_enabled():
            _conditionals.register_data_nodes(list(outputs.values()))

        op_instance.spec.AddArg("feature_names", feature_names)
        op_instance.spec.AddArg("features", features)
        return outputs


def _load_readers_tfrecord():
    _TFRecordReaderImpl.__call__.__doc__ = _docstring_generator_call("readers__TFRecord")

    global _cpu_ops
    _cpu_ops = _cpu_ops.union({'readers__TFRecord', 'TFRecordReader'})

    ops_module = sys.modules[__name__]

    class TFRecordReader(_TFRecordReaderImpl, metaclass=_DaliOperatorMeta):
        pass

    class TFRecord(_TFRecordReaderImpl, metaclass=_DaliOperatorMeta):
        pass

    for op_reg_name, internal_schema, op_class in [
        ('readers__TFRecord', 'readers___TFRecord', TFRecord),
        ('TFRecordReader', '_TFRecordReader', TFRecordReader)
    ]:
        op_class.schema_name = op_reg_name
        op_class._internal_schema_name = internal_schema
        op_full_name, submodule, op_name = _process_op_name(op_reg_name)
        module = _internal.get_submodule(ops_module, submodule)
        if not hasattr(module, op_name):
            op_class.__module__ = module.__name__
            setattr(module, op_name, op_class)
            _wrap_op(op_class, submodule)
