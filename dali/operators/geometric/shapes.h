// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef DALI_OPERATORS_GEOMETRIC_SHAPES_H_
#define DALI_OPERATORS_GEOMETRIC_SHAPES_H_

#include <memory>
#include <vector>
#include "dali/core/tensor_shape.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/core/static_switch.h"

namespace dali {

template <typename Backend>
class Shapes : public Operator<Backend> {
 public:
  Shapes(const Shapes &) = delete;
  explicit Shapes(const OpSpec &spec) : Operator<Backend>(spec) {
    output_type_ = spec.GetArgument<DALIDataType>("type");
    switch (output_type_) {
    case DALI_INT32:
    case DALI_UINT32:
    case DALI_INT64:
    case DALI_UINT64:
    case DALI_FLOAT:
    case DALI_FLOAT64:
      break;
    default:
      {
        auto &name = TypeTable::GetTypeInfo(output_type_).name();
        DALI_FAIL("Operator Shapes can return the output as one of the following:\n"
          "int32, uint32, int64, uint64, float or double;\n"
          "requested: " + name);
        break;
      }
    }
    has_axes_arg_ = spec.HasArgument("axes");
    has_axis_names_arg_ = spec.HasArgument("axis_names");
    DALI_ENFORCE(!(has_axes_arg_ && has_axis_names_arg_),
                 "`axes` and `axis_names` argument are mutually exclusive. Do not provide "
                 "them at the same time.");
    if (has_axes_arg_ || has_axis_names_arg_) {
      full_shape_ = false;
    }
    if (has_axes_arg_) {
      axes_ = spec.GetRepeatedArgument<int>("axes");
      DALI_ENFORCE(!axes_.empty(), "If `axes` argument is specified it should not be empty.");
    } else if (has_axis_names_arg_) {
      axis_names_ = spec.GetArgument<TensorLayout>("axis_names");
      DALI_ENFORCE(!axis_names_.empty(),
                   "If `axis_names` argument is specified it should not be empty.");
    }
  }

  bool CanInferOutputs() const override { return true; }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<Backend> &ws) override {
    output_desc.resize(1);
    output_desc[0].type = TypeTable::GetTypeInfo(output_type_);
    decltype(auto) shape = GetInputShape(ws);

    auto layout = GetInputLayout(ws);
    if (!output_axes_calculated_) {
      output_axes_ = CalculateOutputAxes(shape.sample_dim(), layout);
      output_axes_calculated_ = true;
    }
    output_desc[0].shape = ShapeShape(shape, output_axes_);

    return true;
  }

  void RunImpl(workspace_t<Backend> &ws) override {
    RunBackend(ws);
  }

  template <typename type>
  void ConvertShape(TensorList<CPUBackend> &out, const TensorListShape<> &shape) {
    int n = out.ntensor();
    assert(n == shape.num_samples());
    for (int i = 0; i < n; i++) {
      type *data = out.mutable_tensor<type>(i);
      auto sample_shape = shape.tensor_shape_span(i);
      for (size_t j = 0; j < output_axes_.size(); j++)
        data[j] = sample_shape[output_axes_[j]];
    }
  }

  template <typename type>
  void ConvertShape(TensorVector<CPUBackend> &out, const TensorListShape<> &shape) {
    int n = out.size();
    assert(n == shape.num_samples());
    for (int i = 0; i < n; i++) {
      type *data = out[i].mutable_data<type>();
      auto sample_shape = shape.tensor_shape_span(i);
      for (size_t j = 0; j < output_axes_.size(); j++)
        data[j] = sample_shape[output_axes_[j]];
    }
  }

  template <typename CPUTensorListOrVector>
  void ConvertShape(CPUTensorListOrVector &out, const TensorListShape<> &shape) {
    TYPE_SWITCH(output_type_, type2id, type,
                (int32_t, uint32_t, int64_t, uint64_t, float, double),
      (ConvertShape<type>(out, shape);),
      (DALI_FAIL("Unsupported type for Shapes")));
  }

  void RunBackend(DeviceWorkspace &ws) {
    if (!tmp_.raw_data()) {
      auto &type = TypeTable::GetTypeInfo(output_type_);
      tmp_.set_type(type);
      tmp_.set_pinned(true);
    }

    auto &output = ws.OutputRef<GPUBackend>(0);
    tmp_.Resize(output.shape());
    ConvertShape(tmp_, GetInputShape(ws));
    output.Copy(tmp_, ws.stream());
  }

  void RunBackend(HostWorkspace &ws) {
    ConvertShape(ws.OutputRef<CPUBackend>(0), GetInputShape(ws));
  }

  static TensorListShape<1> ShapeShape(const TensorListShape<> &shape,
                                       const std::vector<int> &output_axes) {
    return uniform_list_shape<1>(shape.num_samples(), { output_axes.size() });
  }

  static const TensorListShape<> &GetInputShape(const DeviceWorkspace &ws) {
    if (ws.InputIsType<GPUBackend>(0)) {
      return ws.InputRef<GPUBackend>(0).shape();
    } else {
      return ws.InputRef<CPUBackend>(0).shape();
    }
  }

  static auto GetInputShape(const HostWorkspace &ws) {
    return ws.InputRef<CPUBackend>(0).shape();
  }

  static const TensorLayout GetInputLayout(const DeviceWorkspace &ws) {
    if (ws.InputIsType<GPUBackend>(0)) {
      return ws.InputRef<GPUBackend>(0).GetLayout();
    } else {
      return ws.InputRef<CPUBackend>(0).GetLayout();
    }
  }

  static const TensorLayout GetInputLayout(const HostWorkspace &ws) {
    return ws.InputRef<CPUBackend>(0).GetLayout();
  }

  std::vector<int> CalculateOutputAxes(int sample_dim, const TensorLayout &layout) {
    if (full_shape_) {
      std::vector<int> result(sample_dim);
      std::iota(result.begin(), result.end(), 0);
      return result;
    } else if (has_axes_arg_) {
      for (auto axis : axes_) {
        DALI_ENFORCE(0 <= axis && axis < sample_dim,
                     make_string_delim("", "Specified axis: \"", axis,
                                       "\" is out of the valid range of [0, ", sample_dim,
                                       ") for given input shape."));
      }
      return axes_;
    } else {  // has_axis_names_arg_
      std::vector<int> result;
      for (auto axis_name : axis_names_) {
        auto dim_idx = layout.find(axis_name);
        DALI_ENFORCE(dim_idx >= 0,
                     make_string("Requested to output dimension", axis_name,
                                 "which is not present in the shape layout", layout.c_str()));
        result.push_back(dim_idx);
      }
      return result;
    }
  }

 private:
  TensorList<CPUBackend> tmp_;
  DALIDataType output_type_ = DALI_INT64;
  bool full_shape_ = true;
  bool has_axes_arg_ = false;
  bool has_axis_names_arg_ = false;
  std::vector<int> axes_ = {};
  TensorLayout axis_names_ = "";
  bool output_axes_calculated_ = false;
  std::vector<int> output_axes_ = {}; // the actual axes we should output
};

}  // namespace dali

#endif  // DALI_OPERATORS_GEOMETRIC_SHAPES_H_
