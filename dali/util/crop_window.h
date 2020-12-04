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

#ifndef DALI_UTIL_CROP_WINDOW_H_
#define DALI_UTIL_CROP_WINDOW_H_

#include <functional>
#include <utility>
#include "dali/core/format.h"
#include "dali/core/tensor_layout.h"
#include "dali/core/tensor_shape.h"
#include "dali/core/tensor_shape_print.h"

namespace dali {

struct CropWindow {
  TensorShape<> anchor;
  TensorShape<> shape;

  CropWindow() : anchor{0, 0}, shape{0, 0} {}

  operator bool() const {
    for (int dim = 0; dim < shape.size(); dim++)
      if (shape[dim] <= 0)
        return false;
    return true;
  }

  inline bool operator==(const CropWindow& oth) const {
    return anchor == oth.anchor && shape == oth.shape;
  }

  inline bool operator!=(const CropWindow& oth) const {
    return !operator==(oth);
  }

  inline bool IsInRange(const TensorShape<>& input_shape) const {
    DALI_ENFORCE(input_shape.size() == anchor.size() && input_shape.size() == shape.size(),
                 make_string("Input shape, output shape and anchor must have the "
                             "same number of dimensions. Got:\ninput: ",
                             input_shape, "\nanchor: ", anchor, "\noutput shape:", shape));
    for (int dim = 0; dim < input_shape.size(); dim++)
      if (anchor[dim] < 0 || anchor[dim] + shape[dim] > input_shape[dim])
        return false;
    return true;
  }

  void SetAnchor(TensorShape<> new_anchor) {
    anchor = std::move(new_anchor);
  }

  void SetShape(TensorShape<> new_shape) {
    shape = std::move(new_shape);
  }
};

using CropWindowGenerator =
    std::function<CropWindow(const TensorShape<>& shape, const TensorLayout& shape_layout)>;

}  // namespace dali

#endif  // DALI_UTIL_CROP_WINDOW_H_
