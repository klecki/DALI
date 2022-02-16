// Copyright (c) 2017-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_PIPELINE_DATA_TENSOR_LIST_H_
#define DALI_PIPELINE_DATA_TENSOR_LIST_H_

#include <assert.h>
#include <cstring>
#include <string>
#include <vector>
#include <list>
#include <memory>
#include <utility>
#include "dali/core/tensor_shape.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/buffer.h"
#include "dali/pipeline/data/meta.h"
#include "dali/pipeline/data/types.h"

#include "dali/pipeline/data/tensor_vector.h"


#include "dali/core/tensor_view.h"
#include "dali/core/backend_tags.h"

namespace dali {

// template <typename Backend>
// class TensorVector;

// template <typename Backend>
// using TensorList = TensorVector<Backend>;

}  // namespace dali

#endif  // DALI_PIPELINE_DATA_TENSOR_LIST_H_
