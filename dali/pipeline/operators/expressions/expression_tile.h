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

#ifndef DALI_PIPELINE_OPERATORS_EXPRESSIONS_EXPRESSION_TILE_H_
#define DALI_PIPELINE_OPERATORS_EXPRESSIONS_EXPRESSION_TILE_H_

#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "dali/core/any.h"
#include "dali/pipeline/data/types.h"
#include "dali/pipeline/operators/expressions/arithmetic_meta.h"
#include "dali/pipeline/util/backend2workspace_map.h"
#include "dali/pipeline/workspace/workspace.h"

namespace dali {

/**
 * @brief Describe a tile of data to be processed in expression evaluation.
 */
struct TileDesc {
  int sample_idx;       // id of sample inside within the batch
  int extent_idx;       // the index of tile within this sample_idx
  int64_t extent_size;  // actually covered extent in this tile, the last can be smaller
  int64_t tile_size;    // the size of regular tile
};

inline std::ostream &operator<<(std::ostream &os, const TileDesc &v) {
  os << "{" << v.sample_idx << ", " << v.extent_idx << ", " << v.extent_size << ", " << v.tile_size
     << "}";
  return os;
}

struct TileRange {
  int begin;
  int end;
};

inline std::ostream &operator<<(std::ostream &os, const TileRange &v) {
  os << "{" << v.begin << ", " << v.end << "}";
  return os;
}

using OutputSamplePtr = void *;
using InputSamplePtr = const void *;
using ArgPack = SmallVector<InputSamplePtr, 3>;

struct ExtendedTileDesc {
  TileDesc desc;
  OutputSamplePtr output;
  ArgPack args;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_EXPRESSIONS_EXPRESSION_TILE_H_
