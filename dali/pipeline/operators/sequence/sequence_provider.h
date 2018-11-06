// Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_PIPELINE_OPERATORS_SEQUENCE_SEQUENCE_PROVIDER_H_
#define DALI_PIPELINE_OPERATORS_SEQUENCE_SEQUENCE_PROVIDER_H_

#include "dali/common.h"
#include "dali/error_handling.h"
#include "dali/pipeline/operators/decoder/host_decoder.h"
#include "dali/pipeline/operators/reader/loader/sequence_loader.h"
#include "dali/pipeline/operators/reader/reader_op.h"

namespace dali {

class SequenceProvider : public DataReader<CPUBackend, TensorSequence>, HostDecoder {
 public:
  explicit SequenceProvider(const OpSpec& spec)
      : DataReader<CPUBackend, TensorSequence>(spec), HostDecoder(spec) {
    loader_.reset(new SequenceLoader(spec));
  }

  void RunImpl(SampleWorkspace* ws, const int i) override;

 protected:
  USE_READER_OPERATOR_MEMBERS(CPUBackend, TensorSequence);
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_SEQUENCE_SEQUENCE_PROVIDER_H_