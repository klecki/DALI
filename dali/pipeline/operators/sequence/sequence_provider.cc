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

#include "dali/pipeline/operators/sequence/sequence_provider.h"

namespace dali {


void SequenceProvider::RunImpl(SampleWorkspace* ws, const int i) {
  const int idx = ws->data_idx();
  auto *output = ws->Output<CPUBackend>(0);

  auto* frame_sequence = prefetched_batch_[idx];

  {
    // First frame to check sizes
    Tensor<CPUBackend> tmp;
    DecodeSingle(frame_sequence->tensors[0].data<uint8_t>(), frame_sequence->tensors[0].size(), &tmp);

    // Calculate shape of sequence tensor, that is Frames x (Frame Shape)
    auto frames_x_shape = std::vector<Index>();
    frames_x_shape.push_back(frame_sequence->tensors.size());
    auto frame_shape = tmp.shape();
    frames_x_shape.insert(frames_x_shape.end(), frame_shape.begin(), frame_shape.end());
    output->Resize(frames_x_shape);
    output->set_type(TypeInfo::Create<uint8_t>());
    // Take a view tensor for first frame and
    auto view_0 = output->SubspaceTensor(0);
    std::memcpy(view_0.raw_mutable_data(), tmp.raw_data(), tmp.size());
  }
  // Rest of frames
  for (size_t frame = 1; frame < frame_sequence->tensors.size(); frame++) {
    auto view_tensor = output->SubspaceTensor(frame);
    DecodeSingle(frame_sequence->tensors[frame].data<uint8_t>(), frame_sequence->tensors[frame].size(), &view_tensor);
    DALI_ENFORCE(view_tensor.shares_data(),
                  "Buffer view was invalidated after image decoding, frames do not match in "
                  "dimensions");
  }
  //     const auto *input_ptr = input.data<uint8_t>();
  //     Index frame_count = metadata_ptr[0];
  //     for (Index frame = 0; frame < frame_count; frame++) {
  //       auto frame_size = metadata_ptr[frame + 1];
  //       if (frame == 0) {
  //         // First frame to check sizes
  //         Tensor<CPUBackend> tmp;
  //         DecodeSingle(input_ptr, frame_size, &tmp);

  //         // Calculate shape of sequence tensor, that is Frames x (Frame Shape)
  //         auto frames_x_shape = std::vector<Index>();
  //         frames_x_shape.push_back(frame_count);
  //         auto frame_shape = tmp.shape();
  //         frames_x_shape.insert(frames_x_shape.end(), frame_shape.begin(), frame_shape.end());
  //         output->Resize(frames_x_shape);
  //         output->set_type(TypeInfo::Create<uint8_t>());
  //         // Take a view tensor for first frame and
  //         auto view_0 = output->SubspaceTensor(frame);
  //         std::memcpy(view_0.raw_mutable_data(), tmp.raw_data(), tmp.size());
  //       } else {
  //         auto view_tensor = output->SubspaceTensor(frame);
  //         DecodeSingle(input_ptr, frame_size, &view_tensor);
  //         DALI_ENFORCE(view_tensor.shares_data(),
  //                      "Buffer view was invalidated after image decoding, frames do not match in "
  //                      "dimensions");
  //       }
  //       input_ptr += frame_size;
  //     }
  //   }

}

DALI_REGISTER_OPERATOR(SequenceProvider, SequenceProvider, CPU);

// DALI_SCHEMA(SequenceReader)
//     .DocStr(
//         "Read [Frame] sequences from a directory representing collection of "
//         "streams")
//     .NumInput(0)
//     .NumOutput(2)  // ([Frames], FrameInfo)
//     .AddArg("file_root",
//             R"code(Path to a directory containing streams (directories representing streams).)code",
//             DALI_STRING)
//     .AddArg("sequence_length",
//             R"code(Lenght of sequence to load for each sample)code", DALI_INT32)
//     .AddParent("LoaderBase");


// DALI_REGISTER_OPERATOR(HostDecoder, HostDecoder, CPU);

// DALI_SCHEMA(HostDecoder)
//     .DocStr(R"code(Decode images on the host using OpenCV.
// When applicable, it will pass execution to faster, format-specific decoders (like libjpeg-turbo).
// Output of the decoder is in `HWC` ordering.
// In case of samples being singular images expects one input, for sequences ([frames], metadata)
// pair is expected, and decode_sequences set to true.)code")
//     .NumInput(1, 2)
//     .NumOutput(1)
//     .AddOptionalArg("output_type", R"code(The color space of output image.)code", DALI_RGB)
//     .AddOptionalArg("decode_sequences", R"code(Is input a sequence of frames.)code", false);

}  // namespace dali


