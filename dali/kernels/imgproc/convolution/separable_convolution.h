// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_KERNELS_IMGPROC_CONVOLUTION_SEPARABLE_CONVOLUTION_H_
#define DALI_KERNELS_IMGPROC_CONVOLUTION_SEPARABLE_CONVOLUTION_H_

#include "dali/core/tensor_view.h"
#include "dali/kernels/common/utils.h"
#include "dali/kernels/kernel.h"

namespace dali {
namespace kernels {

// should it be channel aware?
// let's assume that yes. and that channels are always dense
template <typename T>
class CyclicPixelWrapper {
 public:
  CyclicPixelWrapper(T* ptr, int length, int num_channels)
      : data(ptr), start(0), end(0), elements(0), length(length), num_channels(num_channels) {}

  void PopPixel() {
    assert(elements > 0);
    elements--;
    start++;
    WrapPosition(start);
  }

  void PushPixel(const T* input) {
    assert(elements < length);
    for (int c = 0; c < num_channels; c++) {
      data[end * num_channels + c] = *input;  // todo offset=end*num_channels; offset++
      input++;
    }
    elements++;
    end++;
    WrapPosition(end);
  }

  void PushPixel(span<const T> input) {
    assert(elements < length);
    for (int c = 0; c < num_channels; c++) {
      data[end * num_channels + c] = input[c];  // todo offset=end*num_channels; offset++
    }
    elements++;
    end++;
    WrapPosition(end);
  }

  T* GetPixelOffset(int idx) {
    assert(idx < elements);
    if (start + idx < length) {
      return data + (start + idx) * num_channels;
    } else {
      return data + (start + idx - length) * num_channels;
    }
  }

  // todo: move the if above loop
  template <typename W>
  void CalculateDot(W* accum, const W* window) {
    assert(elements == length);
    for (int c = 0; c < num_channels; c++) {
      accum[c] = 0;
    }
    for (int idx = 0; idx < length; idx++) {
      const auto* pixel = GetPixelOffset(idx);
      for (int c = 0; c < num_channels; c++) {
        accum[c] += window[idx] * pixel[c];
      }
    }
  }

  // size in pixels
  int Size() {
    return elements;
  }

  bool Empty() {
    return elements == 0;
  }

  int NumChannels() {
    return num_channels;
  }

 private:
  void WrapPosition(int& pos) {
    if (pos == length) {
      pos = 0;
    }
  }

  T* data = nullptr;
  int start = 0;
  int end = 0;  ///< next empty element
  int elements = 0;
  int length = 0;
  int num_channels = 0;
};

template <typename T>
void load_pixel_with_border(CyclicPixelWrapper<T>& cpw, const T* in_ptr, int in_idx, int stride,
                            int axis_size, span<const T> fill_value) {
  if (in_idx < 0) {
    cpw.PushPixel(fill_value);
  } else if (in_idx < axis_size) {
    cpw.PushPixel(in_ptr + in_idx * stride);
  } else {
    cpw.PushPixel(fill_value);
  }
}

template <typename T>
void load_pixel_no_border(CyclicPixelWrapper<T>& cpw, const T* in_ptr, int in_idx, int stride) {
  cpw.PushPixel(in_ptr + in_idx * stride);
}


template <typename T>
struct inspect;

// we're in channel dim
template <int dim = 0, typename Out, typename In, typename W, int ndim>
std::enable_if_t<dim == ndim - 1> traverse_axes(Out* out, const In* in, const W* window, int axis,
                                                const TensorShape<ndim>& shape,
                                                const TensorShape<ndim>& strides, int d,
                                                int64_t offset, span<const In> border_fill, In* input_window_buffer,
                                                span<W> pixel_tmp) {
  auto pixel_stride = strides[axis];
  auto axis_size = shape[axis];
  auto num_channels = shape[ndim - 1];  // channel-last is assumed
  int r = (d - 1) / 2;                  // radius = (diameter - 1) / 2
  // offset <- start of current axis
  auto* out_ptr = out + offset;
  auto* in_ptr = in + offset;
  // prolog: fill input window

  CyclicPixelWrapper<In> input_window(input_window_buffer, d, num_channels);

  constexpr int Border = 0;  // FILL

  int in_idx = -r, out_idx = 0;
  for (in_idx = -r; in_idx < 0; in_idx++) {
    printf("[0]: in_idx: %d, out_idx: %d\n", in_idx, out_idx);
    load_pixel_with_border(input_window, in_ptr, in_idx, pixel_stride, axis_size, border_fill);
  }
  // the window fits in axis
  if (r < axis_size) {
    // we load the window without the last element
    for (; in_idx < r; in_idx++) {
      printf("[1]: in_idx: %d, out_idx: %d\n", in_idx, out_idx);
      load_pixel_no_border(input_window, in_ptr, in_idx, pixel_stride);
    }
    for (; out_idx < axis_size - r; out_idx++, in_idx++) {
      // TODO we assume channel-last, still this can be rewritten as two linear loops if compiler
      // doesn't realize
      // we load last element of the input window corresponding to the out_idx
      printf("[2]: in_idx: %d, out_idx: %d\n", in_idx, out_idx);
      load_pixel_no_border(input_window, in_ptr, in_idx, pixel_stride);
      // we have the windows in contiguous buffers
      input_window.CalculateDot(pixel_tmp.data(), window);
      for (int c = 0; c < num_channels; c++) {
        out_ptr[out_idx * pixel_stride + c] = pixel_tmp[c];  // todo scale & clamp
      }
      // remove one pixel, to make space for next out_idx and in_idx
      input_window.PopPixel();
    }
  }
  // the widow didn't fit
  else {
    // we need to load the rest of the window, just handle all with border condition for simplicity
    for (; in_idx < r; in_idx++) {
      printf("[3]: in_idx: %d, out_idx: %d\n", in_idx, out_idx);
      load_pixel_with_border(input_window, in_ptr, in_idx, pixel_stride, axis_size, border_fill);
    }
  }
  // we need write out the rest of the outputs, the input window is full of data
  for (; out_idx < axis_size; out_idx++, in_idx++) {
    printf("[4]: in_idx: %d, out_idx: %d\n", in_idx, out_idx);
    load_pixel_with_border(input_window, in_ptr, in_idx, pixel_stride, axis_size, border_fill);
    input_window.CalculateDot(pixel_tmp.data(), window);
    for (int c = 0; c < num_channels; c++) {
      out_ptr[out_idx * pixel_stride + c] = pixel_tmp[c];  // todo scale & clamp
    }
    input_window.PopPixel();
  }
}

/// todo:rename
template <int dim = 0, typename Out, typename In, typename W, int ndim>
std::enable_if_t<(dim < ndim - 1)> traverse_axes(Out* out, const In* in, const W* window, int axis,
                                                 const TensorShape<ndim>& shape,
                                                 const TensorShape<ndim>& strides, int d,
                                                 int64_t offset, span<const In> border_fill, In* input_window_buffer,
                                                 span<W> pixel_tmp) {
  if (dim == axis) {
    traverse_axes<dim + 1>(out, in, window, axis, shape, strides, d, offset, border_fill, input_window_buffer,
                           pixel_tmp);
  } else if (dim != axis) {
    for (int64_t i = 0; i < shape[dim]; i++) {
      traverse_axes<dim + 1>(out, in, window, axis, shape, strides, d, offset, border_fill, input_window_buffer,
                             pixel_tmp);
      offset += strides[dim];
    }
  }
}

template <typename Out, typename In, typename W, int ndim>
struct SeparableConvolution {
  KernelRequirements Setup(KernelContext& ctx, const InTensorCPU<In, ndim>& in,
                           const TensorView<StorageCPU, const W, 1>& window, int axis) {
    KernelRequirements req;
    ScratchpadEstimator se;
    int channel_dim = ndim - 1;
    DALI_ENFORCE(channel_dim == ndim - 1,
                 "Only channel-last inputs are supposed, even for 1-channel data");
    se.add<In>(AllocType::Host, GetInputWindowBufSize(in, window, channel_dim));
    se.add<In>(AllocType::Host, GetPixelSize(in, channel_dim));  // fill value
    se.add<W>(AllocType::Host, GetPixelSize(in, channel_dim));  // tmp result
    req.scratch_sizes = se.sizes;
    req.output_shapes.push_back(uniform_list_shape<ndim>(1, in.shape));
    return req;
  }

  void Run(KernelContext& ctx, const TensorView<StorageCPU, Out, ndim> out,
           const TensorView<StorageCPU, const In, ndim>& in,
           const TensorView<StorageCPU, const W, 1>& window, int axis, int channel_dim = -1) {
    DALI_ENFORCE(0 <= axis && axis < ndim - 1, "Cannot apply filter in channel dimension");  // todo
    DALI_ENFORCE(channel_dim == -1 || channel_dim == ndim - 1,
                 "Only no-channel or channel-last are supported.");
    DALI_ENFORCE(channel_dim == ndim - 1,
                 "Only channel-last inputs are supposed, even for 1-channel data");
    int num_channels = GetPixelSize(in, channel_dim);
    int input_window_buf_size = GetInputWindowBufSize(in, window, channel_dim);
    auto* input_window_buffer = ctx.scratchpad->Allocate<In>(AllocType::Host, input_window_buf_size);
    auto* border_fill_buf = ctx.scratchpad->Allocate<In>(AllocType::Host, num_channels);
    auto* pixel_tmp_buf = ctx.scratchpad->Allocate<W>(AllocType::Host, num_channels);
    auto strides = GetStrides(in.shape);
    auto diameter = window.num_elements();

    auto border_fill = make_span(border_fill_buf, num_channels);
    // WIP
    border_fill[0] = 0;
    border_fill[1] = 0;
    border_fill[2] = 0;
    auto pixel_tmp = make_span(pixel_tmp_buf, num_channels);
    // inspect<decltype(in.data)> x;

    traverse_axes<0, Out, In, W, ndim>(out.data, in.data, window.data, axis, in.shape, strides, diameter, 0, border_fill, input_window_buffer, pixel_tmp);
  }

 private:
  int GetInputWindowBufSize(const TensorView<StorageCPU, const In, ndim>& in,
                            const TensorView<StorageCPU, const W, 1>& window, int channel_dim) {
    return GetPixelSize(in, channel_dim) * window.num_elements();
  }
  int GetPixelSize(const TensorView<StorageCPU, const In, ndim>& in, int channel_dim) {
    return channel_dim == -1 ? 1 : in.shape[channel_dim];
  }
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_CONVOLUTION_SEPARABLE_CONVOLUTION_H_