// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_PIPELINE_DATA_TENSOR_PROXY_H_
#define DALI_PIPELINE_DATA_TENSOR_PROXY_H_

#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/buffer.h"
#include "dali/pipeline/data/tensor.h"

namespace dali {

/**
 * @brief Proper, reduced TensorList & TensorVector in one
 */
template <typename Backend>
class TensorBatch {
 public:

  // can we make it copyable? let it share more and that's it?


  /**
   * @brief Resize function to allocate a list of tensors. The input vector
   * contains a set of dimensions for each tensor to be allocated in the
   * list.
   *
   * TODO(klecki): Adapted from TL
   */
  DLL_PUBLIC inline void Resize(const TensorListShape<> &new_shape, DALIDataType new_type) {
    // Calculate the new size
    Index num_samples = new_shape.num_samples(), new_size = new_shape.num_elements();
    DALI_ENFORCE(new_size >= 0, "Invalid negative buffer size.");

    // TODO(klecki): need to decide if it's only Reinterpret of current data or proper realloc

    bool is_reallocation = IsReallocation(new_shape, new_type);

    if (is_reallocation) {
      // We do a contiguous allocation if we can
      local_buffer_.ResizeHelper(new_size, new_type);
      state_ = State::contiguous;
      samples_.resize(num_samples);
    }

    int64_t offset = 0;
    // TODO: refactor this as UpdateSamples or something
    // if we did realloc, we need to update with alias shared_ptr
    if (state_ == State::contiguous) {
      for (int64_t sample_idx; sample_idx < num_samples; sample_idx++) {
        // set the aliasing shared_ptr, the shape, etc
        samples_[i].SetTensorFromList(local_buffer_, offset, shape[i]);
        offset += shape[i].num_elements();
      }
    } else {
      // just update the samples
    }

    // Resize the underlying allocation and save the new shape
    // ResizeHelper(new_size, new_type);
    shape_ = new_shape;

    // Tensor views of this TensorList is no longer valid - TODO(klecki): handle this
    // tensor_views_.clear();

    // TODO(klecki): proper metadata storage & propagation
    // meta_.resize(num_tensor, DALIMeta(layout_));
  }

  void BuildBatch(const std::vector<Tensor<Backend>> &batch) {

  }

  void BuildBatch(std::vector<Tensor<Backend>> &&batch) {

  }

  std::vector<Tensor<Backend>> MoveToSamples() {

  }

  void ClearBatch() {
    state_ = Status::contiguous;
    local_buffer_.reset();
    samples_.clear();
    // shape_.clear();
    // and so on
  }

  void ReserveBatch(int batch_size) {
    ClearBatch(); // ?
    samples_.reserve(batch_size);
    // TODO: sample_dim?
    shape_.resize(batch_size, sample_dim);
    // We need a fixed sizing I think, so reserve the shape, and require the batch to be filled
    // fully
  }

  void PushSample(const Tensor<Backend> &sample) {
    samples_.push_back(...);
    shape_.set_tensor_shape(..., sample.shape());
    ...
  }

  // So we can probably cache it on every resize as TensorProxy cannot be resized
  const TensorListShape<> &Shape() const {
    return shape_;
  }

 private:
  enum class State { contiguous, noncontiguous };
  State state_ = State::contiguous;
  Buffer<Backend> local_buffer_;
  std::vector<TensorProxy<Backend>> samples_;
  TensorListShape<> shape_;
  TypeInfo type_;

  template <typename>
  friend class TensorProxy;
};

/**
 * @brief Contains the buffer representing a Tensor, allowing dynamically typed access to memory.
 *
 * How do we crate this thing????
 * TensorList->Resize()
 * TensorList->ShareData()
 *
 * This should be a member of TensorBatch?
 */
template <typename Backend>
class TensorProxy {
 public:

  // can we make it copyable? It's non-replecable share of the data?
  // If the user keeps the copy, they can keep the allocation
  // Two options: non-copyable, just reference to view the contents OR keep the copy - error prone with memory hog

  // TensorBatch can create those

  // Naming for the purpose of find & replace
  template <typename T>
  inline T* tp_data() {
    return buffer_.mutable_data<T>();
  }

  // TODO(klecki): take a look at views, there is some constness fun, consider some convenience
  // overloads for `const T` etc
  template <typename T>
  inline const T* tp_cdata() const {
    return buffer_.data<T>();
  }

  inline void* tp_raw_data() {
    return buffer_.raw_mutable_data();
  }

  inline const void* tp_raw_cdata() const {
    return buffer_.raw_data();
  }

  TensorShape<> Shape() const {
    return shape_;
  }



 private:
  Buffer<Backend> buffer_;
  TensorShape<> shape_;
};

template <typename Backend>
class Tensor : public TensorProxy<Backend> {

};


}  // namespace dali


#endif  // DALI_PIPELINE_DATA_TENSOR_PROXY_H_