// Copyright (c) 2019-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_PIPELINE_DATA_TENSOR_VECTOR_H_
#define DALI_PIPELINE_DATA_TENSOR_VECTOR_H_

#include <atomic>
#include <cassert>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "dali/core/access_order.h"
#include "dali/core/error_handling.h"
#include "dali/core/tensor_layout.h"
#include "dali/core/tensor_shape.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/buffer.h"
#include "dali/pipeline/data/sample_view.h"
#include "dali/pipeline/data/tensor.h"
#include "dali/pipeline/data/tensor_list.h"
#include "dali/pipeline/data/types.h"


namespace dali {


/**
 * @brief Merges TensorList<Backend> and std::vector<std::shared_ptr<Tensor<Backend>>> APIs
 * providing an uniform way of handling a collection/batch of tensors_.
 *
 * Propagates Buffer calls to every tensor uniformly
 *
 * TODO(klecki): Expected improvements to TensorVector
 * 1. Remove superfluous indirection via shared_ptr to samples.
 * 2. Keep metadata (shape, sample_dim, layout, order) at batch level like we already do with type
 * 3. Detect and convert between contiguous and non-contiguous when possible:
 *    a. CopySample of bigger size
 *    b. Resize with coalesce option
 * 4. Contiguity check
 * @tparam Backend
 */
template <typename Backend>
class DLL_PUBLIC TensorVector {
 public:
  TensorVector();

  /**
   * @brief This constructor allows to create a TensorVector with `batch_size` samples,
   * that will be accessible as individual samples that can currently be individually resized which
   * is still utilized by the legacy operators.
   *
   * TODO(klecki): The API for empty tensor batch container of given number of samples
   * will be adjusted in next releases.
   */
  explicit TensorVector(int batch_size);

  TensorVector(const TensorVector &) = delete;
  TensorVector &operator=(const TensorVector &) = delete;

  DLL_PUBLIC TensorVector<Backend>(TensorVector<Backend> &&other) noexcept;


    /**
   * @brief Checks whether the TensorList is
   * contiguous. It returns true if and only if
   * all of the stored Tensors are densely packed in memory.
   */
  inline bool IsContiguousTensor() const;
  // {
  //   if (num_samples() == 0 || _num_elements() == 0) {
  //     return true;
  //   }
  //   if (!IsContiguous()) {
  //     return false;
  //   }
  //   Index offset = 0;

  //   for (int i = 0; i < shape_.size(); ++i) {
  //     if (offset != offsets_[i]) {
  //       return false;
  //     }
  //     offset += volume(shape_[i]);
  //   }
  //   return true;
  // }

  /**
   * @brief Checks whether the TensorList is
   * a dense Tensor. It returns true if and only if
   * all of the stored Tensors have the same shape
   * and they are densely packed in memory.
   */
  inline bool IsDenseTensor() const;
  // {
  //   if (num_samples() == 0 || _num_elements() == 0) {
  //     return true;
  //   }
  //   if (!IsContiguous()) {
  //     return false;
  //   }
  //   if (!is_uniform(shape_)) {
  //     return false;
  //   }
  //   // shapes are uniform, check if offsets are packed
  //   auto tensor_volume = volume(shape_[0]);
  //   Index offset = 0;

  //   for (int i = 0; i < shape_.size(); ++i) {
  //     if (offset != offsets_[i]) {
  //       return false;
  //     }
  //     offset += tensor_volume;
  //   }
  //   return true;
  // }

  /**
   * @brief Returns a Tensor view with given shape or nullptr if no
   * such exists
   */
  inline Tensor<Backend> *GetViewWithShape(const TensorShape<> &shape);
  // {
  //   for (auto &t : tensor_views_) {
  //     if (t.shape() == shape) {
  //       return &t;
  //     }
  //   }
  //   return nullptr;
  // }

  /**
   * @brief Returns a pointer to Tensor which shares the data
   * with this TensorList and give it the provided shape.
   * Tensor list owns the memory. The tensor obtained through
   * this function stays valid for as long as TensorList data is unchanged.
   */
  DLL_PUBLIC inline Tensor<Backend> * AsReshapedTensor(const TensorShape<> &new_shape);
  // {
  //   auto t = GetViewWithShape(new_shape);
  //   if (t) {
  //     return t;
  //   }

  //   // need to create a new view
  //   DALI_ENFORCE(num_samples() > 0,
  //                "To create a view Tensor, the Tensor List must have at least 1 element.");
  //   DALI_ENFORCE(IsValidType(type()),
  //                "To create a view Tensor, the Tensor List must have a valid data type.");
  //   DALI_ENFORCE(IsContiguousTensor(),
  //                "To create a view Tensor, all tensors in the input TensorList must be contiguous "
  //                "in memory.");
  //   Index product = shape().num_elements();
  //   DALI_ENFORCE(product == volume(new_shape),
  //                "To create a view Tensor, Requested shape need to have the same volume as the "
  //                "tensor list.");

  //   tensor_views_.emplace_back();
  //   auto &tensor = tensor_views_.back();

  //   tensor.set_device_id(device_id());
  //   tensor.ShareData(data_.get_data_ptr(), data_.capacity(), data_.is_pinned(),
  //                    new_shape, type(), order());

  //   return &tensor;
  // }

  DLL_PUBLIC inline Tensor<Backend> * AsTensor();
  // {
  //   // To prevent situation when AsReshapedTensor is called first with some shape, and then
  //   // AsTensor which return non-dense tensor after all
  //   // i.e. [[2], [3], [1]] is not dense but requesting [3, 2] AsReshapedTensor will work
  //   // while AsTensor should not return for that case
  //   DALI_ENFORCE(this->IsDenseTensor(),
  //     "All tensors in the input TensorList must have the same shape and be densely packed.");
  //   auto requested_shape = shape_cat(static_cast<int64_t>(this->num_samples()), shape_[0]);

  //   return this->AsReshapedTensor(requested_shape);
  // }



  AccessOrder order() const {
    return order_;
  }

  /**
   * @brief Sets the associated access order.
   *
   * @note The caller must ensure that if `order` represents a CUDA stream, that stream
   *       is alive when this buffer is destroyed. This extends to buffers with which this
   *       one shares data. Use CUDAStreamPool::instance to get streams with indefinite lifetime.
   *
   * @param order       The new access order (stream or host). If the new order doesn't have
   *                    a value, the function has no effect.
   * @param synchronize If true, an appropriate synchronization is inserted between the old
   *                    and the new order. The caller may specify `false` if appropriate
   *                    synchronization is guaranteed by other means.
   */
  void set_order(AccessOrder order, bool synchronize = true);

  SampleView<Backend> operator[](size_t pos) {
    return {tensors_[pos].raw_mutable_data(), tensors_[pos].shape(), tensors_[pos].type()};
  }

  ConstSampleView<Backend> operator[](size_t pos) const {
    return {tensors_[pos].raw_data(), tensors_[pos].shape(), tensors_[pos].type()};
  }

  int num_samples() const noexcept {
    return curr_num_tensors_;
  }

  void set_sample_dim(int sample_dim);

  int sample_dim() const {
    return sample_dim_;
  }

  size_t nbytes() const noexcept;

  size_t capacity() const noexcept;

  /**
   * @brief Returns the size in bytes of the underlying data chunks
   * TODO(klecki): Temporary API to be reworked, do not use.
   */
  std::vector<size_t> _chunks_nbytes() const;

  /**
   * @brief Returns the real size of the underlying allocations
   * TODO(klecki): Temporary API to be reworked, do not use.
   */
  std::vector<size_t> _chunks_capacity() const;

  const TensorListShape<> &shape() const;

  const TensorShape<> &tensor_shape(int idx) const {
    return tensors_[idx].shape();
  }

  /**
   * @brief Returns a typed pointer to the tensor with the given index.
   */
  template <typename T>
  DLL_PUBLIC inline T* mutable_tensor(int idx) {
    return tensors_[idx].template mutable_data<T>();
  }

  /**
   * @brief Returns a const typed pointer to the tensor with the given index.
   */
  template <typename T>
  DLL_PUBLIC inline const T* tensor(int idx) const {
    return tensors_[idx].template data<T>();
  }

  /**
   * @brief Returns a raw pointer to the tensor with the given index.
   */
  DLL_PUBLIC inline void* raw_mutable_tensor(int idx) {
    return tensors_[idx].raw_mutable_data();
  }

  /**
   * @brief Returns a const raw pointer to the tensor with the given index.
   */
  DLL_PUBLIC inline const void* raw_tensor(int idx) const {
    return  tensors_[idx].raw_data();
  }

  /**
   * @brief Analogue of TensorVector[sample_idx].ShareData(src[src_sample_idx]);
   *
   * The target TensorVector (this) must have enough samples for this to work (see SetSize()).
   * After this operation the TensorVector is converted into non-contiguous.
   *
   * Warning: If the TensorVector was contiguous, the samples that weren't overwritten by this
   * function would still report that they are sharing data. It is assumed that all samples are
   * replaced this way - TODO(klecki): this might be adjusted in follow-up.
   *
   * @param sample_idx index of sample to be set
   * @param src owner of source sample
   * @param src_sample_idx index of source sample in owner.
   */
  DLL_PUBLIC void UnsafeSetSample(int sample_idx, const TensorVector<Backend> &src,
                                  int src_sample_idx);

  /**
   * @brief Analogue of TensorVector[sample_idx].ShareData(owner);
   *
   * The target TensorVector (this) must have enough samples for this to work (see SetSize()).
   * After this operation the TensorVector is converted into non-contiguous.
   *
   * Warning: If the TensorVector was contiguous, the samples that weren't overwritten by this
   * function would still report that they are sharing data. It is assumed that all samples are
   * replaced this way - TODO(klecki): this might be adjusted in follow-up.
   *
   * @param sample_idx index of sample to be set
   * @param src sample owner
   */
  DLL_PUBLIC void UnsafeSetSample(int sample_idx, const Tensor<Backend> &src);


  DLL_PUBLIC void UnsafeSetSample(int sample_idx, const shared_ptr<void> &ptr, size_t bytes,
                                  bool pinned, const TensorShape<> &shape, DALIDataType type,
                                  AccessOrder order = {}, const TensorLayout &layout = "");

  /**
   * @brief Analogue of TensorVector[sample_idx].Copy(src[src_sample_idx]);
   *
   * The target TensorVector (this) must have enough samples for this to work (see SetSize()).
   * It must either be already non-contiguous or the shapes of copied samples must match exactly.
   *
   * Warning: It is assumed that the TensorVector is either first resized to desired shape,
   * or all samples are copied over. Automatically converting to non-contiguous container from
   * contiguous one by invoking copy of non-matching size is not supported yet.
   *
   * @param sample_idx index of sample to be set
   * @param src sample owner
   * @param src_sample_idx index of source sample in owner.
   */
  DLL_PUBLIC void UnsafeCopySample(int sample_idx, const TensorVector<Backend> &src,
                                   int src_sample_idx, AccessOrder order = {});

  DLL_PUBLIC void UnsafeCopySample(int sample_idx, const Tensor<Backend> &src,
                                   AccessOrder order = {});


  DLL_PUBLIC void Resize(const TensorListShape<> &new_shape) {
    DALI_ENFORCE(IsValidType(type()),
                 "TensorVector has no type, 'set_type<T>()' or Resize(shape, type) must be called "
                 "on the TensorVector to set a valid type before it can be resized.");
    return Resize(new_shape, type());
  }

  DLL_PUBLIC void Resize(const TensorListShape<> &new_shape, DALIDataType new_type,
                         BatchState state = BatchState::Default);

  /**
   * Change the number of tensors that can be accessed as samples without the need to
   * set them a size.
   * @param new_size
   */
  void SetSize(int new_size);

  /**
   * @name Setup all the batch properties of this TensorVector the same way as the provided tensor:
   *
   * Precondition: the TensorVector should not have data.
   * Configures: type, layout, pinned, order and dimensionality.
   */
  // @{
  void SetupLike(const Tensor<Backend> &sample) {
    SetupLikeImpl(sample);
  }

  void SetupLike(const TensorVector<Backend> &other) {
    SetupLikeImpl(other);
  }

  void SetupLike(const TensorList<Backend> &other) {
    SetupLikeImpl(other);
  }
  // @}

  void set_type(DALIDataType new_type);

  template <typename T>
  void set_type() {
    set_type(TypeTable::GetTypeId<T>());
  }

  DALIDataType type() const;

  const TypeInfo &type_info() const;

  /** @brief Set uniform layout for all samples in the list */
  void SetLayout(const TensorLayout &layout);

  void SetSkipSample(int idx, bool skip_sample);

  void SetSourceInfo(int idx, const std::string& source_info);

  TensorLayout GetLayout() const;

  const DALIMeta &GetMeta(int idx) const;

  void SetMeta(int idx, const DALIMeta &meta);

  void set_pinned(bool pinned);

  bool is_pinned() const;

  int device_id() const;

  /**
   * @brief Reserve as contiguous tensor list internally
   */
  void reserve(size_t total_bytes);

  /**
   * @brief Reserve as vector of `batch_size` tensors internally
   */
  void reserve(size_t bytes_per_sample, int batch_size);

  /**
   * @brief If the TensorVector is backed by TensorList (contiguous memory)
   */
  bool IsContiguous() const noexcept;

  /**
   * @brief Set the current state for further allocating calls like Resize() or set_type
   *        to use contiguous or noncontiguous backing memory
   *        Setting BatchState::Default allows to change it with every call to Resize().
   */
  void SetContiguous(BatchState state);

  void MakeContiguous(std::weak_ptr<void> owner = {});

  void MakeNoncontiguous();

  void Reset();

  template <typename SrcBackend>
  void Copy(const TensorList<SrcBackend> &in_tl, AccessOrder order = {});

  template <typename SrcBackend>
  void Copy(const TensorVector<SrcBackend> &in_tv, AccessOrder order = {},
            bool use_copy_kernel = false);

  // void ShareData(const TensorList<Backend> &in_tl);
  DLL_PUBLIC void ShareData(const shared_ptr<void> &ptr, size_t bytes, bool pinned,
                            const TensorListShape<> &shape, DALIDataType type,
                            AccessOrder order = {}, const TensorLayout &layout = "");

  void ShareData(const TensorVector<Backend> &tv);

  TensorVector<Backend> &operator=(TensorVector<Backend> &&other) noexcept;

  bool has_data() const;
  bool shares_data() const {
    DALI_FAIL("Do we really need it?");
  }


 private:
  /**
   * @brief Tracking the contiguous/noncontiguous state of the batch.
   * By default we keep what was previously set when resizing and can change it during Resize
   * unless it is enforced.
   */
  class State {
   public:
    // TODO(klecki): Any sensible defaults?
    State() : contiguous_(false), forced_(false) {}
    State(BatchState state, bool forced) {
      DALI_ENFORCE(state != BatchState::Default);
      Setup(state, forced);
    }
    State(const State&) = default;
    State &operator=(const State&) = default;

    /**
     * @brief Override current state.
     */
    void Setup(BatchState state, bool forced = false) {
      if (forced) {
        DALI_ENFORCE(state == BatchState::Contiguous || state == BatchState::Noncontiguous,
                     "Only specific state can be enforced");
      }
      if (state != BatchState::Default) {
        contiguous_ = state == BatchState::Contiguous;
      }
      forced_ = forced;
    }

    /**
     * @brief Update current state obeying the enforced state.
     * BatchState::Default is always allowed and does not change the state
     *
     * State can be changed unless it is enforced, in that case it will raise an error.
     *
     * @return true if the state changed.
     */
    bool Update(BatchState requested_state) {
      if (requested_state == BatchState::Default) {
        return false;
      }
      if (forced_) {
        DALI_ENFORCE(Get() == requested_state,
                     make_string("State cannot be changed as it is enforced to ",
                                 contiguous_ ? "contiguous." : "noncontiguous."));
      }
      if (Get() == requested_state) {
        return false;
      }
      contiguous_ = !contiguous_;
      return true;
    }

    /**
     * @brief Returns true if the requested state changes the current state
     * Validates if the enforced state is not broken
     */
    bool IsStateUpdate(BatchState requested_state) {
      if (requested_state == BatchState::Default) {
        return false;
      }
      if (forced_) {
        DALI_ENFORCE(requested_state == Get(), "The state is enforced and cannot be changed");
      }
      return Get() != requested_state;
    }

    bool IsContiguous() const {
      return contiguous_;
    }

    BatchState Get() const {
      return contiguous_ ? BatchState::Contiguous : BatchState::Noncontiguous;
    }

   private:
    bool contiguous_ = false;
    bool forced_ = false;
  };


  // Forward declarations in signature, beware
  friend void MakeSampleView(class SampleWorkspace &sample, class HostWorkspace &batch,
                             int data_idx, int thread_idx);
  friend void FixBatchPropertiesConsistency(class HostWorkspace &ws, bool contiguous);

  auto& tensor_handle(size_t pos) {
    return tensors_[pos];
  }

  auto& tensor_handle(size_t pos) const {
    return tensors_[pos];
  }

  template <typename T>
  void SetupLikeImpl(const T &other) {
    DALI_ENFORCE(!has_data(),
                "Batch object can be initialized this way only when it isn't allocated.");
    set_type(other.type());
    set_sample_dim(other.shape().sample_dim());
    SetLayout(other.GetLayout());
    set_order(other.order());
    set_pinned(other.is_pinned());
  }

  /**
   * @brief Internal change of contiguity. Unconditionally make the batch non-contiguous.
   * Assumes that the state_ will be adjusted separately
   */
  void DoMakeNoncontiguous();

  /**
   * @brief After RunImpl(SampleWorkspace&) operated on individual samples without propagating
   * the allocation metadata back to the the batch structure, take that metadata from the samples
   * and update it in TensorVector.
   *
   * @param contiguous if the Tensor was previously preallocated and should remain contiguous
   * or be treated as non-contiguous set of individual samples.
   */
  void UpdatePropertiesFromSamples(bool contiguous);

  void resize_tensors(int size);

  void recreate_views();

  /**
   * @brief Check if the metadata provided for new sample match the ones currently set for the batch
   *
   * When setting new sample, the source shape doesn't matter as it is adjusted for individual
   * sample.
   *
   * When setting new sample the `shape_` must be adjusted.
   *
   * @param error_suffix Additional description added to the error message
   */
  void VerifySampleShareConformance(DALIDataType type, int sample_dim, TensorLayout layout,
                                    bool pinned, AccessOrder order,
                                    const std::string &error_suffix = ".");

  /**
   * @brief Check if the metadata provided for new sample match the ones currently set for the batch
   *
   * When copying new sample, pinned status and order of source and destination buffer can be
   * different. Necessary synchronization is handled by the copy itself.
   *
   * When copying new sample the `shape_` must be adjusted.
   *
   * @param error_suffix Additional description added to the error message
   */
  void VerifySampleCopyConformance(DALIDataType type, int sample_dim, TensorLayout layout,
                                   const TensorShape<> &current_shape,
                                   const TensorShape<> &new_shape,
                                   const std::string &error_suffix = ".");

  // Memory backing
  Buffer<Backend> contiguous_buffer_;
  std::weak_ptr<void> buffer_bkp_;
  // Memory, sample aliases and metadata - TODO(klecki): Remove SampleWorkspace and swap to plain
  // Buffer instead of using actual Tensors.
  std::vector<Tensor<Backend>> tensors_;

  // State and metadata that should be uniform regardless of the contiguity state.
  // Sample aliases should match the information stored below.
  State state_;
  int curr_num_tensors_;
  TypeInfo type_{};
  int sample_dim_ = -1;
  TensorListShape<> shape_;
  TensorLayout layout_;

  bool pinned_ = true;
  AccessOrder order_;

  // So we can access the members of other TensorVectors
  // with different template types
  template <typename InBackend>
  friend class TensorVector;
};

}  // namespace dali

#endif  // DALI_PIPELINE_DATA_TENSOR_VECTOR_H_
