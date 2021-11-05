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

#ifndef DALI_PIPELINE_DATA_TENSOR_BATCH_H_
#define DALI_PIPELINE_DATA_TENSOR_BATCH_H_


#include <functional>

#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/buffer.h"
// #include "dali/pipeline/data/tensor.h"
#include "dali/core/util.h"

#include "dali/core/tensor_shape.h"

namespace dali {


using AllocFunc = std::function<shared_ptr<uint8_t>(size_t)>;

template <typename Backend>
class TensorBatch;

// template <typename Backend>
// class TensorProxy;

template <typename Backend>
class Tensor;

// TensorList = TensorBatch
// template <typename Backend>
// class TensorList;


// TODO(klecki): Just for prototyping
template <typename Backend>
using TensorProxy = Tensor<Backend>;

template <typename DstBackend, typename SrcBackend>
void SimpleCopy(TensorBatch<DstBackend> &dst, const TensorBatch<SrcBackend> &src, cudaStream_t stream,
                bool use_copy_kernel);

template <typename DstBackend, typename SrcBackend>
void RichCopy(TensorBatch<DstBackend> &dst, const TensorBatch<SrcBackend> &src, cudaStream_t stream,
          bool use_copy_kernel);


/**
 * @brief Proper, reduced TensorList & TensorVector in one
 */
template <typename Backend>
class TensorBatch {
 public:

  DLL_PUBLIC TensorBatch() = default;
  DLL_PUBLIC TensorBatch(const TensorBatch &) = default;
  DLL_PUBLIC TensorBatch& operator=(const TensorBatch&) = default;
  DLL_PUBLIC TensorBatch(TensorBatch &&) = default;
  DLL_PUBLIC TensorBatch& operator=(TensorBatch&&) = default;

  void AllowSampleAccess() {
    can_modify_proxy_samples_ = true;
  }

  void FinalizeSampleAccess() {
    can_modify_proxy_samples_ = false;
    UpdateViews();
  }

  // DLL_PUBLIC explicit TensorBatch(int batch_size) {
  //   SetSize(batch_size);
  // }


  // Weird TV constructor
  // explicit TensorVector(std::shared_ptr<TensorList<Backend>> tl;


  // can we make it copyable? let it share more and that's it?
  // yes we can! :D


  /** @defgroup LegacyBuffer Legacy buffer and TensorList API
   * Some of those APIs have a bit fuzzy meaning when dealing with samples but the use is too
   * widespread to remove it at once.
   * @{
   */

  inline int64_t _num_elements() const {
    return shape().num_elements();
  }

  inline size_t nbytes() const {
    return _num_elements() * type_.size();
  }

  inline size_t capacity() const {
    return capacity_;
  }

  inline DALIDataType type() const {
    return type_.id();
  }

  inline const TypeInfo &type_info() const {
    return type_;
  }

  inline void set_alloc_func(AllocFunc allocate) {
    std::cout << "[TENSOR_BATCH] >> ALLOC FUNCTION SET ON TENSOR BATCH"<< std::endl;
    allocate_ = std::move(allocate);
  }

  const AllocFunc &alloc_func() const noexcept {
    return allocate_;
  }

  inline void set_pinned(bool pinned) {
    DALI_ENFORCE(!has_data(), "Can only set allocation mode before first allocation");
    DALI_ENFORCE(!allocate_, "Cannot set allocation mode when a custom allocator is used.");
    pinned_ = pinned;
  }

  inline bool is_pinned() const {
    return pinned_;
  }

  int device_id() const {
    return device_;
  }

  void set_device_id(int device) {
    device_ = device;
  }

  inline void set_type(const DALIDataType new_type_id) {
    SetType(new_type_id);
  }

  template <typename T>
  inline void set_type() {
    set_type(TypeTable::GetTypeId<T>());
  }


  inline void reserve(size_t new_num_bytes) {}
  inline void reserve(size_t bytes_per_tensor, int batch_size)  {}

  void reset() {
    std::cout << "[TENSOR_BATCH] >> reset()"<< std::endl;
  }

  void Reset() {
    std::cout << "[TENSOR_BATCH] >> Reset()"<< std::endl;
  }

  bool has_data() {
    return shape().num_elements() != 0 && type_.id() != DALI_NO_TYPE;
  }

  bool IsContiguous() const {
    return state_ == State::contiguous;
  }

  void SetContiguous(bool contiguous) {
    std::cout << "[TENSOR_BATCH] >> SetContiguous("<< contiguous << ")" << std::endl;
    DALI_ENFORCE(contiguous, "TensorList cannot be made noncontiguous");
  }


  /**
   * @brief Returns a typed pointer to the tensor with the given index.
   */
  template <typename T>
  DLL_PUBLIC inline T* mutable_tensor(int idx) {
    // return this->template mutable_data<T>() + tensor_offset(idx);
    return samples_[idx].template mutable_data<T>();
  }

  /**
   * @brief Returns a const typed pointer to the tensor with the given index.
   */
  template <typename T>
  DLL_PUBLIC inline const T* tensor(int idx) const {
    // return this->template data<T>() + tensor_offset(idx);
    return samples_[idx].template data<T>();
  }

  /**
   * @brief Returns a raw pointer to the tensor with the given index.
   */
  DLL_PUBLIC inline void* raw_mutable_tensor(int idx) {
    // return static_cast<void*>(
    //     static_cast<uint8*>(this->raw_mutable_data()) +
    //     (tensor_offset(idx) * type_.size()));
    return samples_[idx].raw_mutable_data();
  }

  /**
   * @brief Returns a const raw pointer to the tensor with the given index.
   */
  DLL_PUBLIC inline const void* raw_tensor(int idx) const {
    // return static_cast<const void*>(
    //     static_cast<const uint8*>(this->raw_data()) +
    //     (tensor_offset(idx) * type_.size()));
    return samples_[idx].raw_data();
  }

  /**
   * @brief Returns the number of tensors in the list.
   */
  DLL_PUBLIC inline size_t num_samples() const {
    return shape_.num_samples();
  }

  /**
   * @brief Returns the number of dimensions of the samples.
   */
  DLL_PUBLIC inline int sample_dim() const {
    return shape_.sample_dim();
  }

  inline span<const int64_t> tensor_shape_span(int idx) const {
     return shape_.tensor_shape_span(idx);
  }

  inline const TensorListShape<> &shape() const {
    return shape_;
  }

  /**
   * @brief Checks whether the TensorList is
   * contiguous. It returns true if and only if
   * all of the stored Tensors are densely packed in memory.
   */
  inline bool IsContiguousTensor() const {
    return false;
    // if (ntensor() == 0 || num_elements_ == 0) {
    //   return true;
    // }
    // if (!IsContiguous()) {
    //   return false;
    // }
    // Index offset = 0;

    // for (int i = 0; i < shape_.size(); ++i) {
    //   if (offset != offsets_[i]) {
    //     return false;
    //   }
    //   offset += volume(shape_[i]);
    // }
    // return true;
  }

  /**
   * @brief Checks whether the TensorList is
   * a dense Tensor. It returns true if and only if
   * all of the stored Tensors have the same shape
   * and they are densely packed in memory.
   */
  inline bool IsDenseTensor() const {
    return false;
    // if (ntensor() == 0 || num_elements_ == 0) {
    //   return true;
    // }
    // if (!IsContiguous()) {
    //   return false;
    // }
    // if (!is_uniform(shape_)) {
    //   return false;
    // }
    // // shapes are uniform, check if offsets are packed
    // auto tensor_volume = volume(shape_[0]);
    // Index offset = 0;

    // for (int i = 0; i < shape_.size(); ++i) {
    //   if (offset != offsets_[i]) {
    //     return false;
    //   }
    //   offset += tensor_volume;
    // }
    // return true;
  }

  /**
   * @brief Returns a Tensor view with given shape or nullptr if no
   * such exists
   */
  inline Tensor<Backend> * GetViewWithShape(const TensorShape<> &shape) {
    // for (auto &t : tensor_views_) {
    //   if (t.shape() == shape) {
    //     return &t;
    //   }
    // }
    // return nullptr;
    return nullptr;
  }

  /**
   * @brief Returns a pointer to Tensor which shares the data
   * with this TensorList and give it the provided shape.
   * Tensor list owns the memory. The tensor obtained through
   * this function stays valid for as long as TensorList data is unchanged.
   */
  DLL_PUBLIC inline Tensor<Backend> * AsReshapedTensor(const TensorShape<> &new_shape) {
    // auto t = GetViewWithShape(new_shape);
    // if (t) {
    //   return t;
    // }

    // // need to create a new view
    // tensor_views_.emplace_back();
    // tensor_views_.back().ShareDataReshape(this, new_shape);

    // return &tensor_views_.back();
    return nullptr;
  }

  /**
   * @brief Returns a pointer to Tensor which shares the data
   * with this TensorList. Tensor list owns the memory. The tensor
   * obtained through this function stays valid for as long
   * as TensorList data is unchanged.
   */
  DLL_PUBLIC inline Tensor<Backend> * AsTensor() {
    // // To prevent situation when AsReshapedTensor is called first with some shape, and then
    // // AsTensor which return non-dense tensor after all
    // // i.e. [[2], [3], [1]] is not dense but requesting [3, 2] AsReshapedTensor will work
    // // while AsTensor should not return for that case
    // DALI_ENFORCE(this->IsDenseTensor(),
    //   "All tensors in the input TensorList must have the same shape and be densely packed.");
    // auto requested_shape = shape_cat(static_cast<int64_t>(this->ntensor()), shape_[0]);

    // return this->AsReshapedTensor(requested_shape);
    return nullptr;
  }


  // So we can access the members of other TensorListes
  // with different template types
  // template <typename InBackend>
  // friend class TensorList;

  inline std::string GetSourceInfo(int idx) const {
    // return meta_[idx].GetSourceInfo();
    return "";
  }

  inline void SetSourceInfo(int idx, const std::string& source_info) {
    // meta_[idx].SetSourceInfo(source_info);
  }

  inline TensorLayout GetLayout() const {
    // Layout is enforced to be the same across all the samples
    // return layout_;
    return {};
  }

  /** @brief Set uniform layout for all samples in the list */
  inline void SetLayout(const TensorLayout &layout) {
    // layout_ = layout;
    // for (auto& meta : meta_)
    //   meta.SetLayout(layout);
  }

  inline void SetSkipSample(int idx, bool skip_sample) {
    // return meta_[idx].SetSkipSample(skip_sample);
  }

  inline bool ShouldSkipSample(int idx) const {
    // return meta_[idx].ShouldSkipSample();
    return false;
  }

  inline const DALIMeta &GetMeta(int idx) const {
    // return meta_[idx];
    static DALIMeta meta = {};
    return meta;
  }

  inline void SetMeta(int idx, const DALIMeta &meta) {
    // meta_[idx] = meta;
  }


  // Not needed
  // static void SetGrowthFactor(double factor) {
  //   // assert(factor >= 1.0);
  //   // growth_factor_ = factor;
  // }
  // static void SetShrinkThreshold(double ratio) {
  //   // assert(ratio >= 0 && ratio <= 1);
  //   // shrink_threshold_ = ratio;
  // }
  // static double GetGrowthFactor() {
  //   // return growth_factor_;
  //   return 1.0;
  // }
  // static double GetShrinkThreshold() {
  //   // return shrink_threshold_;
  //   return 1.0;
  // }

  inline bool shares_data() const {
    return uses_foreign_buffer_;
  }


  /** @} */  // end of LegacyBuffer

   /** @defgroup LegacyVector Legacy TensorVector API
   *
   * @{
   */

  TensorProxy<Backend> &operator[](size_t pos) {
    return samples_[pos];
  }

  const TensorProxy<Backend> &operator[](size_t pos) const {
    return samples_[pos];
  }

  // TODO
  void UpdateViews() {
    std::cout << "[TENSOR_BATCH] >> UpdateViews()" << std::endl;
    uses_foreign_buffer_ = false;
    // std::vector<TensorProxy<Backend>> samples_;
    shape_.resize(samples_.size(), samples_[0].shape().sample_dim());
    capacity_ = 0;
    for (size_t i = 0; i < samples_.size(); i++) {
      shape_.set_tensor_shape(i, samples_[i].shape());
      capacity_ += samples_[i].capacity();
    }
    // TensorListShape<> shape_ = {};

    // Buffer-like properties
    // Buffer<Backend> local_buffer_;  // Contiguous storage
    // TypeInfo type_ = {};            // Data type of underlying storage
    type_ = samples_[0].type_info();
    // AllocFunc allocate_;            // Custom allocation function
    // int64_t num_elements_ = 0;      // The total number of elements
    // size_t capacity_ = 0;  // Total underlying capacity, is bit misleading in non_contiguous state,
    //                       // but what can we do, TODO, do we maintain it?
    // int device_ = CPU_ONLY_DEVICE_ID;  // device the buffer was allocated on
    device_ = samples_[0].device_id();
    // bool shares_data_ = false;         // Whether we aren't using our own allocation ->
    // uses_foreign_buffer_
    bool pinned_ = true;  // Whether the allocation uses pinned memory

  }


  // TODO This is here temporary - we need to replace it with split and build batch
  void SetSize(int new_size) {
    std::cout << "[TENSOR_BATCH] >> SetSize(" << new_size << ")" <<std::endl;
    samples_.resize(new_size);
    state_ = State::noncontiguous;
  }
  /** @} */  // end of LegacyVector


  template <typename T>
  DLL_PUBLIC inline void SetType() {
    SetType(TypeTable::GetTypeId<T>());
  }

  DLL_PUBLIC inline void SetType(DALIDataType new_type_id) {
    Resize(shape_, new_type_id);
  }

  DLL_PUBLIC inline void Resize(const TensorListShape<> &new_shape) {
    DALI_ENFORCE(IsValidType(type_.id()),
                 "TensorList has no type, 'set_type<T>()' or Resize(shape, type) must be called "
                 "on the TensorList to set a valid type before it can be resized.");
    Resize(new_shape, type_.id());
  }

  /**
   * @brief Resize function to allocate a list of tensors. The input vector
   * contains a set of dimensions for each tensor to be allocated in the
   * list.
   *
   * TODO(klecki): Adapted from TL
   * This is a question how much we delegate to the buffer, and how much we do here.
   * We can try to reduce number of allocations at all costs, reallocate every time, or
   * do something in between.
   */
  DLL_PUBLIC inline void Resize(const TensorListShape<> &new_shape, DALIDataType new_type_id) {
    DALI_ENFORCE(IsValidType(new_type_id),
                 "TensorList cannot be resized with invalid type. To zero out the TensorList "
                 "Reset() can be used.");
    // TODO: This is bit compute intensive, and I suggest that we remove type-setting
    // on mutable data.
    if (shape_ == new_shape && type_.id() == new_type_id) {
      return;
    }

    // Calculate the new size
    Index num_samples = new_shape.num_samples(), new_size = new_shape.num_elements();


    std::cout << make_string("> Batch::Resize(", new_size, ", ", new_type_id, ").")
              << std::endl;
    DALI_ENFORCE(new_size >= 0, "Invalid negative buffer size.");

    const auto &new_type = new_type_id == type_.id() ? type_ : TypeTable::GetTypeInfo(new_type_id);

    // TODO(klecki): need to decide if it's only Reinterpret of current data or proper realloc
    // TODO(klecki): Sane behaviour for contiguous and non-contiguous when changing the batch_size
    bool is_reallocation = IsReallocation(new_shape, new_type);

    if (uses_foreign_buffer_) {
      DALI_ENFORCE(!is_reallocation,
                   "You don't want to reallocate if you set the backing buffer manually");
    }

    // TODO(klecki): Exception guarantees?
    // * first, we do a allocation to a helper buffer
    // * next, we crate those shared ptrs in temporary vector
    // * at the end we swap everything
    if (is_reallocation) {
      // We wont fit in the current shape, so we request new allocation
      // TODO
      local_buffer_.resize(new_size, new_type_id);

      state_ = State::contiguous;
    }
    samples_.resize(num_samples);

    // TODO: refactor this as UpdateSamples or something
    // if we did realloc, we need to update with alias shared_ptr
    // TODO(klecki): Optimized case, no need to rewrite shared ptrs
    if (state_ == State::contiguous) {
      uint8_t *base_ptr = static_cast<uint8_t*>(local_buffer_.raw_mutable_data());
      for (int64_t sample_idx = 0; sample_idx < num_samples; sample_idx++) {
        // set the aliasing shared_ptr, the shape, etc
        // TODO
        // samples_[sample_idx].SetTensorFromList(local_buffer_, offset, new_shape[sample_idx],
        //                                        new_type);
        auto sample_alias_ptr = std::shared_ptr<void>(local_buffer_.get_data_ptr(), base_ptr);
        size_t bytes = new_shape[sample_idx].num_elements() * new_type.size();
        samples_[sample_idx].ShareData(sample_alias_ptr, bytes, new_shape[sample_idx],
                                       new_type.id());
        base_ptr += bytes;
      }
    } else {
      for (int64_t sample_idx = 0; sample_idx < num_samples; sample_idx++) {
        // set the aliasing shared_ptr, the shape, etc
        // TODO
        // samples_[sample_idx].InternalResize(new_shape[sample_idx], new_type);
        samples_[sample_idx].Resize(new_shape[sample_idx], new_type_id);
      }
    }

    // Resize the underlying allocation and save the new shape
    // ResizeHelper(new_size, new_type);
    shape_ = new_shape;
    type_ = new_type;

    // Tensor views of this TensorList is no longer valid - TODO(): handle this
    // tensor_views_.clear();

    // TODO(): proper metadata storage & propagation
    // meta_.resize(num_tensor, DALIMeta(layout_));
  }

  // private:
  // we reallocate when we don't have enough space.
  bool IsReallocation(const TensorListShape<> &new_shape, const TypeInfo &new_type) {
    // TODO(klecki): THIS IS WORK IN PROGRESS. It needs to take into account the grow and shrink
    // factors, etc, here we just are keeping the data if it fits.
    if (state_ == State::contiguous) {
      return local_buffer_.is_reallocation(new_shape.num_elements(), new_type.id());
    } else {
      if (shape_.num_samples() < new_shape.num_samples()) {
        return true;
      }
      for (int i = 0; i < new_shape.num_samples(); i++) {
        if (samples_[i].is_reallocation(new_shape[i].num_elements(), new_type.id())) {
          return true;
        }
      }
    }
    return false;
  }

  // Consider what copies do we need and which one can be external
  // no stuff like copy from std::vector<T> etc
  template <typename SrcBackend>
  DLL_PUBLIC inline void Copy(const TensorBatch<SrcBackend> &other, cudaStream_t stream,
                              bool use_copy_kernel = false) {
    RichCopy(*this, other, stream, use_copy_kernel);
    // TODO: all the metadata stuff
  }


  // Not necessary as we now will have just assignment that will overwrite the share
  // TODO(klecki): ShareData -> SetBackingAllocation();

  DLL_PUBLIC inline void SetBackingAllocation(const Buffer<Backend> &buffer);

  DLL_PUBLIC inline void ShareData(const TensorBatch<Backend> &other) {
    if (other.state_ == State::contiguous) {
      local_buffer_.SetExternalAllocation(local_buffer_);
      state_ = State::contiguous;
    } else {
      state_ = State::noncontiguous;
    }
    samples_.resize(other.num_samples());
    for (size_t sample_idx = 0; sample_idx < other.num_samples(); sample_idx++) {
      samples_[sample_idx].ShareData(other.samples_[sample_idx]);
    }
    shape_ = other.shape_;
    type_ = other.type_;
    allocate_ = other.allocate_;
    capacity_ = other.capacity_;
    device_ = other.device_;
    pinned_ = other.pinned_;
    uses_foreign_buffer_ = true;
  }

  /**
   * @brief Wraps the raw allocation. The input pointer must not be nullptr.
   * if the size of the allocation is zero, the TensorList is reset to
   * a default state and is NOT marked as sharing data.
   *
   * The size of the tensor list is calculated based on shape and type or reset to 0
   * if the shape is empty or the type is DALI_NO_TYPE.
   * After calling this function any following call to `set_type` and `Resize`
   * must match the total size of underlying allocation (`num_bytes_`) of
   * shared data or the call will fail.
   * Size can be set to 0 and type to NoType as intermediate step.
   *
   * The TensorList object assumes no ownership of the input allocation,
   * and will not de-allocate it when it is done using it. It is up to
   * the user to manage the lifetime of the allocation such that it
   * persist while it is in use by the Tensor.
   */
  inline void ShareData(const shared_ptr<void> &ptr, size_t bytes, const TensorListShape<> &shape,
                        DALIDataType type = DALI_NO_TYPE)  {
    local_buffer_.SetExternalAllocation(ptr, bytes, shape.num_elements(), type);
    uses_foreign_buffer_ = true;
  }

  /**
   * @brief Wraps the raw allocation. The input pointer must not be nullptr.
   * if the size of the allocation is zero, the TensorList is reset to
   * a default state and is NOT marked as sharing data.
   *
   * The size of the tensor list is calculated based on shape and type or reset to 0
   * if the shape is empty or the type is DALI_NO_TYPE.
   * After calling this function any following call to `set_type` and `Resize`
   * must match the total size of underlying allocation (`num_bytes_`) of
   * shared data or the call will fail.
   * Size can be set to 0 and type to NoType as intermediate step.
   *
   * The TensorList object assumes no ownership of the input allocation,
   * and will not de-allocate it when it is done using it. It is up to
   * the user to manage the lifetime of the allocation such that it
   * persist while it is in use by the Tensor.
   */
  DLL_PUBLIC inline void ShareData(void *ptr, size_t bytes, const TensorListShape<> &shape,
                                   DALIDataType type = DALI_NO_TYPE) {
    ShareData(shared_ptr<void>(ptr, [](void *) {}), bytes, shape, type);
  }

  /**
   * @brief Wraps the raw allocation. The input pointer must not be nullptr.
   * if the size of the allocation is zero, the TensorList is reset to
   * a default state and is NOT marked as sharing data.
   *
   * After wrapping the allocation, the TensorLists size is set to 0,
   * and its type is reset to NoType (if not provided otherwise).
   * After calling this function any following call to `set_type` and `Resize`
   * must match the total size of underlying allocation (`num_bytes_`) of
   * shared data or the call will fail.
   * Size can be set to 0 and type to NoType as intermediate step.
   *
   * The TensorList object assumes no ownership of the input allocation,
   * and will not de-allocate it when it is done using it. It is up to
   * the user to manage the lifetime of the allocation such that it
   * persist while it is in use by the Tensor.
   */
  DLL_PUBLIC inline void ShareData(void *ptr, size_t bytes,
                                   const DALIDataType type = DALI_NO_TYPE) {
    ShareData(shared_ptr<void>(ptr, [](void *) {}), bytes, TensorListShape<>{}, type);
  }


  // TODO need to have the contiguous API for batch
  void BuildBatch(const std::vector<Tensor<Backend>> &batch) {
    state_ = State::noncontiguous;
  }

  void BuildBatch(std::vector<Tensor<Backend>> &&batch) {
    state_ = State::noncontiguous;
  }

  // std::vector<Tensor<Backend>> MoveToSamples() {

  // }

  void ClearBatch() {
    // state_ = Status::contiguous;
    // local_buffer_.reset();
    // samples_.clear();
    // shape_.clear();
    // and so on
  }

  void ReserveBatch(int batch_size) {
    // ClearBatch(); // ?
    // samples_.reserve(batch_size);
    // // TODO: sample_dim?
    // shape_.resize(batch_size, sample_dim);
    // // We need a fixed sizing I think, so reserve the shape, and require the batch to be filled
    // // fully
  }

  void PushSample(const Tensor<Backend> &sample) {
    // samples_.push_back(...);
    // shape_.set_tensor_shape(..., sample.shape());
    // ...
  }

  // TODO: Shape vs shape
  // So we can probably cache it on every resize as TensorProxy cannot be resized
  inline const TensorListShape<> &Shape() const {
    return shape();
  }

 private:
  enum class State
  {
    contiguous,
    noncontiguous
  };
  State state_ = State::contiguous;

  // This corresponds to the ancient ShareData() API - if the data allocation was set as shared
  // with the TensorList, from that point we could resize only within that allocation.
  // This behaviour is kept under renamed API, to indicate that we don't want to use new allocation
  // instead the one that was set for us. The need for matching the exact size of allocation
  // was removed long time ago (https://github.com/NVIDIA/DALI/pull/1327/) and it's the
  // responsibility of the user to not break theirs data by accessing them throught two differently
  // configured TensorBatches.
  bool uses_foreign_buffer_ = false;


  // Batch properties
  std::vector<TensorProxy<Backend>> samples_;
  bool can_modify_proxy_samples_ = false;
  TensorListShape<> shape_ = {};

  // Buffer-like properties
  Buffer<Backend> local_buffer_;  // Contiguous storage
  TypeInfo type_ = {};            // Data type of underlying storage
  AllocFunc allocate_;            // Custom allocation function
  // int64_t num_elements_ = 0;      // The total number of elements
  size_t capacity_ = 0;  // Total underlying capacity, is bit misleading in non_contiguous state,
                         // but what can we do, TODO, do we maintain it?
  int device_ = CPU_ONLY_DEVICE_ID;  // device the buffer was allocated on
  // bool shares_data_ = false;         // Whether we aren't using our own allocation ->
  // uses_foreign_buffer_
  bool pinned_ = true;  // Whether the allocation uses pinned memory

  // template <typename>
  // friend class TensorProxy;


  /** @defgroup ContiguousAccessorFunctions Fallback contiguous accessors
   * Fallback access to contiguous data to TensorList. It should not be used for processing,
   * and can be used only for outputs of the pipeline that were made sure to be contiguous.
   * Currently TensorList is contiguous by design, but it is up to change.
   * @{
   */

  /**
   * @brief Return an un-typed pointer to the underlying storage.
   * The TensorList must be either empty or have a valid type and be contiguous.
   */
  friend void *unsafe_raw_mutable_data(TensorBatch<Backend> &tl) {
    DALI_ENFORCE(tl.IsContiguous(), "Data pointer can be obtain only for contiguous TensorList.");
    return tl.local_buffer_.raw_mutable_data();
  }

  /**
   * @brief Return an un-typed const pointer to the underlying storage.
   * The TensorList must be either empty or have a valid type and be contiguous.
   */
  friend const void *unsafe_raw_data(const TensorBatch<Backend> &tl) {
    DALI_ENFORCE(tl.IsContiguous(), "Data pointer can be obtain only for contiguous TensorList.");
    return tl.local_buffer_.raw_data();
  }

  /**
   * @brief Return the shared pointer, that we can use to correctly share the ownership of sample
   * with.
   */
  friend shared_ptr<void> unsafe_sample_owner(TensorBatch<Backend> &tl, int sample_idx) {
    //{tl.data_, tl.raw_mutable_tensor(sample_idx)};
    return {};
  }

  /** @} */  // end of ContiguousAccessorFunctions
};


// template <typename Backend>
// class SampleAccessLock {
//  public:
//   SampleAccessLock(TensorBatch<Backend> &locked) : locked(locked) {}
//   SampleAccessLock() = delete;
//   SampleAccessLock(const SampleAccessLock &) = delete;
//   SampleAccessLock& operator=(SampleAccessLock &) = delete;
//   SampleAccessLock(SampleAccessLock &&other) = delete;
//   SampleAccessLock& operator=(SampleAccessLock &&other) = delete;

//   ~SampleAccessLock() {
//     locked.FinalizeSampleAccess();
//   }
//  private:
//   TensorBatch<Backend> &locked;
// };

template <typename Backend>
class SampleAccessLock {
 public:
  SampleAccessLock(TensorBatch<Backend> &batch) {
    locked.reserve(1);
    locked.push_back(batch);
    batch.AllowSampleAccess();
  }
  template <typename T, typename = if_array_like<T>>
  SampleAccessLock(T &batches) {
    locked.reserve(batches.size());
    for (size_t i = 0; i < batches.size(); i++) {
      locked.push_back(batches[i]);
      locked.back().get().AllowSampleAccess();
    }
  }
  SampleAccessLock() = delete;
  SampleAccessLock(const SampleAccessLock &) = delete;
  SampleAccessLock &operator=(SampleAccessLock &) = delete;
  SampleAccessLock(SampleAccessLock &&other) = delete;
  SampleAccessLock &operator=(SampleAccessLock &&other) = delete;

  ~SampleAccessLock() {
    for (size_t i = 0; i < locked.size(); i++) {
      locked[i].get().FinalizeSampleAccess();
    }
  }

 private:
  SmallVector<std::reference_wrapper<TensorBatch<Backend>>, 6> locked;
};


/**
 * @brief Sample by sample copy between two batches that have equal shape and size.
 */
template <typename DstBackend, typename SrcBackend>
void SimpleCopy(TensorBatch<DstBackend> &dst, const TensorBatch<SrcBackend> &src, cudaStream_t stream,
                bool use_copy_kernel) {
  DALI_ENFORCE(dst.shape() == src.shape() && dst.type() == src.type(),
               "Data can be copied between Tensor Batches of the same shape and type");
  // Do we need it? For which device?
  // DeviceGuard d(src.device_id());

  const auto &type_info = src.type_info();
  const auto &src_shape = src.shape();
  int num_samples = src_shape.num_samples();

  SmallVector<void *, 256> to;
  SmallVector<const void *, 256> from;
  SmallVector<int64_t, 256> sizes;
  to.reserve(num_samples);
  from.reserve(num_samples);
  sizes.reserve(num_samples);
  for (int i = 0; i < num_samples; i++) {
    to.push_back(dst.raw_mutable_tensor(i));
    from.push_back(src.raw_tensor(i));
    sizes.push_back(src_shape.tensor_size(i));
  }

  type_info.template Copy<DstBackend, SrcBackend>(to.data(), from.data(), sizes.data(), num_samples,
                                                  stream, use_copy_kernel);
  // TODO(klecki): metadata
}

/**
 * @brief Rich copy with bells and whistles. It also has built-in resizing of destination.
 */
template <typename DstBackend, typename SrcBackend>
void RichCopy(TensorBatch<DstBackend> &dst, const TensorBatch<SrcBackend> &src, cudaStream_t stream,
          bool use_copy_kernel) {
  dst.Resize(src.shape(), src.type());
  SimpleCopy(dst, src, stream, use_copy_kernel);
}

/**
 * @brief Contains the buffer representing a Tensor, allowing dynamically typed access to memory.
 *
 * How do we crate this thing????
 * TensorList->Resize()
 * TensorList->ShareData()
 *
 * This should be a member of TensorBatch?
 */
// template <typename Backend>
// class TensorProxy {
//  public:
//   // can we make it copyable? It's non-replecable share of the data?
//   // If the user keeps the copy, they can keep the allocation
//   // Two options: non-copyable, just reference to view the contents OR keep the copy - error prone
//   // with memory hog

//   // TensorBatch can create those

//   // Naming for the purpose of find & replace
//   template <typename T>
//   inline T *tp_data() {
//     return buffer_.mutable_data<T>();
//   }

//   // TODO(klecki): take a look at views, there is some constness fun, consider some convenience
//   // overloads for `const T` etc
//   template <typename T>
//   inline const T *tp_cdata() const {
//     return buffer_.data<T>();
//   }

//   inline void *tp_raw_data() {
//     return buffer_.raw_mutable_data();
//   }

//   inline const void *tp_raw_cdata() const {
//     return buffer_.raw_data();
//   }

//   TensorShape<> Shape() const {
//     return shape_;
//   }


//  private:
//   Buffer<Backend> buffer_;
//   TensorShape<> shape_;
// };

// template <typename Backend>
// class Tensor : public TensorProxy<Backend> {};


}  // namespace dali


#endif  // DALI_PIPELINE_DATA_TENSOR_BATCH_H_