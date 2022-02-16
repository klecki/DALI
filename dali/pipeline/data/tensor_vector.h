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
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "dali/core/access_order.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/tensor.h"
// #include "dali/pipeline/data/tensor_list.h"

#include "dali/core/tensor_shape.h"

#include "dali/core/tensor_view.h"
#include "dali/core/backend_tags.h"


namespace dali {

template <typename Backend>
class TensorVector;

template <typename Backend>
using TensorList = TensorVector<Backend>;


template <typename DstBackend, typename SrcBackend>
DLL_PUBLIC inline void ShallowCopy(TensorVector<DstBackend> &dst,
                                   const TensorVector<SrcBackend> &src, AccessOrder order = {},
                                   bool use_copy_kernel = false) {
  // TODO: contiguous optimization!!!
  // auto type = src.type();
  // auto layout = other.GetLayout();

  // int dim = other.sample_dim();
  // TensorListShape<> new_shape(other.num_samples(), dim);
  // if (dim)
  //   for (size_t i = 0; i < other.num_samples(); ++i) {
  //     // todo COPY is samplebroken
  //     // DALI_ENFORCE(other.tensor_shape(i).sample_dim() == dim,
  //     //    "TensorList can only have uniform dimensions across all samples, mismatch at index "
  //     //    + std::to_string(i) + " expected Tensor with dim = " + to_string(dim)
  //     //    + " found Tensor with dim = " + to_string(other[i].shape().sample_dim()));
  //     // assert(type == other[i].type());
  //     // assert(layout == other[i].GetLayout());
  //     new_shape.set_tensor_shape(i, other.tensor_shape(i));
  //   }

  if (!order)
    order = src.order() ? src.order() : dst.order();
  order.wait(dst.order());

  // this->Resize(new_shape, type);
  // order.wait(this->order());
  // this->SetLayout(layout);

  auto num_samples = src.num_samples();
  SmallVector<const void*, 256> srcs;
  srcs.reserve(num_samples);
  SmallVector<void*, 256> dsts;
  dsts.reserve(num_samples);
  SmallVector<Index, 256> sizes;
  sizes.reserve(num_samples);
  for (size_t i = 0; i < num_samples; i++) {
    dsts.emplace_back(dst.raw_mutable_tensor(i));
    srcs.emplace_back(src.raw_tensor(i));
    sizes.emplace_back(src.tensor_shape(i).num_elements()); // todo do this on span
    dst.SetMeta(i, src.GetMeta(i));
  }

  use_copy_kernel &= (std::is_same<SrcBackend, GPUBackend>::value || dst.is_pinned()) &&
                     (std::is_same<DstBackend, GPUBackend>::value || src.is_pinned());
  src.type_info().template Copy<DstBackend, SrcBackend>(
      dsts.data(), srcs.data(), sizes.data(), num_samples, order.stream(), use_copy_kernel);
  dst.order().wait(order);
}


/**
 * @brief This class gives access to individual samples of the TensorVector
 *
 */
// class DLL_PUBLIC SampleAccessKey {
//   DLL_PUBLIC SampleAccessKey();
//   // Forward declarations in signature, beware
//   friend void MakeSampleView(class SampleWorkspace &sample, class HostWorkspace &batch,
//                              int data_idx, int thread_idx);

// };

/**
 * @brief Maps DALI Backend to dali::kernels storage backend.
 */
template <typename Backend>
struct storage_tag_map3;

template <>
struct storage_tag_map3<CPUBackend> {
  using type = StorageCPU;
};

template <>
struct storage_tag_map3<GPUBackend> {
  using type = StorageGPU;
};

template <typename Backend>
using storage_tag_map3_t = typename storage_tag_map3<Backend>::type;
/**
 * @brief Merges TensorList<Backend> and std::vector<std::shared_ptr<Tensor<Backend>>> APIs
 * providing an uniform way of handling a collection/batch of tensors_.
 *
 * Propagates Buffer calls to every tensor uniformly
 *
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

  explicit TensorVector(std::shared_ptr<TensorList<Backend>> tl);

  TensorVector(const TensorVector &) = delete;
  TensorVector &operator=(const TensorVector &) = delete;

  DLL_PUBLIC TensorVector<Backend>(TensorVector<Backend> &&other) noexcept;

  AccessOrder order() const {
    return order_;  // todo, fixme
  }

  void set_order(AccessOrder order, bool synchronize = true);

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

  DLL_PUBLIC void SetSample(int dst, const TensorVector<Backend> &owner, int src);

  DLL_PUBLIC void SetSample(int dst, const Tensor<Backend> &owner);

  DLL_PUBLIC void CopySample(int dst, const TensorVector<Backend> &data, int src, AccessOrder order = {});

  DLL_PUBLIC TensorView<storage_tag_map3_t<Backend>, void, DynamicDimensions> operator[](
      int sample_idx) {
    return {tensors_[sample_idx].raw_mutable_data(), tensor_shape(sample_idx), type()};
  }

  DLL_PUBLIC TensorView<storage_tag_map3_t<Backend>, const void, DynamicDimensions> operator[](
      int sample_idx) const {
    return {tensors_[sample_idx].raw_data(), tensor_shape(sample_idx), type()};
  }


  Tensor<Backend> &GetSample(size_t pos) {
    return tensors_[pos];
  }

  const Tensor<Backend> &GetSample(size_t pos) const {
    return tensors_[pos];
  }

  size_t num_samples() const noexcept {
    return shape_.num_samples();
  }

  int sample_dim() const {
    return sample_dim_ == -1 ? 1 : sample_dim_;
  }

  size_t nbytes() const noexcept;

  size_t capacity() const noexcept;

  const TensorListShape<> &shape() const;

  TensorShape<> tensor_shape(int idx) const {
    return shape_[idx];
  }

  DLL_PUBLIC void Resize(const TensorListShape<> &new_shape) {
    DALI_ENFORCE(IsValidType(type()),
                 "TensorVector has no type, 'set_type<T>()' or Resize(shape, type) must be called "
                 "on the TensorVector to set a valid type before it can be resized.");
    return Resize(new_shape, type());
  }

  DLL_PUBLIC void Resize(const TensorListShape<> &new_shape, DALIDataType new_type);

  /**
   * Change the number of tensors, with optional dimensionality. It resizes the internal
   * structures without allocating that data - if new tensors are added, they are initially 0-volume.
   * This can be used to preprate the state for setting or copying in some samples.
   * @param batch_size
   */
  DLL_PUBLIC void SetSize(int batch_size);
  DLL_PUBLIC void SetSize(int batch_size, int sample_dim);

  void set_type(DALIDataType new_type);

  template <typename T>
  void set_type() {
    set_type(TypeTable::GetTypeId<T>());
  }

  DALIDataType type() const;

  const TypeInfo &type_info() const;

  /** @brief Set uniform layout for all samples in the list */
  void SetLayout(const TensorLayout &layout);

  TensorLayout GetLayout() const;

  DALIMeta &GetMeta(int idx);
  const DALIMeta &GetMeta(int idx) const;

  void SetMeta(int idx, const DALIMeta &meta);

  void set_pinned(bool pinned);

  bool is_pinned() const;

  bool has_data() const;

  bool shares_data() const;

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
   * @brief Set the current state if further calls like Resize() or set_type
   *        should use TensorList or std::vector<Tensor> as backing memory
   */
  void SetContiguous(bool contiguous);

  int device_id() const {
    return 0;  // TODO fixme
  }

  void Reset();

  // template <typename SrcBackend>
  // void Copy(const TensorList<SrcBackend> &in_tl, AccessOrder order = {});

  template <typename SrcBackend>
  void Copy(const TensorVector<SrcBackend> &in_tv, AccessOrder order = {}, bool use_copy_kernel = false);

  // void ShareData(const TensorList<Backend> &in_tl);

  void ShareData(const TensorVector<Backend> &tv);

  void ShareData(const shared_ptr<void> &ptr, size_t bytes, bool pinned = false,
                 const TensorListShape<> &shape ={}, DALIDataType type = DALI_NO_TYPE,
                 AccessOrder order = {});

  TensorVector<Backend> &operator=(TensorVector<Backend> &&other) noexcept;

  void UpdateViews();

  shared_ptr<TensorList<Backend>> AsTensorList(bool check_contiguity = true);


  void PropagateUp();

  // TODO
  void set_device_id(int device);

  /**
   * @brief Checks whether the TensorList is
   * contiguous. It returns true if and only if
   * all of the stored Tensors are densely packed in memory.
   */
  inline bool IsContiguousTensor() const {
    if (num_samples() == 0 || shape().num_elements() == 0) {
      return true;
    }
    if (!IsContiguous()) {
      return false;
    }
    const uint8_t *base_ptr = static_cast<const uint8_t*>(tensors_[0].raw_data());
    size_t size = type_info().size();

    for (int i = 0; i < shape_.size(); ++i) {
      if (base_ptr != tensors_[i].raw_data()) {
        return false;
      }
      base_ptr += shape_[i].num_elements() * size;
    }
    return true;
  }

  /**
   * @brief Checks whether the TensorList is
   * a dense Tensor. It returns true if and only if
   * all of the stored Tensors have the same shape
   * and they are densely packed in memory.
   */
  inline bool IsDenseTensor() const {
    if (num_samples() == 0 || shape().num_elements() == 0) {
      return true;
    }
    if (!IsContiguous()) {
      return false;
    }
    if (!is_uniform(shape_)) {
      return false;
    }
    return IsContiguousTensor();
  }

  /**
   * @brief Returns a Tensor view with given shape or nullptr if no
   * such exists
   */
  inline Tensor<Backend> *GetViewWithShape(const TensorShape<> &shape) {
    for (auto &t : tensor_views_) {
      if (t.shape() == shape) {
        return &t;
      }
    }
    return nullptr;
  }

  /**
   * @brief Returns a pointer to Tensor which shares the data
   * with this TensorList and give it the provided shape.
   * Tensor list owns the memory. The tensor obtained through
   * this function stays valid for as long as TensorList data is unchanged.
   */
  DLL_PUBLIC inline Tensor<Backend> * AsReshapedTensor(const TensorShape<> &new_shape) {
    auto t = GetViewWithShape(new_shape);
    if (t) {
      return t;
    }

    // need to create a new view
    DALI_ENFORCE(num_samples() > 0,
                 "To create a view Tensor, the Tensor List must have at least 1 element.");
    DALI_ENFORCE(IsValidType(type()),
                 "To create a view Tensor, the Tensor List must have a valid data type.");
    DALI_ENFORCE(IsContiguousTensor(),
                 "To create a view Tensor, all tensors in the input TensorList must be contiguous "
                 "in memory.");
    Index product = shape().num_elements();
    DALI_ENFORCE(product == volume(new_shape),
                 "To create a view Tensor, Requested shape need to have the same volume as the "
                 "tensor list.");

    tensor_views_.emplace_back();
    auto &tensor = tensor_views_.back();

    tensor.set_device_id(device_id());
    tensor.ShareData(contiguous_buffer_.get_data_ptr(), contiguous_buffer_.capacity(), contiguous_buffer_.is_pinned(),
                     new_shape, type(), order());

    return &tensor;
  }

  /**
   * @brief Returns a pointer to Tensor which shares the data
   * with this TensorList. Tensor list owns the memory. The tensor
   * obtained through this function stays valid for as long
   * as TensorList data is unchanged.
   */
  DLL_PUBLIC inline Tensor<Backend> * AsTensor() {
    // To prevent situation when AsReshapedTensor is called first with some shape, and then
    // AsTensor which return non-dense tensor after all
    // i.e. [[2], [3], [1]] is not dense but requesting [3, 2] AsReshapedTensor will work
    // while AsTensor should not return for that case
    DALI_ENFORCE(this->IsDenseTensor(),
      "All tensors in the input TensorList must have the same shape and be densely packed.");
    auto requested_shape = shape_cat(static_cast<int64_t>(this->num_samples()), shape_[0]);

    return this->AsReshapedTensor(requested_shape);
  }



 private:
  enum class State { contiguous, noncontiguous };
  // Forward declarations in signature, beware
  friend void MakeSampleView(class SampleWorkspace &sample, class HostWorkspace &batch,
                             int data_idx, int thread_idx);
  friend void EnforceCorrectness(class HostWorkspace &batch);


  auto& tensor_handle(size_t pos) {
    return tensors_[pos];
  }

  const auto& tensor_handle(size_t pos) const {
    return tensors_[pos];
  }

  /**
   * @brief Adjust the metadata structures size, if new tensors were added make them 0-volume
   *
   * Sample dimension is assumed to be meaningfull (non-negative)
   */
  void resize_tensors(int batch_size);
  void resize_tensors(int batch_size, int sample_dim);

  void update_sample_dim(int sample_dim);

  // Fix sample dim in sample views - due to duplication of shapes.
  void war_update_sample_dim();

  /**
   * @brief Propagate all the stuff like pinned, order, etc before we do reallocation?
   *
   */
  void propagate_properties();

  void propagate_properties_to_contiguous();

  void propagate_properties_to_samples();
  void propagate_properties_to_samples(int idx);

  void update_view(int idx);
  void check_consistency();


  bool has_data_ = false;
  int device_id_ = -1;
  std::vector<Tensor<Backend>> tensors_;
  std::vector<DALIMeta> dali_meta_;
  Buffer<Backend> contiguous_buffer_;
  std::weak_ptr<void> buffer_bkp_;
  State state_ = State::noncontiguous;
  // pinned status and type info should be uniform
  bool pinned_ = true;
  TypeInfo type_{};
  AccessOrder order_;


  // In order to not leak memory (and make it slightly faster)
  // when sharing data with a Tensor, we will store a pointer to
  // Tensor that shares the data with this TensorList (valid only
  // if IsDenseTensor returns true)
  std::vector<Tensor<Backend>> tensor_views_;

  /**
   * @brief Although sample_dim_ duplicates what can be set in shape_, we use it for lazy shape
   * initialization, where -1 means that this TensorBatch did not receive sample_dim yet.
   * One can consider if sample_dim should be always set explicitly or inferred from the first
   * sample that is set in it.
   */
  int sample_dim_ = -1;
  TensorListShape<> shape_{};
  TensorLayout layout_;

  // So we can access the members of other TensorVectors
  // with different template types
  template <typename InBackend>
  friend class TensorVector;

    /**
   * @brief Return an un-typed pointer to the underlying storage.
   * The TensorList must be either empty or have a valid type and be contiguous.
   */
  friend void *unsafe_raw_mutable_data(TensorList<Backend> &tl) {
    DALI_ENFORCE(tl.IsContiguous(), "Data pointer can be obtain only for contiguous TensorList.");
    return tl.contiguous_buffer_.raw_mutable_data();
  }

  /**
   * @brief Return an un-typed const pointer to the underlying storage.
   * The TensorList must be either empty or have a valid type and be contiguous.
   */
  friend const void *unsafe_raw_data(const TensorList<Backend> &tl) {
    DALI_ENFORCE(tl.IsContiguous(), "Data pointer can be obtain only for contiguous TensorList.");
    return tl.contiguous_buffer_.raw_data();
  }

  /**
   * @brief Return the shared pointer, that we can use to correctly share the ownership of sample
   * with.
   */
  friend shared_ptr<void> unsafe_sample_owner(TensorList<Backend> &tl, int sample_idx) {
    // create new aliasing pointer to current data allocation, so we share the use count
    // and the deleter correctly.
    return {tl.tensors_[sample_idx].get_data_ptr(), tl.raw_mutable_tensor(sample_idx)};
  }
};

}  // namespace dali

#endif  // DALI_PIPELINE_DATA_TENSOR_VECTOR_H_
