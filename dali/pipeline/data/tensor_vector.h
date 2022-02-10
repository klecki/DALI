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
#include <utility>
#include <vector>

#include "dali/core/access_order.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/tensor.h"
#include "dali/pipeline/data/tensor_list.h"

#include "dali/core/tensor_shape.h"

#include "dali/core/tensor_view.h"
#include "dali/core/backend_tags.h"


namespace dali {


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

  const DALIMeta &GetMeta(int idx) const;

  void SetMeta(int idx, const DALIMeta &meta);

  void set_pinned(bool pinned);

  bool is_pinned() const;

  bool has_data() const;

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

  template <typename SrcBackend>
  void Copy(const TensorList<SrcBackend> &in_tl, AccessOrder order = {});

  template <typename SrcBackend>
  void Copy(const TensorVector<SrcBackend> &in_tv, AccessOrder order = {});

  void ShareData(const TensorList<Backend> &in_tl);

  void ShareData(const TensorVector<Backend> &tv);

  TensorVector<Backend> &operator=(TensorVector<Backend> &&other) noexcept;

  void UpdateViews();

  shared_ptr<TensorList<Backend>> AsTensorList(bool check_contiguity = true);




 private:
  enum class State { contiguous, noncontiguous };
  // Forward declarations in signature, beware
  friend void MakeSampleView(class SampleWorkspace &sample, class HostWorkspace &batch,
                             int data_idx, int thread_idx);
  friend void EnforceCorrectness(class HostWorkspace &batch);

  void PropagateUp();

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
  std::vector<Tensor<Backend>> tensors_;
  std::vector<DALIMeta> dali_meta_;
  Buffer<Backend> contiguous_buffer_;
  std::weak_ptr<void> buffer_bkp_;
  State state_ = State::noncontiguous;
  // pinned status and type info should be uniform
  bool pinned_ = true;
  TypeInfo type_{};
  AccessOrder order_;
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
};

}  // namespace dali

#endif  // DALI_PIPELINE_DATA_TENSOR_VECTOR_H_
