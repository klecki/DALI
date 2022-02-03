// Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/pipeline/data/tensor_vector.h"
#include <cstdint>
#include "dali/core/error_handling.h"

namespace dali {

/**
 * @brief Check if both shared pointers have the same managed pointer (not the one returned by
 * .get())
 */
bool same_owner(const std::shared_ptr<void> &x, const std::shared_ptr<void> &y) {
    if (x.owner_before(y) || y.owner_before(x))
        return false;
    return true;
}


template <typename Backend>
TensorVector<Backend>::TensorVector() = default;


template <typename Backend>
TensorVector<Backend>::TensorVector(int batch_size) {
  resize_tensors(batch_size);
}


// template <typename Backend>
// TensorVector<Backend>::TensorVector(std::shared_ptr<TensorList<Backend>> tl)
//     : views_count_(0), curr_tensors_size_(0), tl_(std::move(tl)) {
//   assert(tl_ && "Construction with null TensorList is illegal");
//   pinned_ = tl_->is_pinned();
//   type_ = tl_->type_info();
//   state_ = State::contiguous;
//   resize_tensors(tl_->num_samples());
//   UpdateViews();
// }


template <typename Backend>
TensorVector<Backend>::TensorVector(TensorVector<Backend> &&other) noexcept {
  state_ = other.state_;
  pinned_ = other.pinned_;
  contiguous_buffer_ = std::move(other.contiguous_buffer_);
  type_ = std::move(other.type_);
  tensors_ = std::move(other.tensors_);
  dali_meta_ = std::move(other.dali_meta_);
  shape_ = std::move(other.shape);
  sample_dim_ = other.sample_dim_;
  // for (auto &t : tensors_) {
  //   if (t) {
  //     if (auto *del = std::get_deleter<ViewRefDeleter>(t->data_)) del->ref = &views_count_;
  //   }
  // }

  other.views_count_ = 0;
  other.curr_tensors_size_ = 0;
  other.contiguous_buffer_.clear();
  other.tensors_.clear();
  other.Reset();
}


template <typename Backend>
size_t TensorVector<Backend>::nbytes() const noexcept {
  // todo fixme
  // if (state_ == State::contiguous) {
  //   return tl_->nbytes();
  // }
  // else
  size_t total_nbytes = 0;
  for (const auto &t : tensors_) {
    total_nbytes += t.nbytes();
  }
  return total_nbytes;
}


template <typename Backend>
size_t TensorVector<Backend>::capacity() const noexcept {
  // todo fixme
  // if (state_ == State::contiguous) {
  //   return tl_->capacity();
  // }
  // else
  size_t total_capacity = 0;
  for (const auto &t : tensors_) {
    total_capacity += t.capacity();
  }
  return total_capacity;
}


template <typename Backend>
void TensorVector<Backend>::SetSample(int dst, const TensorVector<Backend> &owner, int src) {
  // TODO checks
  if (type() == DALI_NO_TYPE && owner.type() != DALI_NO_TYPE) {
    set_type(owner.type());
  }
  DALI_ENFORCE(type() == owner.type(), "Sample must have the same type as batch");

  tensors_[dst].ShareData(owner.tensors_[src]);
}

template <typename Backend>
void TensorVector<Backend>::SetSample(int dst, const Tensor<Backend> &owner) {
  // TODO checks
  // DALI_ENFORCE(owner.shape().sample_dim() == shape().sample_dim(), "Sample must have the same
  // dim");
  if (type() == DALI_NO_TYPE && owner.type() != DALI_NO_TYPE) {
    set_type(owner.type());
  }
  DALI_ENFORCE(type() == owner.type(), "Sample must have the same type as batch");
  // kind (pinned?), order, layout, etc...
  // The metadata

  if (tensors_[dst].shape().num_elements() != owner.shape().num_elements()) {
    SetContiguous(false);
  }
  tensors_[dst].ShareData(owner);
  // todo v update shape
  // shape().set_tensor_shape(idx, owner.shape());
}

template <typename Backend>
void TensorVector<Backend>::CopySample(int dst, const TensorVector<Backend> &data, int src,
                                       AccessOrder order) {
  // TODO checks
  // DALI_ENFORCE(owner.shape().sample_dim() == shape().sample_dim(), "Sample must have the same
  // dim");
  if (type() == DALI_NO_TYPE && data.type() != DALI_NO_TYPE) {
    set_type(data.type());
  }
  DALI_ENFORCE(type() == data.type(), "Sample must have the same type as batch");
  // kind (pinned?), order, layout, etc...
  // The metadata

  if (tensors_[dst].shape().num_elements() != data.tensors_[src].shape().num_elements()) {
    SetContiguous(false);
  }
  tensors_[dst].Copy(data.tensors_[src], order);
  // todo v update shape
  // shape().set_tensor_shape(idx, owner.shape());
}

template <typename Backend>
const TensorListShape<> &TensorVector<Backend>::shape() const {
  return shape_;
}

template <typename Backend>
void TensorVector<Backend>::set_order(AccessOrder order, bool synchronize) {
  // TODO FIXME
  // Optimization: synchronize only once, if needed.
  // TODO need a better check if there is data in contiguous buffer
  if (this->order().is_device() && order && synchronize) {
    bool need_sync = !contiguous_buffer_.expired(); // todo has data?
    if (!need_sync) {
      for (auto &t : tensors_) {
        if (t.has_data()) {
          need_sync = true;
          break;
        }
      }
    }
    if (need_sync)
      this->order().wait(order);
  }

  // if we need to replicate data based on the contiguous_buffer_, we keep the order updated
  contiguous_buffer_.set_order(order);
  for (auto &t : tensors_)
    t.set_order(order, false);
  order_ = order;
}

template <typename Backend>
void TensorVector<Backend>::Resize(const TensorListShape<> &new_shape, DALIDataType new_type) {
  DALI_ENFORCE(IsValidType(new_type),
                "TensorVector cannot be resized with invalid type. To zero out the TensorVector "
                "Reset() can be used.");
  resize_tensors(new_shape.num_samples());
  if (type_.id() != new_type) {
    type_ = TypeTable::GetTypeInfo(new_type);
  }
  sample_dim_ = new_shape.sample_dim();
  shape_ = new_shape;


  if (state_ == State::contiguous) { // todo or policy == Coalesce
    // even if we don't have anything here, we don't care, and still use it to allocate
    auto buffer = lock(std::move(contiguous_buffer_));
    // buffer.
    int64_t num_tensor = new_shape.num_samples(), new_size = new_shape.num_elements();
    buffer.resize(new_size, new_type);
    uint8_t *base_ptr = static_cast<uint8_t*>(buffer.raw_mutable_data());
    for (int64_t i = 0; i < num_tensor; i++) {
      auto tensor_size = volume(new_shape.tensor_shape_span(i));

      std::shared_ptr<void> sample_alias(buffer.get_data_ptr(), base_ptr);
      // todo, convert this to buffers
      tensors_[i].ShareData(sample_alias, tensor_size * type_.size(), buffer.is_pinned(),
                            new_shape[i], new_type, order());
      base_ptr += tensor_size * type_.size();
    }
    contiguous_buffer_ = std::move(buffer);
    return;

    // This is corresponding part from TL:
    // // Calculate the new size
    // Index num_tensor = new_shape.size(), new_size = 0;
    // offsets_.resize(num_tensor);
    // for (Index i = 0; i < num_tensor; ++i) {
    //   auto tensor_size = volume(new_shape[i]);

    //   // Save the offset of the current sample & accumulate the size
    //   offsets_[i] = new_size;
    //   new_size += tensor_size;
    // }
    // DALI_ENFORCE(new_size >= 0, "Invalid negative buffer size.");

    // // Resize the underlying allocation and save the new shape
    // data_.resize(new_size, new_type);
    // shape_ = new_shape;

    // // Tensor views of this TensorList is no longer valid
    // tensor_views_.clear();

    // meta_.resize(num_tensor, DALIMeta(layout_));
  }


  // if (state_ == State::contiguous) {
  //   tl_->Resize(new_shape, new_type);
  //   UpdateViews();
  //   return;
  // }

  auto buffer = lock(std::move(contiguous_buffer_));

  for (int64_t i = 0; i < shape_.num_samples(); i++) {
    // problem: we have a case, where we were first contiguous, now we want non-contiguous.
    // so we should probably set the buffers as not sharing data.
    if (same_owner(buffer.get_data_ptr(), tensors_[i].get_data_ptr())) {
      // if we have same owner as contiguous buffer, we can assume we share into that contiguous
      // buffer, se we need to break this share, to be able to resize sample-wise.
      // todo, convert to regular assert
      DALI_ENFORCE(tensors_[i].shares_data());
      tensors_[i].Reset();
    }
    tensors_[i].Resize(new_shape[i], new_type);
  }
}


template <typename Backend>
void TensorVector<Backend>::SetSize(int batch_size) {
  SetSize(batch_size, sample_dim_);
  // resize_tensors(batch_size);
}


template <typename Backend>
void TensorVector<Backend>::SetSize(int batch_size, int sample_dim) {
  DALI_ENFORCE(batch_size >= 0,
               make_string("Batch size must be non-negative, got: ", batch_size, "."));
  DALI_ENFORCE(sample_dim >= 0,
               make_string("Sample dimension must be non-negative or -1, got: ", sample_dim, "."));
  // DALI_ENFORCE(new_type_id != DALI_NO_TYPE, "new_type_id must represent a valid type.");
  // preserve sample_dim if we got -1 as argument
  sample_dim = sample_dim == -1 ? sample_dim_ : sample_dim;

  if (sample_dim == -1) {
    // We didn't get new sample dim, and we currently don't have one.
    // Just expand the metadata structures
    tensors_.resize(batch_size);
    shape_.resize(batch_size);
    dali_meta_.resize(batch_size);
  } else {


  }


  resize_tensors(batch_size, sample_dim);
}




template <typename Backend>
void TensorVector<Backend>::set_type(DALIDataType new_type_id) {
  DALI_ENFORCE(new_type_id != DALI_NO_TYPE, "new_type_id must represent a valid type.");
  if (type_.id() == new_type_id)
    return;
  type_ = TypeTable::GetTypeInfo(new_type_id);
}


template <typename Backend>
DALIDataType TensorVector<Backend>::type() const {
  return type_.id();
}

template <typename Backend>
const TypeInfo &TensorVector<Backend>::type_info() const {
  return type_;
}


template <typename Backend>
void TensorVector<Backend>::SetLayout(const TensorLayout &layout) {
  if (state_ == State::noncontiguous) {
    DALI_ENFORCE(!tensors_.empty(), "Layout cannot be set uniformly for empty batch");
  }
  tl_->SetLayout(layout);
  for (auto &t : tensors_) {
    t.SetLayout(layout);
  }
}


template <typename Backend>
TensorLayout TensorVector<Backend>::GetLayout() const {
  if (state_ == State::contiguous) {
    auto layout = tl_->GetLayout();
    if (!layout.empty()) return layout;
  }
  if (curr_tensors_size_ > 0) {
    auto layout = tensors_[0].GetLayout();
    for (size_t i = 1; i < curr_tensors_size_; i++) assert(layout == tensors_[i]->GetLayout());
    return layout;
  }
  return {};
}


template <typename Backend>
const DALIMeta &TensorVector<Backend>::GetMeta(int idx) const {
  assert(static_cast<size_t>(idx) < curr_tensors_size_);
  return tensors_[idx].GetMeta();
}


template <typename Backend>
void TensorVector<Backend>::SetMeta(int idx, const DALIMeta &meta) {
  assert(static_cast<size_t>(idx) < curr_tensors_size_);
  tensors_[idx].SetMeta(meta);
}


template <typename Backend>
void TensorVector<Backend>::set_pinned(bool pinned) {
  // Store the value, in case we pin empty vector and later call Resize
  pinned_ = pinned;
}


template <typename Backend>
bool TensorVector<Backend>::is_pinned() const {
  return pinned_;
}


template <typename Backend>
void TensorVector<Backend>::reserve(size_t total_bytes) {
  if (state_ == State::noncontiguous) {
    tensors_.clear();
  }
  state_ = State::contiguous;
  // TODO: the reserve doesn't work with weak ptr
  // tl_->reserve(total_bytes);
  // UpdateViews();
}


template <typename Backend>
void TensorVector<Backend>::reserve(size_t bytes_per_sample, int batch_size) {
  assert(batch_size > 0);
  state_ = State::noncontiguous;
  resize_tensors(batch_size);
  for (int64_t i = 0; i < shape_.num_samples(); i++) {
    tensors_[i].reserve(bytes_per_sample);
  }
}


template <typename Backend>
bool TensorVector<Backend>::IsContiguous() const noexcept {
  return state_ == State::contiguous;
}


template <typename Backend>
void TensorVector<Backend>::SetContiguous(bool contiguous) {
  if (contiguous) {
    state_ = State::contiguous;
  } else {
    state_ = State::noncontiguous;
  }
  // TODO: get rid of weak buffer, make it free stuff when we switch to non contiguous
}


template <typename Backend>
void TensorVector<Backend>::Reset() {
  tensors_.clear();
  contiguous_buffer_.reset();
  dali_meta_.clear();
  type_ = {};
  sample_dim_ = -1;
  shape_ = {};
}


template <typename Backend>
template <typename SrcBackend>
void TensorVector<Backend>::Copy(const TensorList<SrcBackend> &in_tl, AccessOrder order) {
  SetContiguous(true);
  type_ = in_tl.type_info();
  tl_->Copy(in_tl, order);

  resize_tensors(tl_->num_samples());
  UpdateViews();
}


template <typename Backend>
template <typename SrcBackend>
void TensorVector<Backend>::Copy(const TensorVector<SrcBackend> &in_tv, AccessOrder order) {
  SetContiguous(true);
  type_ = in_tv.type_;
  tl_->Copy(in_tv, order);

  resize_tensors(tl_->num_samples());
  UpdateViews();
}


template <typename Backend>
void TensorVector<Backend>::ShareData(const TensorList<Backend> &in_tl) {
  SetContiguous(true);
  type_ = in_tl.type_info();
  pinned_ = in_tl.is_pinned();
  tl_->ShareData(in_tl);

  resize_tensors(in_tl.num_samples());
  UpdateViews();
}

template <typename Backend>
void TensorVector<Backend>::ShareData(const TensorVector<Backend> &tv) {
  type_ = tv.type_;
  state_ = tv.state_;
  pinned_ = tv.is_pinned();
  views_count_ = 0;
  if (tv.state_ == State::contiguous) {
    ShareData(*tv.tl_);
  } else {
    state_ = State::noncontiguous;
    tl_->Reset();
    int batch_size = tv.num_samples();
    for (int i = 0; i < batch_size; i++) {
      resize_tensors(batch_size);
      tensors_[i].ShareData(tv.tensors_[i]);
    }
  }
}


template <typename Backend>
TensorVector<Backend> &TensorVector<Backend>::operator=(TensorVector<Backend> &&other) noexcept {
  if (&other != this) {
    state_ = other.state_;
    pinned_ = other.pinned_;
    curr_tensors_size_ = other.curr_tensors_size_;
    tl_ = std::move(other.tl_);
    type_ = other.type_;
    views_count_ = other.views_count_.load();
    tensors_ = std::move(other.tensors_);
    // for (auto &t : tensors_) {
    //   if (t) {
    //     if (auto *del = std::get_deleter<ViewRefDeleter>(t.data_)) del->ref = &views_count_;
    //   }
    // }

    other.views_count_ = 0;
    other.curr_tensors_size_ = 0;
    other.tensors_.clear();
  }
  return *this;
}


template <typename Backend>
void TensorVector<Backend>::UpdateViews() {
  // Return if we do not have a valid allocation
  if (!IsValidType(tl_->type())) return;
  // we need to be able to share empty view as well so don't check if tl_ has any data
  type_ = tl_->type_info();

  assert(curr_tensors_size_ == tl_->num_samples());

  views_count_ = curr_tensors_size_;
  for (size_t i = 0; i < curr_tensors_size_; i++) {
    update_view(i);
  }
}


template <typename Backend>
std::shared_ptr<TensorList<Backend>> TensorVector<Backend>::AsTensorList(bool check_contiguity) {
  DALI_ENFORCE(IsContiguous() || !check_contiguity,
               "Cannot cast non continuous TensorVector to TensorList.");
  // Update the metadata when we are exposing the TensorList to the outside, as it might have been
  // kept in the individual tensors
  for (size_t idx = 0; idx < curr_tensors_size_; idx++) {
    tl_->SetMeta(idx, tensors_[idx].GetMeta());
  }
  return tl_;
}

template <typename Backend>
void TensorVector<Backend>::update_sample_dim(int sample_dim) {
  if (sample_dim_ == sample_dim) {
    return;
  }
  DALI_ENFORCE(sample_dim >= 0, "The dimensionality must be known.");

  sample_dim_ = sample_dim;
  shape_.resize(shape_.num_samples(), sample_dim_);
  for (int i = 0; i < shape_.num_samples(); i++) {
    for (auto &elem : shape_.tensor_shape_span(i)) {
      elem = 0;
    }
  }

}

template <typename Backend>
void TensorVector<Backend>::resize_tensors(int batch_size) {
  resize_tensors(batch_size, sample_dim_);
}

template <typename Backend>
void TensorVector<Backend>::resize_tensors(int batch_size, int sample_dim) {
  // DALI_ENFORCE(sample_dim >= 0, "To insert new samples, the dimensionality must be known.");
  if (static_cast<size_t>(batch_size) > tensors_.size()) {
    auto old_size = curr_tensors_size_;
    tensors_.resize(batch_size);
    dali_meta_.resize(batch_size);
    shape_.resize(batch_size);
    update_sample_dim(sample_dim);
    // for (int i = old_size; i < batch_size; i++) {
    //   // if (!tensors_[i]) {
    //   //   tensors_[i] = std::make_shared<Tensor<Backend>>();  // todo create empty buffer?
    //   //   // tensors_[i]->set_pinned(is_pinned());
    //   // }
    // }
  } else if (static_cast<size_t>(batch_size) < curr_tensors_size_) {
    tensors_.resize(batch_size);
    dali_meta_.resize(batch_size);
    shape_.resize(batch_size);
    update_sample_dim(sample_dim);
    // for (size_t i = new_size; i < curr_tensors_size_; i++) {
    //   if (tensors_[i]->shares_data()) {
    //     tensors_[i]->Reset();
    //   }
    // }
  } else {
    update_sample_dim(sample_dim);
  }
  // curr_tensors_size_ = new_size;
}


template <typename Backend>
void TensorVector<Backend>::update_view(int idx) {
  assert(static_cast<size_t>(idx) < curr_tensors_size_);
  assert(static_cast<size_t>(idx) < tl_->num_samples());

  auto *ptr = tl_->raw_mutable_tensor(idx);

  TensorShape<> shape = tl_->tensor_shape(idx);

  tensors_[idx].Reset();
  // TODO(klecki): deleter that reduces views_count or just noop sharing?
  // tensors_[i]->ShareData(tl_.get(), static_cast<int>(idx));
  if (tensors_[idx].raw_data() != ptr || tensors_[idx].shape() != shape) {
    tensors_[idx].ShareData(unsafe_sample_owner(*tl_, idx),
                             volume(tl_->tensor_shape(idx)) * tl_->type_info().size(),
                             tl_->is_pinned(),
                             shape, tl_->type(),
                             order());
  } else if (IsValidType(tl_->type())) {
    tensors_[idx].set_type(tl_->type());
  }
  tensors_[idx].SetMeta(tl_->GetMeta(idx));
}


template class DLL_PUBLIC TensorVector<CPUBackend>;
template class DLL_PUBLIC TensorVector<GPUBackend>;
template void TensorVector<CPUBackend>::Copy<CPUBackend>(const TensorVector<CPUBackend>&, AccessOrder);  // NOLINT
template void TensorVector<CPUBackend>::Copy<GPUBackend>(const TensorVector<GPUBackend>&, AccessOrder);  // NOLINT
template void TensorVector<GPUBackend>::Copy<CPUBackend>(const TensorVector<CPUBackend>&, AccessOrder);  // NOLINT
template void TensorVector<GPUBackend>::Copy<GPUBackend>(const TensorVector<GPUBackend>&, AccessOrder);  // NOLINT
template void TensorVector<CPUBackend>::Copy<CPUBackend>(const TensorList<CPUBackend>&, AccessOrder);  // NOLINT
template void TensorVector<CPUBackend>::Copy<GPUBackend>(const TensorList<GPUBackend>&, AccessOrder);  // NOLINT
template void TensorVector<GPUBackend>::Copy<CPUBackend>(const TensorList<CPUBackend>&, AccessOrder);  // NOLINT
template void TensorVector<GPUBackend>::Copy<GPUBackend>(const TensorList<GPUBackend>&, AccessOrder);  // NOLINT

}  // namespace dali
