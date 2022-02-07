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
#include <memory>
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

bool same_owner(const std::weak_ptr<void> &x, const std::shared_ptr<void> &y) {
    if (x.owner_before(y) || y.owner_before(x))
        return false;
    return true;
}

template <typename Backend>
TensorVector<Backend>::TensorVector() = default;


template <typename Backend>
TensorVector<Backend>::TensorVector(int batch_size) {

  SetContiguous(false);
  SetSize(batch_size, 1);
  shape_ = uniform_list_shape(batch_size, TensorShape<>{0});
  check_consistency();
}


template <typename Backend>
TensorVector<Backend>::TensorVector(std::shared_ptr<TensorList<Backend>> tl) {
  // assert(tl_ && "Construction with null TensorList is illegal");
  pinned_ = tl->is_pinned();
  type_ = tl->type_info();
  shape_ = tl->shape();
  sample_dim_ = tl->sample_dim();
  SetContiguous(true);
  contiguous_buffer_.set_backing_allocation(unsafe_sample_owner(*tl, 0), tl->nbytes(),
                                            tl->is_pinned(), tl->type(), shape_.num_elements());
  resize_tensors(tl->num_samples());
  UpdateViews();
  check_consistency();
}


template <typename Backend>
TensorVector<Backend>::TensorVector(TensorVector<Backend> &&other) noexcept {
  state_ = other.state_;
  pinned_ = other.pinned_;
  contiguous_buffer_ = std::move(other.contiguous_buffer_);
  type_ = std::move(other.type_);
  tensors_ = std::move(other.tensors_);
  dali_meta_ = std::move(other.dali_meta_);
  shape_ = std::move(other.shape_);
  sample_dim_ = other.sample_dim_;
  // for (auto &t : tensors_) {
  //   if (t) {
  //     if (auto *del = std::get_deleter<ViewRefDeleter>(t->data_)) del->ref = &views_count_;
  //   }
  // }

  other.contiguous_buffer_.reset();
  other.tensors_.clear();
  other.Reset();
  check_consistency();
}

template <typename Backend>
bool TensorVector<Backend>::has_data() const {
  return has_data_;
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
  SetContiguous(false);

  tensors_[dst].ShareData(owner.tensors_[src]);
  check_consistency();
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

  // if (tensors_[dst].shape().num_elements() != owner.shape().num_elements()) {
  SetContiguous(false);
  // }
  tensors_[dst].ShareData(owner);
  // todo v update shape
  // shape().set_tensor_shape(idx, owner.shape());
  check_consistency();
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
  check_consistency();
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
    bool need_sync = contiguous_buffer_.has_data();
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
  contiguous_buffer_.set_order(order, false);
  for (auto &t : tensors_)
    t.set_order(order, false);
  order_ = order;
  check_consistency();
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
  has_data_ = false;


  if (state_ == State::contiguous) { // todo or policy == Coalesce
    // even if we don't have anything here, we don't care, and still use it to allocate
    int64_t num_samples = new_shape.num_samples(), new_size = new_shape.num_elements();
    propagate_properties_to_contiguous();
    contiguous_buffer_.resize(new_size, new_type);
    order_ = contiguous_buffer_.order();  // propagate order after allocation
    uint8_t *base_ptr = static_cast<uint8_t*>(contiguous_buffer_.raw_mutable_data());
    for (int64_t i = 0; i < num_samples; i++) {
      auto tensor_size = volume(new_shape.tensor_shape_span(i));

      std::shared_ptr<void> sample_alias(contiguous_buffer_.get_data_ptr(), base_ptr);
      // todo, convert this to buffers
      tensors_[i].ShareData(sample_alias, tensor_size * type_.size(), pinned_,
                            new_shape[i], new_type, order());
      base_ptr += tensor_size * type_.size();
    }
    has_data_ = contiguous_buffer_.has_data();
    return;

    // This is corresponding part from TL:
    // // Calculate the new size
    // Index num_samples = new_shape.size(), new_size = 0;
    // offsets_.resize(num_samples);
    // for (Index i = 0; i < num_samples; ++i) {
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

    // meta_.resize(num_samples, DALIMeta(layout_));
  }


  // if (state_ == State::contiguous) {
  //   tl_->Resize(new_shape, new_type);
  //   UpdateViews();
  //   return;
  // }

  SetContiguous(false);
  propagate_properties();
  for (int64_t i = 0; i < shape_.num_samples(); i++) {
    // TODO: test this scenario? - we should not be able to get here
    // problem: we have a case, where we were first contiguous, now we want non-contiguous.
    // so we should probably set the buffers as not sharing data.
    if (tensors_[i].get_data_ptr() && same_owner(buffer_bkp_, tensors_[i].get_data_ptr())) {
      // if we have same owner as contiguous buffer, we can assume we share into that contiguous
      // buffer, se we need to break this share, to be able to resize sample-wise.
      // todo, convert to regular assert
      DALI_ENFORCE(tensors_[i].shares_data());
      tensors_[i].Reset();
      propagate_properties_to_samples(i);
    }
    tensors_[i].Resize(new_shape[i], new_type);
    // can we have different order?
    has_data_ = has_data_ || tensors_[i].has_data();
  }
  order_ = tensors_[0].order();  // propagate order after allocation
  buffer_bkp_.reset();
  check_consistency();
}


template <typename Backend>
void TensorVector<Backend>::SetSize(int batch_size) {
  SetSize(batch_size, sample_dim_);
  // resize_tensors(batch_size);
  check_consistency();
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

  // if (sample_dim == -1) {
  //   // We didn't get new sample dim, and we currently don't have one.
  //   // Just expand the metadata structures
  //   tensors_.resize(batch_size);
  //   shape_.resize(batch_size);
  //   dali_meta_.resize(batch_size);
  //   return; // ????
  // } else {


  // }


  resize_tensors(batch_size, sample_dim);
  check_consistency();
}




template <typename Backend>
void TensorVector<Backend>::set_type(DALIDataType new_type_id) {
  DALI_ENFORCE(new_type_id != DALI_NO_TYPE, "new_type_id must represent a valid type.");
  if (type_.id() == new_type_id)
    return;
  type_ = TypeTable::GetTypeInfo(new_type_id);
  propagate_properties();
  check_consistency();
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
  layout_ = layout;

  propagate_properties();
  check_consistency();
}


template <typename Backend>
TensorLayout TensorVector<Backend>::GetLayout() const {
  return layout_;
}


template <typename Backend>
const DALIMeta &TensorVector<Backend>::GetMeta(int idx) const {
  // assert(static_cast<size_t>(idx) < curr_tensors_size_);
  return tensors_[idx].GetMeta();
}


template <typename Backend>
void TensorVector<Backend>::SetMeta(int idx, const DALIMeta &meta) {
  // assert(static_cast<size_t>(idx) < curr_tensors_size_);
  tensors_[idx].SetMeta(meta);
}


template <typename Backend>
void TensorVector<Backend>::set_pinned(bool pinned) {
  // Store the value, in case we pin empty vector and later call Resize
  DALI_ENFORCE(!has_data());
  pinned_ = pinned;
  propagate_properties();
  check_consistency();
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
  contiguous_buffer_.reserve(total_bytes);
  order_ = contiguous_buffer_.order();  // propagate order after allocation
  // TODO: the reserve doesn't work with weak ptr
  // tl_->reserve(total_bytes);
  // UpdateViews();
}


template <typename Backend>
void TensorVector<Backend>::reserve(size_t bytes_per_sample, int batch_size) {
  assert(batch_size > 0);
  // TODO: consider keeping a counter of actual number of elements as now, to keep the allocations
  // OTOH it leaks memory
  state_ = State::noncontiguous;
  resize_tensors(batch_size);
  propagate_properties();
  for (int64_t i = 0; i < shape_.num_samples(); i++) {
    tensors_[i].reserve(bytes_per_sample);
  }
  order_ = tensors_[0].order();  // propagate order after allocation
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
    // We clear the contiguous_buffer_, as we are now non-contiguous.
    buffer_bkp_ = contiguous_buffer_.get_data_ptr();
    contiguous_buffer_.reset();
  }
  // TODO: get rid of weak buffer, make it free stuff when we switch to non contiguous
  // check_consistency();
}


template <typename Backend>
void TensorVector<Backend>::Reset() {
  contiguous_buffer_.reset();
  tensors_.clear();
  dali_meta_.clear();
  type_ = {};
  sample_dim_ = -1;
  shape_ = {};
  has_data_ = false;
  check_consistency();
}


template <typename Backend>
template <typename SrcBackend>
void TensorVector<Backend>::Copy(const TensorList<SrcBackend> &in_tl, AccessOrder order) {
  type_ = in_tl.type_info();
  SetContiguous(true); // this resets the buffers as needed
  sample_dim_ = in_tl.sample_dim();
  shape_ = in_tl.shape();
  layout_ = in_tl.GetLayout();
  pinned_ = in_tl.is_pinned();
  has_data_ = in_tl.has_data();


  if (!order)
    order = in_tl.order() ? in_tl.order() : this->order();
  order.wait(this->order());

  propagate_properties();
  TensorList<Backend> tmp;
  contiguous_buffer_.resize(shape().num_elements(), type());
  order_ = contiguous_buffer_.order();  // propagate order after allocation

  tmp.ShareData(contiguous_buffer_.get_data_ptr(), contiguous_buffer_.nbytes(),
                is_pinned(), shape(), type(), order_);
  tmp.Copy(in_tl, order);
  resize_tensors(shape_.num_samples());
  UpdateViews();
  this->order().wait(order);
  // type_ = in_tl.type_info();
  // tl_->Copy(in_tl, order);

  // resize_tensors(tl_->num_samples());
  // UpdateViews();
  check_consistency();
}


template <typename Backend>
template <typename SrcBackend>
void TensorVector<Backend>::Copy(const TensorVector<SrcBackend> &in_tv, AccessOrder order) {
  type_ = in_tv.type_info();
  SetContiguous(true); // this resets the buffers as needed
  sample_dim_ = in_tv.sample_dim();
  shape_ = in_tv.shape();
  layout_ = in_tv.GetLayout();
  pinned_ = in_tv.is_pinned();
  has_data_ = in_tv.has_data();


  if (!order)
    order = in_tv.order() ? in_tv.order() : this->order();
  order.wait(this->order());

  propagate_properties();
  TensorList<Backend> tmp;
  contiguous_buffer_.resize(shape().num_elements(), type());
  order_ = contiguous_buffer_.order();  // propagate order after allocation

  tmp.ShareData(contiguous_buffer_.get_data_ptr(), contiguous_buffer_.nbytes(),
                is_pinned(), shape(), type(), order_);
  tmp.Copy(in_tv, order);
  resize_tensors(shape_.num_samples());
  UpdateViews();
  this->order().wait(order);
  // SetContiguous(true);
  // type_ = in_tv.type_;
  // tl_->Copy(in_tv, order);

  // resize_tensors(tl_->num_samples());
  // UpdateViews();
  check_consistency();
}


template <typename Backend>
void TensorVector<Backend>::ShareData(const TensorList<Backend> &in_tl) {
  type_ = in_tl.type_info();
  SetContiguous(true); // this resets the buffers as needed
  shape_ = in_tl.shape();
  sample_dim_ = in_tl.sample_dim();
  order_ = in_tl.order();
  layout_ = in_tl.GetLayout();
  pinned_ = in_tl.is_pinned();
  resize_tensors(in_tl.num_samples());
  // contiguous_buffer_.set_backing_allocation(unsafe(in_tl, size_t bytes, bool pinned)

  // todo fixme: assumes contiguous
  // todo, set_backing_allocation needs direct order information?
  contiguous_buffer_.reset(order_);
  contiguous_buffer_.set_order(order_);
  contiguous_buffer_.set_backing_allocation(
      unsafe_sample_owner(const_cast<TensorList<Backend> &>(in_tl), 0), in_tl.nbytes(), pinned_,
      type(), shape_.num_elements());


  has_data_ = in_tl.has_data();

  // Alternative: just dummy samples
  // int batch_size = in_tl.num_samples();
  // for (int i = 0; i < batch_size; i++) {
  //   tensors_[i].ShareData(unsafe_sample_owner(in_tl, i), volume(shape_[i]) * type_.size(),
  //                         is_pinned(), shape_[i], type(), order());
  // }

  UpdateViews();
  check_consistency();
}

template <typename Backend>
void TensorVector<Backend>::ShareData(const TensorVector<Backend> &tv) {
  type_ = tv.type_;
  SetContiguous(tv.state_ == State::contiguous); // this resets the buffers as needed
  shape_ = tv.shape_;
  sample_dim_ = tv.sample_dim_;
  order_ = tv.order_;
  layout_ = tv.layout_;
  pinned_ = tv.is_pinned();
  has_data_ = tv.has_data();
  resize_tensors(shape_.num_samples()); // update internal structures, no need to adjust dim in shape
  if (tv.state_ == State::contiguous) {
    contiguous_buffer_.ShareData(tv.contiguous_buffer_);
    UpdateViews();
  } else {
    int batch_size = tv.num_samples();
    for (int i = 0; i < batch_size; i++) {
      tensors_[i].ShareData(tv.tensors_[i]);
    }
  }
  check_consistency();
}


template <typename Backend>
TensorVector<Backend> &TensorVector<Backend>::operator=(TensorVector<Backend> &&other) noexcept {
  if (&other != this) {
    has_data_ = other.has_data();
    tensors_ = std::move(other.tensors_);
    dali_meta_ = std::move(other.dali_meta_);
    contiguous_buffer_ = std::move(other.contiguous_buffer_);
    buffer_bkp_ = std::move(other.buffer_bkp_);
    state_ = other.state_;
    pinned_ = other.pinned_;
    type_ = other.type_;
    order_ = other.order_;
    sample_dim_ = other.sample_dim_;
    shape_ = std::move(other.shape_);
    layout_ = other.layout_;

    other.tensors_.clear();
    other.dali_meta_.clear();
    other.Reset();
  }
  check_consistency();
  return *this;
}


template <typename Backend>
void TensorVector<Backend>::UpdateViews() {
  // Return if we do not have a valid allocation
  if (!has_data()) {
    return;
  }
  if (!IsValidType(type())) {
    return;
  }
  // Return if we are already non-contiguous, no need to update
  if (state_ == State::noncontiguous) {
    return;
  }

  int64_t num_samples = shape_.num_samples();
  uint8_t *base_ptr = static_cast<uint8_t*>(contiguous_buffer_.raw_mutable_data());
  for (int64_t i = 0; i < num_samples; i++) {
    auto tensor_size = volume(shape_.tensor_shape_span(i));

    std::shared_ptr<void> sample_alias(contiguous_buffer_.get_data_ptr(), base_ptr);
    // todo, convert this to buffers
    tensors_[i].ShareData(sample_alias, tensor_size * type_.size(), is_pinned(),
                          shape_[i], type(), order());
    base_ptr += tensor_size * type_.size();
  }
  // check_consistency();
}


template <typename Backend>
std::shared_ptr<TensorList<Backend>> TensorVector<Backend>::AsTensorList(bool check_contiguity) {
  DALI_ENFORCE(IsContiguous() || !check_contiguity,
               "Cannot cast non continuous TensorVector to TensorList.");
  // Update the metadata when we are exposing the TensorList to the outside, as it might have been
  // kept in the individual tensors
  // TODO FIXME
  // for (size_t idx = 0; idx < curr_tensors_size_; idx++) {
  //   tl_->SetMeta(idx, tensors_[idx].GetMeta());
  // }
  auto result = std::make_shared<TensorList<Backend>>();
  result->ShareData(contiguous_buffer_.get_data_ptr(), contiguous_buffer_.nbytes(),
                    is_pinned(), shape(), type(), order());
  return result;
}



template <typename Backend>
void TensorVector<Backend>::resize_tensors(int batch_size) {
  resize_tensors(batch_size, sample_dim_);
}

template <typename Backend>
void TensorVector<Backend>::resize_tensors(int batch_size, int sample_dim) {
  // DALI_ENFORCE(sample_dim >= 0, "To insert new samples, the dimensionality must be known.");
  if (static_cast<size_t>(batch_size) != tensors_.size()) {
    tensors_.resize(batch_size);
    dali_meta_.resize(batch_size);
    shape_.resize(batch_size);
  }
  update_sample_dim(sample_dim);
}

template <typename Backend>
void TensorVector<Backend>::update_sample_dim(int sample_dim) {
  if (sample_dim_ == sample_dim) {
    return;
  }
  DALI_ENFORCE(sample_dim >= 0, "The dimensionality must be known.");

  sample_dim_ = sample_dim;
  // todo, possibly second resize of sample dim?
  shape_.resize(shape_.num_samples(), sample_dim_);
  for (int i = 0; i < shape_.num_samples(); i++) {
    for (auto &elem : shape_.tensor_shape_span(i)) {
      elem = 0;
    }
  }
  // check_consistency();
}


template <typename Backend>
void TensorVector<Backend>::update_view(int idx) {
  DALI_FAIL("DO NOT USE");
  // assert(static_cast<size_t>(idx) < curr_tensors_size_);
  // assert(static_cast<size_t>(idx) < tl_->num_samples());

  // auto *ptr = tl_->raw_mutable_tensor(idx);

  // TensorShape<> shape = tl_->tensor_shape(idx);

  // tensors_[idx].Reset();
  // if (tensors_[idx].raw_data() != ptr || tensors_[idx].shape() != shape) {
  //   tensors_[idx].ShareData(unsafe_sample_owner(*tl_, idx),
  //                            volume(tl_->tensor_shape(idx)) * tl_->type_info().size(),
  //                            tl_->is_pinned(),
  //                            shape, tl_->type(),
  //                            order());
  // } else if (IsValidType(tl_->type())) {
  //   tensors_[idx].set_type(tl_->type());
  // }
  // tensors_[idx].SetMeta(tl_->GetMeta(idx));
}



template <typename Backend>
void TensorVector<Backend>::propagate_properties() {
  propagate_properties_to_contiguous();
  propagate_properties_to_samples();
  // check_consistency();
}

template <typename Backend>
void TensorVector<Backend>::propagate_properties_to_contiguous() {
  if (!contiguous_buffer_.has_data()) {
    contiguous_buffer_.set_pinned(is_pinned());
  }
  if (IsValidType(type())) {
    contiguous_buffer_.set_type(type());
  }
  contiguous_buffer_.set_order(order());
}

template <typename Backend>
void TensorVector<Backend>::propagate_properties_to_samples() {
  for (int i = 0; i < tensors_.size(); i++) {
    propagate_properties_to_samples(i);
  }
}


template <typename Backend>
void TensorVector<Backend>::propagate_properties_to_samples(int idx) {
  auto &tensor = tensors_[idx];
  if (!tensor.has_data()) {
    tensor.set_pinned(is_pinned());
  }
  if (IsValidType(type())) {
    tensor.set_type(type());
  }
  tensor.set_order(order(), false); // we already synced
  tensor.SetLayout(GetLayout());
  auto &meta = dali_meta_[idx];
  meta.SetLayout(GetLayout());
  // check_consistency();
  // for (int i = 0; i < )
}

template <typename Backend>
void TensorVector<Backend>::check_consistency() {
  if (has_data_) {
    assert(shape_.num_samples() != 0);
    assert(IsValidType(type_));
    assert(sample_dim_ != -1);
    assert(sample_dim_ == shape_.sample_dim());
    assert(shape_.num_samples() == tensors_.size());
    assert(shape_.num_samples() == dali_meta_.size());
    if (state_ == State::contiguous) {
      assert(contiguous_buffer_.has_data() == has_data_);
      assert(pinned_ == contiguous_buffer_.is_pinned());
      assert(type_.id() == contiguous_buffer_.type());
      assert(order_ == contiguous_buffer_.order());
      // assert(shape_[i] = contiguous_buffer_.shape());
    }
    bool has_data_impl = false;
    for (int i = 0; i < shape_.num_samples(); i++) {
      // assert(has_data_ == tensors_[i].has_data());
      has_data_impl = has_data_impl || tensors_[i].has_data();
      assert(pinned_ == tensors_[i].is_pinned());
      assert(type_.id() == tensors_[i].type());
      // assert(order_ == tensors_[i].order());///WHY????

      assert(shape_[i] == tensors_[i].shape());
    }
    assert(has_data_ == has_data_impl);
  }
  // bool has_data_ = false;
  // std::vector<Tensor<Backend>> tensors_;
  // std::vector<DALIMeta> dali_meta_;
  // Buffer<Backend> contiguous_buffer_;
  // std::weak_ptr<void> buffer_bkp_;
  // State state_ = State::noncontiguous;
  // // pinned status and type info should be uniform
  // bool pinned_ = true;
  // TypeInfo type_{};
  // AccessOrder order_;
  // int sample_dim_ = -1;
  // TensorListShape<> shape_{};
  // TensorLayout layout_;
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
