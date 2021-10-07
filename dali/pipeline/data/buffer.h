// Copyright (c) 2017-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_PIPELINE_DATA_BUFFER_H_
#define DALI_PIPELINE_DATA_BUFFER_H_

#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "dali/core/common.h"
#include "dali/core/device_guard.h"
#include "dali/core/error_handling.h"
#include "dali/core/util.h"
#include "dali/pipeline/data/types.h"
#include "dali/core/format.h"

namespace dali {

class GPUBackend;
class CPUBackend;

DLL_PUBLIC shared_ptr<uint8_t> AllocBuffer(size_t bytes, bool pinned, GPUBackend *);
DLL_PUBLIC shared_ptr<uint8_t> AllocBuffer(size_t bytes, bool pinned, CPUBackend *);

/**
 * @brief Indicates, based on environment cues, whether pinned memory allocations should be avoided.
 */
DLL_PUBLIC bool RestrictPinnedMemUsage();

template <typename Backend>
inline shared_ptr<uint8_t> AllocBuffer(size_t bytes, bool pinned) {
  return AllocBuffer(bytes, pinned, static_cast<Backend*>(nullptr));
}

// Helper function to get a string of the data shape
inline string ShapeString(vector<Index> shape) {
  string tmp;
  for (auto& val : shape) {
    tmp += std::to_string(val) + " ";
  }
  return tmp;
}

// NOTE: Data storage types in DALI use delayed allocation, and have a
// small custom type system that allows us to circumvent template
// paramters. This is turn allows the Pipeline to manage all intermediate
// memory, opening the door for optimizations and reducing the work that
// must be done by the user when defining operations.

/**
 * @brief Base class to provide common functionality needed by Pipeline data
 * structures. Not meant for use, does not provide methods for allocating
 * any actual storage. The 'Backend' template parameter dictates where the
 * underlying storage is located (CPU or GPU).
 *
 * Buffers are untyped on construction, and don't receive a valid type until
 * 'set_type' or 'data<T>()' is called on a non-const buffer. Upon receiving
 * a valid type, the underlying storage for the buffer is allocated. The type
 * of the underlying data can change over the lifetime of an object if
 * 'set_type' or 'data<T>()' is called again where the calling type does not
 * match the underlying type on the buffer. In this case, the Buffer swaps its
 * current type, but only re-allocates memory if it does not have enough bytes
 * of allocated storage to store the number of elements in the buffer with the
 * new data type size.
 *
 * Buffers should be used to store POD types only. No construction or
 * destruction is provided, only raw memory allocation.
 *
 *
 * We enable move & copy for Buffer. Buffer should behave as shared_ptr + some metadata.
 * Details about who now is the owner of allocation and who shares the memory might get fuzzy
 * and some may go out of the use in that case.
 *
 * As you can see the Buffer uses snake_case_naming of member functions to indicate that it is
 * internal class and should, in most cases, not be handled directly.
 */
// TODO: consider writing copy and move by hand - shared_ptr should behave in sane way.
template <typename Backend>
class DLL_PUBLIC Buffer {
 public:
  /**
   * @brief Initializes a buffer of size 0.
   */
  inline Buffer() = default;
  virtual ~Buffer() = default;

  /**
   * @brief Returns a typed pointer to the underlying storage. If the
   * buffer has not been allocated because it does not yet have a type,
   * the calling type is taken to be the type of the data and the memory
   * is allocated.
   *
   * If the buffer already has a valid type, and the calling type does
   * not match, the type of the buffer is reset and the underlying
   * storage is re-allocated if the buffer does not currently own
   * enough memory to store the current number of elements with the
   * new data type.
   */
  template <typename T>
  inline T* mutable_data() {
    // Note: Call to 'set_type' will immediately return if the calling
    // type matches the current type of the buffer.
    set_type<T>();
    return static_cast<T*>(data_.get());
  }

  /**
   * @brief Returns a const, typed pointer to the underlying storage.
   * The calling type must match the underlying type of the buffer.
   */
  template <typename T>
  inline const T* data() const {
    // clang-format off
    DALI_ENFORCE(IsValidType(type_),
                 "Buffer has no type, 'mutable_data<T>()' must be called "
                 "on non-const buffer to set valid type for " + type_.name());
    DALI_ENFORCE(type_.id() == TypeTable::GetTypeID<T>(),
                 "Calling type does not match buffer data type: " +
                 TypeTable::GetTypeName<T>() + " vs " + type_.name());
    // clang-format on
    return static_cast<T*>(data_.get());
  }

  /**
   * @brief Return an un-typed pointer to the underlying storage.
   * A valid type must be set prior to calling this method by calling
   * the non-const version of the method, or calling 'set_type'.
   */
  inline void* raw_mutable_data() {
    // Empty tensor
    if (data_ == nullptr) return nullptr;
    DALI_ENFORCE(IsValidType(type_),
                 "Buffer has no type, 'mutable_data<T>()' or 'set_type' must "
                 "be called on non-const buffer to set valid type");
    return static_cast<void*>(data_.get());
  }

  /**
   * @brief Return an const, un-typed pointer to the underlying storage.
   * A valid type must be set prior to calling this method by calling
   * the non-const version of the method, or calling 'set_type'.
   */
  inline const void* raw_data() const {
    // Empty tensor
    if (data_ == nullptr) return nullptr;
    DALI_ENFORCE(IsValidType(type_),
                 "Buffer has no type, 'mutable_data<T>()' or 'set_type' must "
                 "be called on non-const buffer to set valid type");
    return static_cast<void*>(data_.get());
  }

  /**
   * @brief Returns the size in elements of the underlying data
   */
  inline Index size() const {
    return size_;
  }

  /**
   * @brief Returns the size in bytes of the underlying data
   */
  inline size_t nbytes() const {
    // Note: This returns the number of bytes occupied by the current
    // number of elements stored in the buffer. This is not neccessarily
    // the number of bytes of the underlying allocation (num_bytes_)
    return size_ * type_.size();
  }

  /**
   * @brief Returns the real size of the allocation
   */
  inline size_t capacity() const {
    return num_bytes_;
  }

  /**
   * @brief Returns the id of the datatype of the underlying storage.
   */
  inline DALIDataType type() const {
    return type_.id();
  }

  /**
   * @brief Returns the TypeInfo object that keeps track of the
   * datatype of the underlying storage.
   */
  inline const TypeInfo &type_info() const {
    return type_;
  }

  using AllocFunc = std::function<shared_ptr<uint8_t>(size_t)>;

  /**
   * @brief Sets a custom allocation function.
   *
   * Sets a custom allocation function. The allocation function returns
   * a shared pointer with a matching deleter.
   *
   * @remarks Experimental - subject to change
   */
  inline void set_alloc_func(AllocFunc allocate) {
    allocate_ = std::move(allocate);
  }

  /**
   * @brief Returns the current custom allocation function.
   *
   * @return Allocation function. If not set, an empty function object is returned.
   *
   * @remarks Experimental - subject to change
   */
  const AllocFunc &alloc_func() const noexcept {
    return allocate_;
  }

  /**
   * @brief Return true if there was data allocation
   */
  inline bool has_data() const noexcept {
    return !!data_;
  }

  /**
   * @brief Sets the type of allocation (pinned/non-pinned) for CPU buffers
   */
  inline void set_pinned(bool pinned) {
    DALI_ENFORCE(!has_data(), "Can only set allocation mode before first allocation");
    DALI_ENFORCE(!allocate_, "Cannot set allocation mode when a custom allocator is used.");
    pinned_ = pinned;
  }

  inline bool is_pinned() const {
    return pinned_;
  }

  /**
   * @brief Returns a device this buffer was allocated on
   * If the backend is CPUBackend, return -1
   */
  int device_id() const {
    return device_;
  }

  /**
   * @brief Sets a device this buffer was allocated on
   * If the backend is CPUBackend, should be -1
   */
  void set_device_id(int device) {
    device_ = device;
  }

  // TODO(): write some tests for Buffer
  /**
   * @brief Check if the buffer needs to be reallocated to handle the requested number of elements
   * and type of data.
   *
   * This function can return true even when the allocation is big enough. This is needed to handle
   * shrinking of big buffers in accordance with shrink threshold.
   *
   * Empty resize request is a special case that keeps the memory - use reset if you want to release
   * the buffer.
   *
   * @param new_elements Number of elements of given type in the requested allocation.
   * @param new_type_id  Type id for the requested allocation.
   */
  inline bool is_reallocation(int64_t new_elements, DALIDataType new_type_id) const {
    // Currently DALI does not shrink when the requested allocation results in empty buffer
    if (new_type_id == DALI_NO_TYPE || new_elements == 0) {
      return false;
    }

    const auto &new_type = new_type_id == type_.id() ? type_ : TypeTable::GetTypeInfo(new_type_id);
    size_t new_num_bytes = new_alloc_size(new_elements, new_type_id);

    // We need to reallocate when:
    // * there is not enough memory - with shares_data_ it indicates an error,
    // * the requested allocation is so small that we should shrink our buffer to it.
    //
    // In the second case we never shrink if the data is pinned allocation or the buffer was
    // shared with us (provided as external memory allocation - we can't request new buffer).
    if (new_num_bytes > num_bytes_) {
      return true;
    } else if (!shares_data_ && !is_pinned() && new_num_bytes < num_bytes_ * shrink_threshold_) {
      return true;
    }
    return false;
  }


  /**
   * @brief Sets the type of the buffer. If the buffer has not been
   * allocated because it does not yet have a type, the calling type
   * is taken to be the type of the data and the memory is allocated.
   *
   * If the buffer already has a valid type, and the calling type does
   * not match, the type of the buffer is reset and the underlying
   * storage is re-allocated if the buffer does not currently own
   * enough memory to store the current number of elements with the
   * new data type.
   *
   * @param new_type_id  Type id for the requested allocation.
   */
  inline void set_type(const DALIDataType new_type_id) {
    // DALI_ENFORCE(new_type_id != DALI_NO_TYPE, "new_type must be valid type.");
    if (new_type_id == type_.id()) return;

    resize(size_, new_type_id);
  }

  template <typename T>
  inline void set_type() {
    set_type(TypeTable::GetTypeID<T>());
  }

  /**
   * @brief Resize the buffer, adjusting the allocation size according to the shrink and growth
   * factors.
   *
   * If the `shares_data() == true` indicating that the external allocation was set for this buffer,
   * it won't be reallocated, ignoring the shrink threshold and raising error if too big number
   * of elements is set.
   *
   * @param new_elements Number of elements in the requested allocation.
   */
  inline void resize(int64_t new_elements) {
    resize(new_elements, type_.id());
  }

  /**
   * @brief Resize and set the type of the buffer, adjusting the allocation size according to the
   * shrink and growth factors.
   *
   * If the `shares_data() == true` indicating that the external allocation was set for this buffer,
   * it won't be reallocated, ignoring the shrink threshold and raising error if too big number
   * of elements is set.
   *
   * @param new_elements Number of elements in the requested allocation.
   * @param new_type_id  Type id for the requested allocation.
   */
  inline void resize(int64_t new_elements, DALIDataType new_type_id) {
    if (!is_reallocation(new_elements, new_type_id)) {
      // no reallocation, adjust relevant metadata
      size_ = new_elements;
      if (type_.id() != new_type_id) {
        type_ = TypeTable::GetTypeInfo(new_type_id);
      }
      return;
    }
    // We try to make the buffer bigger while sharing the data - this is prohibited.
    if (shares_data_) {
      DALI_FAIL(
          make_string("Buffer cannot be resized - it is sharing data with external allocation - "
                      "requested size of ",
                      new_elements, " elements of type ", new_type_id,
                      " would require reallocating the current buffer of ",
                      size_," elements of type ", type_.id()));
    }

    // Reallocate and adjust the metadata
    size_t new_num_bytes = new_alloc_size(new_elements, new_type_id);
    if (new_num_bytes > num_bytes_) {
      size_t grow = num_bytes_ * growth_factor_;
      if (grow > new_num_bytes) new_num_bytes = grow;
      reserve(new_num_bytes);
    } else {
      data_.reset();
      num_bytes_ = 0;
      // TODO: Exception safety?
      // num_elements_ = 0;
      // type_ = {};
      reserve(new_num_bytes);
    }
    size_ = new_elements;
    if (type_.id() != new_type_id) {
      type_ = TypeTable::GetTypeInfo(new_type_id);
    }
  }

  inline void reserve(size_t new_num_bytes) {
    if (new_num_bytes <= num_bytes_) return;

    // re-allocating: get the device
    if (std::is_same<Backend, GPUBackend>::value) {
      CUDA_CALL(cudaGetDevice(&device_));
    } else {
      device_ = CPU_ONLY_DEVICE_ID;
    }

    DALI_ENFORCE(!shares_data_,
                 "Cannot reallocate Buffer if it is sharing data. "
                 "Clear the status by `Reset()` first.");
    data_.reset();
    data_ = allocate_ ? allocate_(new_num_bytes)
                      : AllocBuffer<Backend>(new_num_bytes, pinned_);

    num_bytes_ = new_num_bytes;
  }

  inline void SetExternalAllocation(shared_ptr<void> ptr, size_t bytes, size_t num_elements,
                                    DALIDataType type_id = DALI_NO_TYPE) {
    reset();
    type_ = TypeTable::GetTypeInfo(type_id);
    data_ = ptr;
    size_ = num_elements;
    num_bytes_ = bytes;
    shares_data_ = true;
  }

  void reset() {
    type_ = {};
    data_.reset();
    allocate_ = {};
    size_ = 0;
    shares_data_ = false;
    num_bytes_ = 0;
    device_ =  CPU_ONLY_DEVICE_ID;
  }

  /**
   * @brief Returns a bool indicating if the list shares its underlying storage.
   */
  inline bool shares_data() const {
    return shares_data_;
  }

  // here be dragons
  // DISABLE_COPY_MOVE_ASSIGN(Buffer);

  static void SetGrowthFactor(double factor) {
    assert(factor >= 1.0);
    growth_factor_ = factor;
  }
  static void SetShrinkThreshold(double ratio) {
    assert(ratio >= 0 && ratio <= 1);
    shrink_threshold_ = ratio;
  }
  static double GetGrowthFactor() {
    return growth_factor_;
  }
  static double GetShrinkThreshold() {
    return shrink_threshold_;
  }

  static constexpr double kMaxGrowthFactor = 4;

  std::shared_ptr<void> get_data_ptr() {
    return data_;
  }

 protected:
  /**
   * @brief Helper function to get the size in bytes required to handle requested allocation
   *
   * @param new_elements Number of elements of given type in the requested allocation.
   * @param new_type_id  Type id for the requested allocation.
   */
  inline size_t new_alloc_size(int64_t new_elements, DALIDataType new_type_id) const {
    if (new_type_id == DALI_NO_TYPE || new_elements == 0) {
      return 0;
    }
    const auto &new_type = new_type_id == type_.id() ? type_ : TypeTable::GetTypeInfo(new_type_id);
    return new_elements * new_type.size();
  }

 protected:
  void move_buffer(Buffer &&buffer) {
    type_         = std::move(buffer.type_);
    data_         = std::move(buffer.data_);
    allocate_     = std::move(buffer.allocate_);
    size_         = buffer.size_;
    num_bytes_    = buffer.num_bytes_;
    device_       = buffer.device_;
    shares_data_  = buffer.shares_data_;
    pinned_       = buffer.pinned_;

    buffer.reset();
  }

  static double growth_factor_;
  static double shrink_threshold_;

  static bool default_pinned();

  TypeInfo type_ = {};               // Data type of underlying storage
  shared_ptr<void> data_ = nullptr;  // Pointer to underlying storage
  AllocFunc allocate_;               // Custom allocation function
  Index size_ = 0;                   // The number of elements in the buffer
  size_t num_bytes_ = 0;             // To keep track of the true size of the underlying allocation
  int device_ = CPU_ONLY_DEVICE_ID;  // device the buffer was allocated on
  bool shares_data_ = false;         // Whether we aren't using our own allocation
  bool pinned_ = !RestrictPinnedMemUsage();  // Whether the allocation uses pinned memory
};

template <typename Backend>
DLL_PUBLIC double Buffer<Backend>::growth_factor_ = 1.1;

template <typename Backend>
DLL_PUBLIC double Buffer<Backend>::shrink_threshold_ =
  std::is_same<Backend, CPUBackend>::value ? 0.5 : 0;

template <typename Backend>
DLL_PUBLIC constexpr double Buffer<Backend>::kMaxGrowthFactor;


// Macro so we don't have to list these in all
// classes that derive from Buffer
#define USE_BUFFER_MEMBERS()              \
  using Buffer<Backend>::is_reallocation; \
  using Buffer<Backend>::set_type;        \
  using Buffer<Backend>::resize;          \
  using Buffer<Backend>::reset;           \
  using Buffer<Backend>::type_;           \
  using Buffer<Backend>::data_;           \
  using Buffer<Backend>::size_;           \
  using Buffer<Backend>::shares_data_;    \
  using Buffer<Backend>::num_bytes_;      \
  using Buffer<Backend>::device_;         \
  using Buffer<Backend>::pinned_;         \
  using Buffer<Backend>::move_buffer


}  // namespace dali

#endif  // DALI_PIPELINE_DATA_BUFFER_H_
