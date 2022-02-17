// Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_PIPELINE_DATA_DYNAMIC_TENSOR_VIEW_H_
#define DALI_PIPELINE_DATA_DYNAMIC_TENSOR_VIEW_H_

#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>
#include "dali/core/tensor_shape.h"
#include "dali/core/tensor_view.h"
#include "dali/pipeline/data/types.h"

namespace dali {

// We do not introduce DynamicTensorListView as the usage compared to TensorVector
// is a bit stretched.

// TODO: do we inherit from DynamicTensorView<void, DynamicDimensions>??

// struct DynamicType {
//   DynamicType() = delete;
// };

template <typename Backend, typename DataType, int ndim = DynamicDimensions>
struct DynamicTensorViewBase {
  static_assert(std::is_same<std::remove_const_t<DataType>, void>::value,
                "The underlying type must either be const or const void");
  using element_type = DataType;
  int dim() const {
    return shape.sample_dim();
  };
  DALIDataType type() const {
    return type_id;
  };

  ptrdiff_t num_elements() const {
    return volume(shape);
  }

  /**
   * @brief Utility to calculate pointer to element at given coordinates
   */
  template <typename... Indices>
  DataType *operator()(int64_t idx0, Indices &&...idx) const {
    return data + CalcOffset(shape, std::array<ptrdiff_t, sizeof...(Indices) + 1>{
                                        idx0, (ptrdiff_t{idx})...});
  }

  /**
   * @brief Utility to calculate pointer to element at given coordinates
   */
  template <typename Offset>
  DataType *operator()(const Offset &pos) const {
    return data + CalcOffset(shape, pos);
  }

  DataType *data = nullptr;
  TensorShape<ndim> shape = {};
  DALIDataType type_id = DALI_NO_TYPE;

 protected:
  DynamicTensorViewBase() = default;
  DynamicTensorViewBase(const DynamicTensorViewBase &) = default;
  DynamicTensorViewBase(DataType *data, const TensorShape<ndim> &shape, DALIDataType type_id)
      : data(data), shape(shape), type_id(type_id) {}
  DynamicTensorViewBase(DataType *data, TensorShape<ndim> &&shape, DALIDataType type_id)
      : data(data), shape(std::move(shape)), type_id(type_id) {}
};

// TODO(klecki): type2id<T>::value instead of TypeTable::GetTypeId<T>?
template <typename Backend, int ndim = DynamicDimensions>
struct DynamicTensorView : DynamicTensorViewBase<Backend, void, ndim> {
  using Base = DynamicTensorViewBase<Backend, void, ndim>;

  DynamicTensorView() = default;

  /**
   * @name Construct the view inferring the type_id from the pointer value.
   */
  // @{
  template <typename T, typename = std::enable_if_t<!std::is_const<T>::value>>
  DynamicTensorView(T *data, const TensorShape<ndim> &shape)
      : Base(data, shape, TypeTable::GetTypeId<T>()) {
  }

  template <typename T, typename = std::enable_if_t<!std::is_const<T>::value>>
  DynamicTensorView(T *data, TensorShape<ndim> &&shape)
      : Base(data, std::move(shape), TypeTable::GetTypeId<T>()) {}

  template <typename T, int other_ndim, typename = std::enable_if_t<!std::is_const<T>::value>>
  DynamicTensorView(T *data, const TensorShape<other_ndim> &shape)
      : Base(data, shape, TypeTable::GetTypeId<T>()) {
    // TODO(klecki): The tensor shape goes through a runtime check for some reason, so we
    // temporarily plug it here before we evaluate TensorShape fix
    detail::check_compatible_ndim<ndim, other_ndim>();
  }

  template <typename T, int other_ndim, typename = std::enable_if_t<!std::is_const<T>::value>>
  DynamicTensorView(T *data, TensorShape<other_ndim> &&shape)
      : Base(data, std::move(shape), TypeTable::GetTypeId<T>()) {
    detail::check_compatible_ndim<ndim, other_ndim>();
  }
  // @}


  /**
   * @name Construct the view with explicitly provided type_id.
   */
  // @{
  DynamicTensorView(void *data, const TensorShape<ndim> &shape, DALIDataType type_id)
      : Base(data, shape, type_id) {}

  DynamicTensorView(void *data, TensorShape<ndim> &&shape, DALIDataType type_id)
      : Base(data, std::move(shape), type_id) {}

  template <int other_ndim>
  DynamicTensorView(void *data, const TensorShape<other_ndim> &shape, DALIDataType type_id)
      : Base(data, shape, type_id) {
    detail::check_compatible_ndim<ndim, other_ndim>();
  }

  template <int other_ndim>
  DynamicTensorView(void *data, TensorShape<other_ndim> &&shape, DALIDataType type_id)
      : Base(data, std::move(shape), type_id) {
    detail::check_compatible_ndim<ndim, other_ndim>();
  }
  // @}

  /**
   * @name nullptr overloads with DALI_NO_TYPE
   */
  // @{
  DynamicTensorView(std::nullptr_t *, const TensorShape<ndim> &shape)
      : Base(nullptr, shape, DALI_NO_TYPE) {}

  DynamicTensorView(std::nullptr_t *, TensorShape<ndim> &&shape)
      : Base(nullptr, std::move(shape), DALI_NO_TYPE) {}

  template <int other_ndim>
  DynamicTensorView(std::nullptr_t *, const TensorShape<other_ndim> &shape)
      : Base(nullptr, shape, DALI_NO_TYPE) {
    detail::check_compatible_ndim<ndim, other_ndim>();
  }

  template <int other_ndim>
  DynamicTensorView(std::nullptr_t *, TensorShape<other_ndim> &&shape)
      : Base(nullptr, std::move(shape), DALI_NO_TYPE) {
    detail::check_compatible_ndim<ndim, other_ndim>();
  }
  // @}


  DynamicTensorView(const DynamicTensorView &) = default;
  DynamicTensorView &operator=(const DynamicTensorView &) = default;

  DynamicTensorView(const DynamicTensorView &&other) {
    this->data = other.data;
    other.data = nullptr;
    this->shape = std::move(other.shape);
    this->type_id = other.type_id;
    other.type_id = DALI_NO_TYPE;
  };

  DynamicTensorView &operator=(const DynamicTensorView &&other) {
    this->data = other.data;
    other.data = nullptr;
    this->shape = std::move(other.shape);
    this->type_id = other.type_id;
    other.type_id = DALI_NO_TYPE;
    return *this;
  }

  // TODO: Dynamic with other_ndim

  /**
   * @name Converters from static TensorView
   *
   * We keep the `ndim` and the `other_ndim` variants to allow for better overload resolution
   * with in-place construction.
   */
  // @{
  template <typename T, typename = std::enable_if_t<!std::is_const<T>::value>>
  DynamicTensorView(const TensorView<Backend, T, ndim> &other) {
    this->data = other.data;
    this->shape = other.shape;
    this->type_id = TypeTable::GetTypeId<T>();
  }

  template <typename T, typename = std::enable_if_t<!std::is_const<T>::value>>
  DynamicTensorView(TensorView<Backend, T, ndim> &&other) {
    this->data = other.data;
    other.data = nullptr;
    this->shape = std::move(other.shape);
    this->type_id = TypeTable::GetTypeId<T>();
  }

  template <typename T, int other_ndim, typename = std::enable_if_t<!std::is_const<T>::value>>
  DynamicTensorView(const TensorView<Backend, T, other_ndim> &other) {
    detail::check_compatible_ndim<ndim, other_ndim>();
    this->data = other.data;
    this->shape = other.shape;
    this->type_id = TypeTable::GetTypeId<T>();
  }

  template <typename T, int other_ndim, typename = std::enable_if_t<!std::is_const<T>::value>>
  DynamicTensorView(TensorView<Backend, T, other_ndim> &&other) {
    detail::check_compatible_ndim<ndim, other_ndim>();
    this->data = other.data;
    other.data = nullptr;
    this->shape = std::move(other.shape);
    this->type_id = TypeTable::GetTypeId<T>();
  }

  template <typename T, typename = std::enable_if_t<!std::is_const<T>::value>>
  DynamicTensorView &operator=(const TensorView<Backend, T, ndim> &other) {
    this->data = other.data;
    this->shape = other.shape;
    this->type_id = TypeTable::GetTypeId<T>();
    return *this;
  }

  template <typename T, typename = std::enable_if_t<!std::is_const<T>::value>>
  DynamicTensorView &operator=(TensorView<Backend, T, ndim> &&other) {
    this->data = other.data;
    other.data = nullptr;
    this->shape = std::move(other.shape);
    this->type_id = TypeTable::GetTypeId<T>();
    return *this;
  }

  template <typename T, int other_ndim, typename = std::enable_if_t<!std::is_const<T>::value>>
  DynamicTensorView &operator=(const TensorView<Backend, T, other_ndim> &other) {
    detail::check_compatible_ndim<ndim, other_ndim>();
    this->data = other.data;
    this->shape = other.shape;
    this->type_id = TypeTable::GetTypeId<T>();
    return *this;
  }

  template <typename T, int other_ndim, typename = std::enable_if_t<!std::is_const<T>::value>>
  DynamicTensorView &operator=(TensorView<Backend, T, other_ndim> &&other) {
    detail::check_compatible_ndim<ndim, other_ndim>();
    this->data = other.data;
    other.data = nullptr;
    this->shape = std::move(other.shape);
    this->type_id = TypeTable::GetTypeId<T>();
    return *this;
  }
  // @}


  /**
   * @name Explicitly delted constructors disallowing passing a pointer to const.
   *
   * Listing all the variants here, and blocking others with SFINAE allows the compiler
   * to state that such constructor is deleted rather than trying to instantiate the type2id
   * trait with const type and failing miserably with supper long message.
   *
   * If you see any of the constructors below being used, it means that you tired to pass pointer to
   * const to a non-const view container, which is not allowed.
   */
  // @{
  template <typename T>
  DynamicTensorView(const T *data, const TensorShape<ndim> &shape) = delete;
  // template <typename T>
  // DynamicTensorView(const T *data, TensorShape<ndim> &&shape) = delete;
  template <typename T>
  DynamicTensorView(const T *data, const TensorShape<ndim> &shape, DALIDataType type_id) = delete;
  // template <typename T>
  // DynamicTensorView(const T *data, TensorShape<ndim> &&shape, DALIDataType type_id) = delete;
  template <typename T, int other_ndim>
  DynamicTensorView(const T *data, const TensorShape<other_ndim> &shape) = delete;
  // template <typename T, int other_ndim>
  // DynamicTensorView(const T *data, TensorShape<other_ndim> &&shape) = delete;
  template <typename T, int other_ndim>
  DynamicTensorView(const T *data, const TensorShape<other_ndim> &shape,
                    DALIDataType type_id) = delete;
  // template <typename T, int other_ndim>
  // DynamicTensorView(const T *data, TensorShape<other_ndim> &&shape, DALIDataType type_id) =
  // delete;
  template <typename T>
  DynamicTensorView(const TensorView<Backend, const T, ndim> &other) = delete;
  template <typename T, int other_ndim>
  DynamicTensorView(const TensorView<Backend, const T, other_ndim> &other) = delete;
  // @}


  // TODO: coversions to static_type
  // template <typename DataType, int other_ndim = DynamicDimensions>
  // DynamicTensorView<Backend, DataType, other_ndim> to_static() {
  //   DALI_ENFORCE(type() == type2id<std::remove_cv_t<DataType>>::value,
  //                "Type must match for the conversion");
  //   DALI_ENFORCE(shape.sample_dim() == other_ndim || shape.static_ndim == DynamicDimensions,
  //                "Dimensionality must match for the conversion");
  //   return {static_cast<DataType *>(data), shape};
  // }
};

template <typename Backend, int ndim = DynamicDimensions>
struct ConstDynamicTensorView : DynamicTensorViewBase<Backend, const void, ndim> {
  using Base = DynamicTensorViewBase<Backend, const void, ndim>;

  ConstDynamicTensorView() = default;

  template <typename T, int other_ndim>
  ConstDynamicTensorView(T *data, const TensorShape<other_ndim> &shape)
      : Base(data, shape, type2id<T>::value) {
    // static_assert(!std::is_const<T>::value, ""); // This will probably not compile either way
  }
  template <typename T, int other_ndim>
  ConstDynamicTensorView(T *data, TensorShape<other_ndim> &&shape)
      : Base(data, std::move(shape), type2id<T>::value) {}

  template <int other_ndim>
  ConstDynamicTensorView(void *data, const TensorShape<other_ndim> &shape, DALIDataType type_id)
      : Base(data, shape, type_id) {}
  template <int other_ndim>
  ConstDynamicTensorView(void *data, TensorShape<other_ndim> &&shape, DALIDataType type_id)
      : Base(data, std::move(shape), type_id) {}

  ConstDynamicTensorView(const ConstDynamicTensorView &) = default;
  ConstDynamicTensorView &operator=(const ConstDynamicTensorView &) = default;


  // template <typename DataType, int other_ndim = DynamicDimensions>
  // DynamicTensorView<Backend, DataType, other_ndim> to_static() {
  //   DALI_ENFORCE(type() == type2id<std::remove_cv_t<DataType>>::value,
  //                "Type must match for the conversion");
  //   DALI_ENFORCE(shape.sample_dim() == other_ndim || shape.static_ndim == DynamicDimensions,
  //                "Dimensionality must match for the conversion");
  //   return {static_cast<DataType *>(data), shape};
  // }
};


// template <typename Backend>
// struct DynamicTensorView<Backend, const void, DynamicDimensions> {
//   DynamicTensorView() = default;

//   template <typename DataType>
//   DynamicTensorView(const DataType *data, const TensorShape<DynamicDimensions> &shape)
//       : data{data}, shape{shape}, type_id{type2id<DataType>::value} {}
//   template <typename DataType>
//   DynamicTensorView(const DataType *data, TensorShape<DynamicDimensions> &&shape)
//       : data{data}, shape{std::move(shape)}, type_id{type2id<DataType>::value} {}
//   template <typename DataType, int ndim>
//   DynamicTensorView(const DataType *data, const TensorShape<ndim> &shape)
//       : data{data}, shape{shape}, type_id{type2id<DataType>::value} {}
//   template <typename DataType, int ndim>
//   DynamicTensorView(const DataType *data, TensorShape<ndim> &&shape)
//       : data{data}, shape{std::move(shape)}, type_id{type2id<DataType>::value} {}

//   DynamicTensorView(const void *data, const TensorShape<DynamicDimensions> &shape, DALIDataType
//   type_id)
//       : data{data}, shape{shape}, type_id{type_id} {}
//   DynamicTensorView(const void *data, TensorShape<DynamicDimensions> &&shape, DALIDataType
//   type_id)
//       : data{data}, shape{std::move(shape)}, type_id{type_id} {}
//   template <int ndim>
//   DynamicTensorView(const void *data, const TensorShape<ndim> &shape, DALIDataType type_id)
//       : data{data}, shape{shape}, type_id{type_id} {}
//   template <int ndim>
//   DynamicTensorView(const void *data, TensorShape<ndim> &&shape, DALIDataType type_id)
//       : data{data}, shape{std::move(shape)}, type_id{type_id} {}

//   DynamicTensorView(const DynamicTensorView &) = default;
//   DynamicTensorView &operator=(const DynamicTensorView &) = default;

//   const void *data = nullptr;
//   TensorShape<DynamicDimensions> shape;
//   DALIDataType type_id = DALI_NO_TYPE;

//   DALIDataType type() const {
//     return type_id;
//   }


//   template <typename DataType, int other_ndim = DynamicDimensions>
//   DynamicTensorView<Backend, DataType, other_ndim> to_static() {
//     DALI_ENFORCE(type() == type2id<std::remove_cv_t<DataType>>::value,
//                  "Type must match for the conversion");
//     DALI_ENFORCE(shape.sample_dim() == other_ndim || shape.static_ndim == DynamicDimensions,
//                  "Dimensionality must match for the conversion");
//     return {static_cast<DataType *>(data), shape};
//   }
// };


}  // namespace dali

#endif  // DALI_PIPELINE_DATA_DYNAMIC_TENSOR_VIEW_H_
