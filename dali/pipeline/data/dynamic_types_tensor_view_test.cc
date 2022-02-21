// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gtest/gtest.h>
#include <numeric>
#include <utility>

#include "dali/core/tensor_shape.h"
#include "dali/core/tensor_view.h"
#include "dali/pipeline/data/dynamic_tensor_view.h"
#include "dali/pipeline/data/types.h"
#include "dali/pipeline/data/views.h"

namespace dali {

template <typename DynamicTV, int ndim>
void compare(const DynamicTV &dtv, const void *ptr, const TensorShape<ndim> &shape, DALIDataType dtype) {
  EXPECT_EQ(dtv.data, ptr);
  EXPECT_EQ(dtv.shape, shape);
  EXPECT_EQ(dtv.type_id, dtype);
}

template <typename DynamicTV, typename StaticTV>
void compare(const DynamicTV &ltv, const StaticTV &rtv) {
  static_assert(std::is_same<std::remove_const_t<typename DynamicTV::element_type>, void>::value,
                "DynamicTensorView type is expected as left argument");
  static_assert(!std::is_same<std::remove_const_t<typename StaticTV::element_type>, void>::value,
                "Typed TensorView type is expected as right argument");
  EXPECT_EQ(ltv.data, rtv.data);
  EXPECT_EQ(ltv.shape, rtv.shape);
  EXPECT_EQ(ltv.type_id, TypeTable::GetTypeId<typename StaticTV::element_type>());
}


TEST(DynamicTypesTensorViewTest, DefaultConstructors) {
  TensorView<EmptyBackendTag, DynamicType, 4> empty_static_dim{};
  compare(empty_static_dim, nullptr, TensorShape<4>{}, DALI_NO_TYPE);

  TensorView<EmptyBackendTag, DynamicType> empty_dynamic_dim{};
  compare(empty_dynamic_dim, nullptr, TensorShape<>{}, DALI_NO_TYPE);

  TensorView<EmptyBackendTag, const DynamicType, 4> const_empty_static_dim{};
  compare(const_empty_static_dim, nullptr, TensorShape<4>{}, DALI_NO_TYPE);

  TensorView<EmptyBackendTag, const DynamicType> const_empty_dynamic_dim{};
  compare(const_empty_dynamic_dim, nullptr, TensorShape<>{}, DALI_NO_TYPE);
}

TEST(DynamicTypesTensorViewTest, PtrConstructors) {
  int data = {};
  TensorView<EmptyBackendTag, DynamicType, 4> empty_static_dim{&data, {1, 2, 3, 4}};
  compare(empty_static_dim, &data, TensorShape<4>{1, 2, 3, 4}, DALI_INT32);

  TensorView<EmptyBackendTag, DynamicType, 4> empty_static_dim_conv{&data, TensorShape<>{1, 2, 3, 4}};
  compare(empty_static_dim_conv, &data, TensorShape<4>{1, 2, 3, 4}, DALI_INT32);

  TensorView<EmptyBackendTag, DynamicType> empty_dynamic_dim{&data, {1, 2, 3, 4}};
  compare(empty_dynamic_dim, &data, TensorShape<>{1, 2, 3, 4}, DALI_INT32);

  TensorView<EmptyBackendTag, DynamicType> empty_dynamic_dim_conv{&data, TensorShape<4>{1, 2, 3, 4}};
  compare(empty_dynamic_dim_conv, &data, TensorShape<>{1, 2, 3, 4}, DALI_INT32);
}

TEST(ConstDynamicTypesTensorViewTest, PtrConstructors) {
  int data = {};
  TensorView<EmptyBackendTag, const DynamicType, 4> const_empty_static_dim{&data, {1, 2, 3, 4}};
  compare(const_empty_static_dim, &data, TensorShape<4>{1, 2, 3, 4}, DALI_INT32);

  TensorView<EmptyBackendTag, const DynamicType, 4> const_empty_static_dim_conv{&data, TensorShape<>{1, 2, 3, 4}};
  compare(const_empty_static_dim_conv, &data, TensorShape<4>{1, 2, 3, 4}, DALI_INT32);

  TensorView<EmptyBackendTag, const DynamicType> const_empty_dynamic_dim{&data, {1, 2, 3, 4}};
  compare(const_empty_dynamic_dim, &data, TensorShape<>{1, 2, 3, 4}, DALI_INT32);

  TensorView<EmptyBackendTag, const DynamicType> const_empty_dynamic_dim_conv{&data, TensorShape<4>{1, 2, 3, 4}};
  compare(const_empty_dynamic_dim_conv, &data ,  TensorShape<>{1, 2, 3, 4} , DALI_INT32);
}

TEST(ConstDynamicTypesTensorViewTest, ConstPtrConstructors) {
  const int data = {};
  TensorView<EmptyBackendTag, const DynamicType, 4> const_empty_static_dim{&data, {1, 2, 3, 4}};
  compare(const_empty_static_dim, &data ,  TensorShape<4>{1, 2, 3, 4} , DALI_INT32);

  TensorView<EmptyBackendTag, const DynamicType, 4> const_empty_static_dim_conv{&data, TensorShape<>{1, 2, 3, 4}};
  compare(const_empty_static_dim_conv, &data , TensorShape<4>{1, 2, 3, 4} , DALI_INT32);

  TensorView<EmptyBackendTag, const DynamicType> const_empty_dynamic_dim{&data, {1, 2, 3, 4}};
  compare(const_empty_dynamic_dim, &data ,  TensorShape<>{1, 2, 3, 4} , DALI_INT32);

  TensorView<EmptyBackendTag, const DynamicType> const_empty_dynamic_dim_conv{&data, TensorShape<4>{1, 2, 3, 4}};
  compare(const_empty_dynamic_dim_conv, &data ,  TensorShape<>{1, 2, 3, 4} , DALI_INT32);
}

TEST(DynamicTypesTensorViewTest, NullPtrConstructors) {
  TensorView<EmptyBackendTag, DynamicType, 4> empty_static_dim{nullptr, {1, 2, 3, 4}};
  compare(empty_static_dim, nullptr ,  TensorShape<4>{1, 2, 3, 4} , DALI_NO_TYPE);

  TensorView<EmptyBackendTag, DynamicType> empty_dynamic_dim{nullptr, {1, 2, 3, 4}};
  compare(empty_dynamic_dim, nullptr ,  TensorShape<>{1, 2, 3, 4} , DALI_NO_TYPE);

  TensorView<EmptyBackendTag, const DynamicType, 4> const_empty_static_dim{nullptr, {1, 2, 3, 4}};
  compare(const_empty_static_dim, nullptr ,  TensorShape<4>{1, 2, 3, 4} , DALI_NO_TYPE);

  TensorView<EmptyBackendTag, const DynamicType> const_empty_dynamic_dim{nullptr, {1, 2, 3, 4}};
  compare(const_empty_dynamic_dim, nullptr ,  TensorShape<>{1, 2, 3, 4} , DALI_NO_TYPE);
}

TEST(DynamicTypesTensorViewTest, TypeIdConstructors) {
  TensorView<EmptyBackendTag, DynamicType, 4> empty_static_dim{nullptr, {1, 2, 3, 4}, DALI_INT32};
  compare(empty_static_dim, nullptr ,  TensorShape<4>{1, 2, 3, 4} , DALI_INT32);

  TensorView<EmptyBackendTag, DynamicType> empty_dynamic_dim{nullptr, {1, 2, 3, 4}, DALI_INT32};
  compare(empty_dynamic_dim, nullptr , TensorShape<>{1, 2, 3, 4} , DALI_INT32);

  TensorView<EmptyBackendTag, DynamicType> empty_dynamic_dim_conv{nullptr, TensorShape<4>{1, 2, 3, 4}, DALI_INT32};
  compare(empty_dynamic_dim_conv, nullptr ,  TensorShape<>{1, 2, 3, 4} , DALI_INT32);
}

TEST(DynamicTypesTensorViewTest, ViewConverterConstructors) {
  int data = {};
  TensorView<EmptyBackendTag, int, 3> tv{&data, {1, 2, 3}};
  TensorView<EmptyBackendTag, int> dyn_tv{&data, {1, 2, 3}};

  TensorView<EmptyBackendTag, DynamicType, 3> static_dim{tv}, static_dim_2{dyn_tv};
  compare(static_dim, tv);
  compare(static_dim_2, dyn_tv);

  TensorView<EmptyBackendTag, DynamicType> dynamic_dim{tv}, dynamic_dim_2{dyn_tv};
  compare(dynamic_dim, tv);
  compare(dynamic_dim_2, dyn_tv);

  TensorView<EmptyBackendTag, DynamicType, 3> copy_static_to_static{static_dim};
  compare(copy_static_to_static, tv);

  TensorView<EmptyBackendTag, DynamicType> copy_static_to_dynamic{static_dim};
  compare(copy_static_to_dynamic, tv);


  TensorView<EmptyBackendTag, DynamicType, 3> copy_dynamic_to_static{dynamic_dim};
  compare(copy_dynamic_to_static, tv);

  TensorView<EmptyBackendTag, DynamicType, 3> copy_dynamic_to_dynamic{dynamic_dim};
  compare(copy_dynamic_to_dynamic, tv);
}

TEST(ConstDynamicTypesTensorViewTest, ViewConverterConstructors) {
  const int cdata = {};
  TensorView<EmptyBackendTag, const int, 3> ctv{&cdata, {1, 2, 3}};
  TensorView<EmptyBackendTag, const int> dyn_ctv{&cdata, {1, 2, 3}};

  TensorView<EmptyBackendTag, const DynamicType, 3> static_dim{ctv}, static_dim_2{dyn_ctv};
  compare(static_dim, ctv);
  compare(static_dim_2, dyn_ctv);

  TensorView<EmptyBackendTag, const DynamicType> dynamic_dim{ctv}, dynamic_dim_2{dyn_ctv};
  compare(dynamic_dim, ctv);
  compare(dynamic_dim_2, dyn_ctv);

  int data = {};
  TensorView<EmptyBackendTag, int, 3> tv{&data, {1, 2, 3}};
  TensorView<EmptyBackendTag, int> dyn_tv{&data, {1, 2, 3}};

  TensorView<EmptyBackendTag, const DynamicType, 3> static_dim_nonconst{tv}, static_dim_nonconst_2{dyn_tv};
  compare(static_dim_nonconst, tv);
  compare(static_dim_nonconst_2, dyn_tv);

  TensorView<EmptyBackendTag, const DynamicType> dynamic_dim_nonconst{tv}, dynamic_dim_nonconst_2{dyn_tv};
  compare(dynamic_dim_nonconst, tv);
  compare(dynamic_dim_nonconst_2, dyn_tv);

  TensorView<EmptyBackendTag, const DynamicType, 3> copy_static_to_static{static_dim};
  compare(copy_static_to_static, ctv);

  TensorView<EmptyBackendTag, const DynamicType> copy_static_to_dynamic{static_dim};
  compare(copy_static_to_dynamic, ctv);

  TensorView<EmptyBackendTag, const DynamicType, 3> copy_dynamic_to_static{dynamic_dim};
  compare(copy_dynamic_to_static, ctv);

  TensorView<EmptyBackendTag, const DynamicType, 3> copy_dynamic_to_dynamic{dynamic_dim};
  compare(copy_dynamic_to_dynamic, ctv);
}

template <typename Expected, typename Actual>
void compare(const Actual &) {
  static_assert(std::is_same<Expected, Actual>::value, "Static type test");
}


TEST(DynamicTypesTensorViewTest, ViewFunction) {
  // TensorView<EmptyBackendTag, int, 3> tv;
  // TensorView<EmptyBackendTag, int, 3> tv1{view<int, 3>(tv)};
  // static_assert(std::is_same<TensorView<EmptyBackendTag, int, 3>,
  //                            decltype(view<int, 3>(TensorView<EmptyBackendTag, int, 3>{}))>::value,
  //               "Static type test");
  auto tv1 = TensorView<EmptyBackendTag, int, 3>{};
  compare<TensorView<EmptyBackendTag, int, 3>>(view<int, 3>(tv1));
  compare<TensorView<EmptyBackendTag, int, DynamicDimensions>>(view<int>(tv1));
  compare<TensorView<EmptyBackendTag, DynamicType, 3>>(view<DynamicType, 3>(tv1));
  compare<TensorView<EmptyBackendTag, DynamicType, DynamicDimensions>>(view<DynamicType>(tv1));

  compare<TensorView<EmptyBackendTag, const int, 3>>(view<const int, 3>(tv1));
  compare<TensorView<EmptyBackendTag, const int, DynamicDimensions>>(view<const int>(tv1));
  compare<TensorView<EmptyBackendTag, const DynamicType, 3>>(view<const DynamicType, 3>(tv1));
  compare<TensorView<EmptyBackendTag, const DynamicType, DynamicDimensions>>(view<const DynamicType>(tv1));

  auto ctv1 = TensorView<EmptyBackendTag, const int, 3>{};
  compare<TensorView<EmptyBackendTag, const int, 3>>(view<const int, 3>(ctv1));
  compare<TensorView<EmptyBackendTag, const int, DynamicDimensions>>(view<const int>(ctv1));
  compare<TensorView<EmptyBackendTag, const DynamicType, 3>>(view<const DynamicType, 3>(ctv1));
  compare<TensorView<EmptyBackendTag, const DynamicType, DynamicDimensions>>(view<const DynamicType>(ctv1));

  auto tv2 = TensorView<EmptyBackendTag, int>{};
  compare<TensorView<EmptyBackendTag, int, 3>>(view<int, 3>(tv2));
  compare<TensorView<EmptyBackendTag, int, DynamicDimensions>>(view<int>(tv2));
  compare<TensorView<EmptyBackendTag, DynamicType, 3>>(view<DynamicType, 3>(tv2));
  compare<TensorView<EmptyBackendTag, DynamicType, DynamicDimensions>>(view<DynamicType>(tv2));

  compare<TensorView<EmptyBackendTag, const int, 3>>(view<const int, 3>(tv2));
  compare<TensorView<EmptyBackendTag, const int, DynamicDimensions>>(view<const int>(tv2));
  compare<TensorView<EmptyBackendTag, const DynamicType, 3>>(view<const DynamicType, 3>(tv2));
  compare<TensorView<EmptyBackendTag, const DynamicType, DynamicDimensions>>(view<const DynamicType>(tv2));

  auto ctv2 = TensorView<EmptyBackendTag, const int>{};
  compare<TensorView<EmptyBackendTag, const int, 3>>(view<const int, 3>(ctv2));
  compare<TensorView<EmptyBackendTag, const int, DynamicDimensions>>(view<const int>(ctv2));
  compare<TensorView<EmptyBackendTag, const DynamicType, 3>>(view<const DynamicType, 3>(ctv2));
  compare<TensorView<EmptyBackendTag, const DynamicType, DynamicDimensions>>(view<const DynamicType>(ctv2));

  auto tv3 = TensorView<EmptyBackendTag, DynamicType, 3>{};
  compare<TensorView<EmptyBackendTag, int, 3>>(view<int, 3>(tv3));
  compare<TensorView<EmptyBackendTag, int, DynamicDimensions>>(view<int>(tv3));
  compare<TensorView<EmptyBackendTag, DynamicType, 3>>(view<DynamicType, 3>(tv3));
  compare<TensorView<EmptyBackendTag, DynamicType, DynamicDimensions>>(view<DynamicType>(tv3));

  compare<TensorView<EmptyBackendTag, const int, 3>>(view<const int, 3>(tv3));
  compare<TensorView<EmptyBackendTag, const int, DynamicDimensions>>(view<const int>(tv3));
  compare<TensorView<EmptyBackendTag, const DynamicType, 3>>(view<const DynamicType, 3>(tv3));
  compare<TensorView<EmptyBackendTag, const DynamicType, DynamicDimensions>>(view<const DynamicType>(tv3));

  auto ctv3 = TensorView<EmptyBackendTag, const DynamicType, 3>{};
  compare<TensorView<EmptyBackendTag, const int, 3>>(view<const int, 3>(ctv3));
  compare<TensorView<EmptyBackendTag, const int, DynamicDimensions>>(view<const int>(ctv3));
  compare<TensorView<EmptyBackendTag, const DynamicType, 3>>(view<const DynamicType, 3>(ctv3));
  compare<TensorView<EmptyBackendTag, const DynamicType, DynamicDimensions>>(view<const DynamicType>(ctv3));

  auto tv4 = TensorView<EmptyBackendTag, DynamicType>{};
  compare<TensorView<EmptyBackendTag, int, 3>>(view<int, 3>(tv4));
  compare<TensorView<EmptyBackendTag, int, DynamicDimensions>>(view<int>(tv4));
  compare<TensorView<EmptyBackendTag, DynamicType, 3>>(view<DynamicType, 3>(tv4));
  compare<TensorView<EmptyBackendTag, DynamicType, DynamicDimensions>>(view<DynamicType>(tv4));

  compare<TensorView<EmptyBackendTag, const int, 3>>(view<const int, 3>(tv4));
  compare<TensorView<EmptyBackendTag, const int, DynamicDimensions>>(view<const int>(tv4));
  compare<TensorView<EmptyBackendTag, const DynamicType, 3>>(view<const DynamicType, 3>(tv4));
  compare<TensorView<EmptyBackendTag, const DynamicType, DynamicDimensions>>(view<const DynamicType>(tv4));

  auto ctv4 = TensorView<EmptyBackendTag, DynamicType>{};
  compare<TensorView<EmptyBackendTag, const int, 3>>(view<const int, 3>(ctv4));
  compare<TensorView<EmptyBackendTag, const int, DynamicDimensions>>(view<const int>(ctv4));
  compare<TensorView<EmptyBackendTag, const DynamicType, 3>>(view<const DynamicType, 3>(ctv4));
  compare<TensorView<EmptyBackendTag, const DynamicType, DynamicDimensions>>(view<const DynamicType>(ctv4));
}


// TEST(DynamicTypesTensorViewTest, Conversions) {
//   TensorView<EmptyBackendTag, int, 4> static_dim{static_cast<int*>(nullptr), {1, 2, 3, 4}};
//   ASSERT_EQ(static_dim.dim(), 4);
//   // Allowed conversions
//   TensorView<EmptyBackendTag, int, DynamicDimensions> dynamic_dim{static_dim};
//   compare(dynamic_dim.shape, static_dim.shape);
//   ASSERT_EQ(dynamic_dim.dim(), 4);
//   TensorView<EmptyBackendTag, int, 4> static_dim_2(dynamic_dim.to_static<4>());
//   compare(static_dim_2.shape, static_dim.shape);
//   compare(static_dim_2.shape, dynamic_dim.shape);

//   dynamic_dim = TensorView<EmptyBackendTag, int, 2>{static_cast<int*>(nullptr), {1, 2}};
//   ASSERT_EQ(dynamic_dim.dim(), 2);
// }

// TEST(DynamicTypesTensorViewTest, Addressing) {
//   TensorView<EmptyBackendTag, int, 3> tv{static_cast<int*>(nullptr), {4, 100, 50}};
//   compare(tv(0, 0, 0), static_cast<int*>(nullptr));
//   compare(tv(0, 0, 1), static_cast<int*>(nullptr) + 1);
//   compare(tv(0, 1, 0), static_cast<int*>(nullptr) + 50);
//   compare(tv(1, 0, 0), static_cast<int*>(nullptr) + 5000);
//   compare(tv(1, 1, 1), static_cast<int*>(nullptr) + 5051);
//   compare(tv(1, 1), static_cast<int*>(nullptr) + 5050);
//   compare(tv(1), static_cast<int*>(nullptr) + 5000);
// }

// TEST(DynamicTypesTensorViewTest, TypePromotion) {
//   int junk_data = 0;
//   TensorView<EmptyBackendTag, int, 10> tv{&junk_data, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}};
//   TensorView<EmptyBackendTag, const int, 10> tvc = tv;
//   compare(tvc.shape, tv.shape);
//   compare(tvc, tv);
//   tvc = {};
//   EXPECT_NE(tvc.shape, tv.shape);
//   compare(tvc, nullptr);
//   tvc = tv;
//   compare(tvc.shape, tv.shape);
//   compare(tvc, tv);

//   TensorView<EmptyBackendTag, int> tv_dyn = tv;
//   compare(tv_dyn.shape, tv.shape);
//   compare(tv_dyn, tv);

//   TensorView<EmptyBackendTag, const int> tvc_dyn = tv;
//   compare(tvc_dyn.shape, tv.shape);
//   compare(tvc_dyn, tv);
//   tvc_dyn = {};
//   EXPECT_NE(tvc_dyn.shape, tv.shape);
//   compare(tvc_dyn, nullptr);
//   tvc_dyn = tv;
//   compare(tvc_dyn.shape, tv.shape);
//   compare(tvc_dyn, tv);

//   auto *ptr = tv_dyn.shape.shape();
//   tvc_dyn = std::move(tv_dyn);
//   compare(tvc_dyn.shape.shape(), ptr) << "Move is broken - a copy appeared somewhere.";
// }


// TEST(TensorListViewTest, ObtainingTensorViewFromStatic) {
//   TensorListView<EmptyBackendTag, int, 3> tlv_static{
//       static_cast<int*>(nullptr), {{4, 100, 50}, {2, 10, 5}, {4, 50, 25}, {4, 100, 50}}};

//   auto t0 = tlv_static[0];
//   static_assert(std::is_same<decltype(t0), TensorView<EmptyBackendTag, int, 3>>::value,
//                 "Wrong type");
//   auto t1 = tlv_static.tensor_view<3>(1);
//   static_assert(std::is_same<decltype(t1), TensorView<EmptyBackendTag, int, 3>>::value,
//                 "Wrong type");
//   auto t2 = tlv_static.tensor_view<DynamicDimensions>(2);
//   static_assert(
//       std::is_same<decltype(t2), TensorView<EmptyBackendTag, int, DynamicDimensions>>::value,
//       "Wrong type");
//   compare(t2.dim(), 3);
// }

// TEST(TensorListViewTest, ObtainingTensorViewFromDynamic) {
//   TensorListView<EmptyBackendTag, int> tlv_dynamic{
//       static_cast<int*>(nullptr), {{4, 100, 50}, {2, 10, 5}, {4, 50, 25}, {4, 100, 50}}};
//   auto t0 = tlv_dynamic[0];
//   static_assert(
//       std::is_same<decltype(t0), TensorView<EmptyBackendTag, int, DynamicDimensions>>::value,
//       "Wrong type");
//   auto t1 = tlv_dynamic.tensor_view<3>(1);
//   static_assert(std::is_same<decltype(t1), TensorView<EmptyBackendTag, int, 3>>::value,
//                 "Wrong type");
//   auto t2 = tlv_dynamic.tensor_view<DynamicDimensions>(2);
//   static_assert(
//       std::is_same<decltype(t2), TensorView<EmptyBackendTag, int, DynamicDimensions>>::value,
//       "Wrong type");
//   compare(t2.dim(), 3);
// }


// namespace {

// template<typename DataType, typename Iterable>
// void VerifySubtensor(const DataType *data, Iterable dims, int idx) {
//   auto subtensor_volume = volume(dims.begin() + 1, dims.end());
//   for (int i = 0; i < subtensor_volume; i++) {
//     compare(idx * subtensor_volume + i, data[i]) << "Failed at idx: " << idx << " offset " << i;
//   }
// }

// }  // namespace


// TEST(DynamicTypesTensorViewTest, StaticSubtensorTest) {
//   using namespace std;  // NOLINT
//   constexpr size_t kNDims = 4;
//   array<int64_t, kNDims> dims = {4, 1, 2, 3};
//   vector<int> data(volume(dims), 0);
//   iota(data.begin(), data.end(), 0);
//   auto tv = make_tensor_cpu<kNDims>(data(), dims);
//   for (int i = 0; i < dims[0]; i++) {
//     auto ret = subtensor(tv, i);
//     VerifySubtensor(ret, dims, i);
//   }
// }


// TEST(DynamicTypesTensorViewTest, DynamicSubtensorTest) {
//   using namespace std;  // NOLINT
//   vector<int64_t> dims = {4, 2, 1, 2, 3};
//   vector<int> data(volume(dims), 0);
//   iota(data.begin(), data.end(), 0);
//   auto tv = make_tensor_cpu<-1>(data(), dims);
//   for (int i = 0; i < dims[0]; i++) {
//     auto ret = subtensor(tv, i);
//     VerifySubtensor(ret, dims, i);
//   }
// }

// TEST(DynamicTypesTensorViewTest, CollapseDim) {
//   int d1[5] = {};
//   TensorView<EmptyBackendTag, int, 2> t2(d1, { 3, 4 });
//   compare(collapse_dim(t2, 0).shape, (TensorShape<1>{12}));
//   TensorView<EmptyBackendTag, int, 3> t3(d1, { 3, 4, 5});
//   compare(collapse_dim(t3, 0).shape, (TensorShape<2>{12, 5}));
//   compare(collapse_dim(t3, 1).shape, (TensorShape<2>{3, 20}));
//   compare(collapse_dim(t3, 1), d1);
//   TensorView<EmptyBackendTag, int, -1> td(d1, TensorShape<>{ 5, 4, 3, 2});
//   compare(collapse_dim(td, 0).shape, (TensorShape<>{20, 3, 2}));
//   compare(collapse_dim(td, 1).shape, (TensorShape<>{5, 12, 2}));
//   compare(collapse_dim(td, 2).shape, (TensorShape<>{5, 4, 6}));
// }

}  // namespace dali
