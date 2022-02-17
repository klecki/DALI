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

#include <gtest/gtest.h>
#include <numeric>
#include <utility>

#include "dali/core/tensor_shape.h"
#include "dali/core/tensor_view.h"
#include "dali/pipeline/data/dynamic_tensor_view.h"
#include "dali/pipeline/data/types.h"

namespace dali {
namespace kernels {


TEST(DynamicTensorViewTest, Constructors) {
  DynamicTensorView<EmptyBackendTag, 4> empty_static_dim{};
  EXPECT_EQ(empty_static_dim.data, nullptr);
  EXPECT_EQ(empty_static_dim.shape, TensorShape<4>{});
  EXPECT_EQ(empty_static_dim.type_id, DALI_NO_TYPE);


  DynamicTensorView<EmptyBackendTag> empty_dynamic_dim{};
  EXPECT_EQ(empty_dynamic_dim.data, nullptr);
  EXPECT_EQ(empty_dynamic_dim.shape, TensorShape<>{});
  EXPECT_EQ(empty_dynamic_dim.type_id, DALI_NO_TYPE);
}

TEST(DynamicTensorViewTest, PtrConstructors) {
  int data = {};
  DynamicTensorView<EmptyBackendTag, 4> empty_static_dim{&data, {1, 2, 3, 4}};
  EXPECT_EQ(empty_static_dim.data, &data);
  EXPECT_EQ(empty_static_dim.shape, (TensorShape<4>{1, 2, 3, 4}));
  EXPECT_EQ(empty_static_dim.type_id, DALI_INT32);

  DynamicTensorView<EmptyBackendTag, 4> empty_static_dim_conv{&data, TensorShape<>{1, 2, 3, 4}};
  EXPECT_EQ(empty_static_dim_conv.data, &data);
  EXPECT_EQ(empty_static_dim_conv.shape, (TensorShape<4>{1, 2, 3, 4}));
  EXPECT_EQ(empty_static_dim_conv.type_id, DALI_INT32);

  DynamicTensorView<EmptyBackendTag> empty_dynamic_dim{&data, {1, 2, 3, 4}};
  EXPECT_EQ(empty_dynamic_dim.data, &data);
  EXPECT_EQ(empty_dynamic_dim.shape, (TensorShape<>{1, 2, 3, 4}));
  EXPECT_EQ(empty_dynamic_dim.type_id, DALI_INT32);

  DynamicTensorView<EmptyBackendTag> empty_dynamic_dim_conv{&data, TensorShape<4>{1, 2, 3, 4}};
  EXPECT_EQ(empty_dynamic_dim_conv.data, &data);
  EXPECT_EQ(empty_dynamic_dim_conv.shape, (TensorShape<>{1, 2, 3, 4}));
  EXPECT_EQ(empty_dynamic_dim_conv.type_id, DALI_INT32);
}

TEST(DynamicTensorViewTest, NullPtrConstructors) {
  DynamicTensorView<EmptyBackendTag, 4> empty_static_dim{nullptr, {1, 2, 3, 4}};
  EXPECT_EQ(empty_static_dim.data, nullptr);
  EXPECT_EQ(empty_static_dim.shape, (TensorShape<4>{1, 2, 3, 4}));
  EXPECT_EQ(empty_static_dim.type_id, DALI_NO_TYPE);

  DynamicTensorView<EmptyBackendTag> empty_dynamic_dim{nullptr, {1, 2, 3, 4}};
  EXPECT_EQ(empty_dynamic_dim.data, nullptr);
  EXPECT_EQ(empty_dynamic_dim.shape, (TensorShape<>{1, 2, 3, 4}));
  EXPECT_EQ(empty_dynamic_dim.type_id, DALI_NO_TYPE);
}

TEST(DynamicTensorViewTest, TypeIdConstructors) {
  DynamicTensorView<EmptyBackendTag, 4> empty_static_dim{nullptr, {1, 2, 3, 4}, DALI_INT32};
  EXPECT_EQ(empty_static_dim.data, nullptr);
  EXPECT_EQ(empty_static_dim.shape, (TensorShape<4>{1, 2, 3, 4}));
  EXPECT_EQ(empty_static_dim.type_id, DALI_INT32);

  DynamicTensorView<EmptyBackendTag> empty_dynamic_dim{nullptr, {1, 2, 3, 4}, DALI_INT32};
  EXPECT_EQ(empty_dynamic_dim.data, nullptr);
  EXPECT_EQ(empty_dynamic_dim.shape, (TensorShape<>{1, 2, 3, 4}));
  EXPECT_EQ(empty_dynamic_dim.type_id, DALI_INT32);

  DynamicTensorView<EmptyBackendTag> empty_dynamic_dim_conv{nullptr, TensorShape<4>{1, 2, 3, 4}, DALI_INT32};
  EXPECT_EQ(empty_dynamic_dim_conv.data, nullptr);
  EXPECT_EQ(empty_dynamic_dim_conv.shape, (TensorShape<>{1, 2, 3, 4}));
  EXPECT_EQ(empty_dynamic_dim_conv.type_id, DALI_INT32);
}

TEST(DynamicTensorViewTest, ViewConverterConstructors) {
  int data = {};
  TensorView<EmptyBackendTag, int, 3> tv{&data, {1, 2, 3}};

  DynamicTensorView<EmptyBackendTag, 3> static_dim{tv};
  EXPECT_EQ(static_dim.data, tv.data);
  EXPECT_EQ(static_dim.shape, tv.shape);
  EXPECT_EQ(static_dim.type_id, DALI_INT32);


  DynamicTensorView<EmptyBackendTag> dynamic_dim{tv};
  EXPECT_EQ(dynamic_dim.data, tv.data);
  EXPECT_EQ(dynamic_dim.shape, tv.shape);
  EXPECT_EQ(dynamic_dim.type_id, DALI_INT32);

}

// TEST(DynamicTensorViewTest, Conversions) {
//   TensorView<EmptyBackendTag, int, 4> static_dim{static_cast<int*>(nullptr), {1, 2, 3, 4}};
//   ASSERT_EQ(static_dim.dim(), 4);
//   // Allowed conversions
//   TensorView<EmptyBackendTag, int, DynamicDimensions> dynamic_dim{static_dim};
//   EXPECT_EQ(dynamic_dim.shape, static_dim.shape);
//   ASSERT_EQ(dynamic_dim.dim(), 4);
//   TensorView<EmptyBackendTag, int, 4> static_dim_2(dynamic_dim.to_static<4>());
//   EXPECT_EQ(static_dim_2.shape, static_dim.shape);
//   EXPECT_EQ(static_dim_2.shape, dynamic_dim.shape);

//   dynamic_dim = TensorView<EmptyBackendTag, int, 2>{static_cast<int*>(nullptr), {1, 2}};
//   ASSERT_EQ(dynamic_dim.dim(), 2);
// }

// TEST(DynamicTensorViewTest, Addressing) {
//   TensorView<EmptyBackendTag, int, 3> tv{static_cast<int*>(nullptr), {4, 100, 50}};
//   EXPECT_EQ(tv(0, 0, 0), static_cast<int*>(nullptr));
//   EXPECT_EQ(tv(0, 0, 1), static_cast<int*>(nullptr) + 1);
//   EXPECT_EQ(tv(0, 1, 0), static_cast<int*>(nullptr) + 50);
//   EXPECT_EQ(tv(1, 0, 0), static_cast<int*>(nullptr) + 5000);
//   EXPECT_EQ(tv(1, 1, 1), static_cast<int*>(nullptr) + 5051);
//   EXPECT_EQ(tv(1, 1), static_cast<int*>(nullptr) + 5050);
//   EXPECT_EQ(tv(1), static_cast<int*>(nullptr) + 5000);
// }

// TEST(DynamicTensorViewTest, TypePromotion) {
//   int junk_data = 0;
//   TensorView<EmptyBackendTag, int, 10> tv{&junk_data, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}};
//   TensorView<EmptyBackendTag, const int, 10> tvc = tv;
//   EXPECT_EQ(tvc.shape, tv.shape);
//   EXPECT_EQ(tvc.data, tv.data);
//   tvc = {};
//   EXPECT_NE(tvc.shape, tv.shape);
//   EXPECT_EQ(tvc.data, nullptr);
//   tvc = tv;
//   EXPECT_EQ(tvc.shape, tv.shape);
//   EXPECT_EQ(tvc.data, tv.data);

//   TensorView<EmptyBackendTag, int> tv_dyn = tv;
//   EXPECT_EQ(tv_dyn.shape, tv.shape);
//   EXPECT_EQ(tv_dyn.data, tv.data);

//   TensorView<EmptyBackendTag, const int> tvc_dyn = tv;
//   EXPECT_EQ(tvc_dyn.shape, tv.shape);
//   EXPECT_EQ(tvc_dyn.data, tv.data);
//   tvc_dyn = {};
//   EXPECT_NE(tvc_dyn.shape, tv.shape);
//   EXPECT_EQ(tvc_dyn.data, nullptr);
//   tvc_dyn = tv;
//   EXPECT_EQ(tvc_dyn.shape, tv.shape);
//   EXPECT_EQ(tvc_dyn.data, tv.data);

//   auto *ptr = tv_dyn.shape.shape.data();
//   tvc_dyn = std::move(tv_dyn);
//   EXPECT_EQ(tvc_dyn.shape.shape.data(), ptr) << "Move is broken - a copy appeared somewhere.";
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
//   EXPECT_EQ(t2.dim(), 3);
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
//   EXPECT_EQ(t2.dim(), 3);
// }


// namespace {

// template<typename DataType, typename Iterable>
// void VerifySubtensor(const DataType *data, Iterable dims, int idx) {
//   auto subtensor_volume = volume(dims.begin() + 1, dims.end());
//   for (int i = 0; i < subtensor_volume; i++) {
//     EXPECT_EQ(idx * subtensor_volume + i, data[i]) << "Failed at idx: " << idx << " offset " << i;
//   }
// }

// }  // namespace


// TEST(DynamicTensorViewTest, StaticSubtensorTest) {
//   using namespace std;  // NOLINT
//   constexpr size_t kNDims = 4;
//   array<int64_t, kNDims> dims = {4, 1, 2, 3};
//   vector<int> data(volume(dims), 0);
//   iota(data.begin(), data.end(), 0);
//   auto tv = make_tensor_cpu<kNDims>(data.data(), dims);
//   for (int i = 0; i < dims[0]; i++) {
//     auto ret = subtensor(tv, i);
//     VerifySubtensor(ret.data, dims, i);
//   }
// }


// TEST(DynamicTensorViewTest, DynamicSubtensorTest) {
//   using namespace std;  // NOLINT
//   vector<int64_t> dims = {4, 2, 1, 2, 3};
//   vector<int> data(volume(dims), 0);
//   iota(data.begin(), data.end(), 0);
//   auto tv = make_tensor_cpu<-1>(data.data(), dims);
//   for (int i = 0; i < dims[0]; i++) {
//     auto ret = subtensor(tv, i);
//     VerifySubtensor(ret.data, dims, i);
//   }
// }

// TEST(DynamicTensorViewTest, CollapseDim) {
//   int d1[5] = {};
//   TensorView<EmptyBackendTag, int, 2> t2(d1, { 3, 4 });
//   EXPECT_EQ(collapse_dim(t2, 0).shape, (TensorShape<1>{12}));
//   TensorView<EmptyBackendTag, int, 3> t3(d1, { 3, 4, 5});
//   EXPECT_EQ(collapse_dim(t3, 0).shape, (TensorShape<2>{12, 5}));
//   EXPECT_EQ(collapse_dim(t3, 1).shape, (TensorShape<2>{3, 20}));
//   EXPECT_EQ(collapse_dim(t3, 1).data, d1);
//   TensorView<EmptyBackendTag, int, -1> td(d1, TensorShape<>{ 5, 4, 3, 2});
//   EXPECT_EQ(collapse_dim(td, 0).shape, (TensorShape<>{20, 3, 2}));
//   EXPECT_EQ(collapse_dim(td, 1).shape, (TensorShape<>{5, 12, 2}));
//   EXPECT_EQ(collapse_dim(td, 2).shape, (TensorShape<>{5, 4, 6}));
// }

}  // namespace kernels
}  // namespace dali
