// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/kernels/shape.h"
#include "dali/kernels/tensor_view.h"

namespace tensor {

TEST(TensorShapeTest, StaticShapeConstructor) {
  TensorShape<0> zero_tensor;

  constexpr int test_dim = 5;
  std::array<int64_t, test_dim> test_shape = {1, 2, 3, 4, 5};

  // Default constructor
  TensorShape<test_dim> empty_tensor;
  for (int i = 0; i < test_dim; i++) {
    ASSERT_EQ(empty_tensor[i], int64_t(0));
  }

  // std::array and expanded list constructor
  TensorShape<test_dim> a(test_shape);
  TensorShape<test_dim> b(test_shape[0], test_shape[1], test_shape[2], test_shape[3],
                          test_shape[4]);
  ASSERT_EQ(a.size(), test_dim);
  ASSERT_EQ(b.size(), test_dim);
  for (int i = 0; i < test_dim; i++) {
    ASSERT_EQ(a[i], test_shape[i]);
    ASSERT_EQ(b[i], test_shape[i]);
  }

  // Copy constructor
  TensorShape<test_dim> check_construct(a);
  for (int i = 0; i < test_dim; i++) {
    ASSERT_EQ(check_construct[i], a[i]);
  }

  // Assignement
  TensorShape<test_dim> check_assign;
  check_assign = a;
  for (int i = 0; i < test_dim; i++) {
    ASSERT_EQ(check_assign[i], a[i]);
  }

  // Move rvalue
  TensorShape<test_dim> check_move_construct(TensorShape<5>{test_shape});
  for (int i = 0; i < test_dim; i++) {
    ASSERT_EQ(check_move_construct[i], test_shape[i]);
  }

  // Assignement for rvalue
  TensorShape<test_dim> check_move_assign;
  check_move_assign = TensorShape<5>{test_shape};
  for (int i = 0; i < test_dim; i++) {
    ASSERT_EQ(check_move_assign[i], test_shape[i]);
  }
}

TEST(TensorShapeTest, DynamicShapeConstructor) {
  // Default
  TensorShape<DynamicDimensions> zero_tensor;
  ASSERT_EQ(zero_tensor.size(), 0);

  // std::array constructor
  constexpr int test_dim = 5;
  std::array<int64_t, test_dim> test_shape_arr = {1, 2, 3, 4, 5};
  TensorShape<DynamicDimensions> a(test_shape_arr);
  ASSERT_EQ(a.size(), test_dim);
  for (int i = 0; i < test_dim; i++) {
    ASSERT_EQ(a[i], test_shape_arr[i]);
  }

  // std::vector constructor
  std::vector<int64_t> test_shape_vec = {1, 2, 3, 4, 5, 6, 7};
  TensorShape<DynamicDimensions> b(test_shape_vec);
  ASSERT_EQ(b.size(), test_shape_vec.size());
  for (int i = 0; i < test_shape_vec.size(); i++) {
    ASSERT_EQ(b[i], test_shape_vec[i]);
  }

  // Expanded arguments constructor
  TensorShape<DynamicDimensions> c(1);
  ASSERT_EQ(c.size(), 1);
  ASSERT_EQ(c[0], 1);

  TensorShape<DynamicDimensions> d(1, 2, 3, 4);
  ASSERT_EQ(d.size(), 4);
  ASSERT_EQ(d[0], 1);
  ASSERT_EQ(d[1], 2);
  ASSERT_EQ(d[2], 3);
  ASSERT_EQ(d[3], 4);

  // Copy constructor
  TensorShape<DynamicDimensions> check_construct(a);
  for (int i = 0; i < test_dim; i++) {
    ASSERT_EQ(check_construct[i], a[i]);
  }

  // Asignement
  TensorShape<DynamicDimensions> check_assign;
  check_assign = a;
  ASSERT_EQ(check_assign.size(), a.size());
  for (int i = 0; i < a.size(); i++) {
    ASSERT_EQ(check_assign[i], a[i]);
  }

  // Second asignement to the same value
  check_assign = b;
  ASSERT_EQ(check_assign.size(), b.size());
  for (int i = 0; i < b.size(); i++) {
    ASSERT_EQ(check_assign[i], b[i]);
  }

  // Move rvalue
  TensorShape<DynamicDimensions> check_move_construct(
      TensorShape<DynamicDimensions>{test_shape_arr});
  for (int i = 0; i < test_dim; i++) {
    ASSERT_EQ(check_move_construct[i], test_shape_arr[i]);
  }

  // Assignement for rvalue
  TensorShape<DynamicDimensions> check_move_assign;
  check_move_assign = TensorShape<DynamicDimensions>{test_shape_arr};
  for (int i = 0; i < test_dim; i++) {
    ASSERT_EQ(check_move_assign[i], test_shape_arr[i]);
  }
}

TEST(TensorShapeTest, StaticDynamicConversions) {
  TensorShape<3> static_shape_3(1, 2, 3);
  TensorShape<5> static_shape_5(1, 2, 3, 4, 5);

  TensorShape<DynamicDimensions> check_construct_3(static_shape_3);
  ASSERT_EQ(check_construct_3.size(), static_shape_3.size());
  for (int i = 0; i < static_shape_3.size(); i++) {
    ASSERT_EQ(check_construct_3[i], static_shape_3[i]);
  }

  TensorShape<DynamicDimensions> check_construct_5(static_shape_5);
  ASSERT_EQ(check_construct_5.size(), static_shape_5.size());
  for (int i = 0; i < static_shape_5.size(); i++) {
    ASSERT_EQ(check_construct_5[i], static_shape_5[i]);
  }

  TensorShape<DynamicDimensions> check_assign;
  check_assign = static_shape_3;
  ASSERT_EQ(check_assign.size(), static_shape_3.size());
  for (int i = 0; i < static_shape_3.size(); i++) {
    ASSERT_EQ(check_assign[i], static_shape_3[i]);
  }

  check_assign = static_shape_5;
  ASSERT_EQ(check_assign.size(), static_shape_5.size());
  for (int i = 0; i < static_shape_5.size(); i++) {
    ASSERT_EQ(check_assign[i], static_shape_5[i]);
  }

  auto s3 = TensorShape<3>{2, 4, 6};
  check_assign = s3;
  static_shape_3 = check_assign.to_static<3>();
  for (int i = 0; i < s3.size(); i++) {
    ASSERT_EQ(static_shape_3[i], s3[i]);
  }
}

TEST(TensorShapeTest, Comparisons) {
  // Static ndim
  ASSERT_TRUE(TensorShape<1>(1) == TensorShape<1>(1));
  ASSERT_FALSE(TensorShape<1>(1) != TensorShape<1>(1));

  ASSERT_FALSE(TensorShape<1>(1) == TensorShape<1>(2));
  ASSERT_TRUE(TensorShape<1>(1) != TensorShape<1>(2));

  ASSERT_TRUE(TensorShape<3>(1, 2, 3) == TensorShape<3>(1, 2, 3));
  ASSERT_FALSE(TensorShape<3>(1, 2, 3) != TensorShape<3>(1, 2, 3));

  ASSERT_FALSE(TensorShape<3>(1, 2, 3) == TensorShape<3>(1, 4, 3));
  ASSERT_TRUE(TensorShape<3>(1, 2, 3) != TensorShape<3>(1, 4, 3));

  ASSERT_FALSE(TensorShape<1>(1) == TensorShape<2>(1, 2));
  ASSERT_TRUE(TensorShape<1>(1) != TensorShape<2>(1, 2));
  ASSERT_FALSE(TensorShape<2>(1, 2) == TensorShape<1>(1));
  ASSERT_TRUE(TensorShape<2>(1, 2) != TensorShape<1>(1));

  // Dynamic ndim
  ASSERT_TRUE(TensorShape<DynamicDimensions>(1) == TensorShape<DynamicDimensions>(1));
  ASSERT_FALSE(TensorShape<DynamicDimensions>(1) != TensorShape<DynamicDimensions>(1));

  ASSERT_FALSE(TensorShape<DynamicDimensions>(1) == TensorShape<DynamicDimensions>(2));
  ASSERT_TRUE(TensorShape<DynamicDimensions>(1) != TensorShape<DynamicDimensions>(2));

  ASSERT_TRUE(TensorShape<DynamicDimensions>(1, 2, 3) == TensorShape<DynamicDimensions>(1, 2, 3));
  ASSERT_FALSE(TensorShape<DynamicDimensions>(1, 2, 3) != TensorShape<DynamicDimensions>(1, 2, 3));

  ASSERT_FALSE(TensorShape<DynamicDimensions>(1, 2, 3) == TensorShape<DynamicDimensions>(1, 4, 3));
  ASSERT_TRUE(TensorShape<DynamicDimensions>(1, 2, 3) != TensorShape<DynamicDimensions>(1, 4, 3));

  ASSERT_FALSE(TensorShape<DynamicDimensions>(1) == TensorShape<DynamicDimensions>(1, 2));
  ASSERT_TRUE(TensorShape<DynamicDimensions>(1) != TensorShape<DynamicDimensions>(1, 2));
  ASSERT_FALSE(TensorShape<DynamicDimensions>(1, 2) == TensorShape<DynamicDimensions>(1));
  ASSERT_TRUE(TensorShape<DynamicDimensions>(1, 2) != TensorShape<DynamicDimensions>(1));

  // Mixed ndim
  ASSERT_TRUE(TensorShape<1>(1) == TensorShape<DynamicDimensions>(1));
  ASSERT_FALSE(TensorShape<1>(1) != TensorShape<DynamicDimensions>(1));
  ASSERT_TRUE(TensorShape<DynamicDimensions>(1) == TensorShape<1>(1));
  ASSERT_FALSE(TensorShape<DynamicDimensions>(1) != TensorShape<1>(1));

  ASSERT_FALSE(TensorShape<1>(1) == TensorShape<DynamicDimensions>(2));
  ASSERT_TRUE(TensorShape<1>(1) != TensorShape<DynamicDimensions>(2));
  ASSERT_FALSE(TensorShape<DynamicDimensions>(1) == TensorShape<1>(2));
  ASSERT_TRUE(TensorShape<DynamicDimensions>(1) != TensorShape<1>(2));

  ASSERT_TRUE(TensorShape<3>(1, 2, 3) == TensorShape<DynamicDimensions>(1, 2, 3));
  ASSERT_FALSE(TensorShape<3>(1, 2, 3) != TensorShape<DynamicDimensions>(1, 2, 3));
  ASSERT_TRUE(TensorShape<DynamicDimensions>(1, 2, 3) == TensorShape<3>(1, 2, 3));
  ASSERT_FALSE(TensorShape<DynamicDimensions>(1, 2, 3) != TensorShape<3>(1, 2, 3));

  ASSERT_FALSE(TensorShape<3>(1, 2, 3) == TensorShape<DynamicDimensions>(1, 4, 3));
  ASSERT_TRUE(TensorShape<3>(1, 2, 3) != TensorShape<DynamicDimensions>(1, 4, 3));
  ASSERT_FALSE(TensorShape<DynamicDimensions>(1, 2, 3) == TensorShape<3>(1, 4, 3));
  ASSERT_TRUE(TensorShape<DynamicDimensions>(1, 2, 3) != TensorShape<3>(1, 4, 3));

  ASSERT_FALSE(TensorShape<1>(1) == TensorShape<DynamicDimensions>(1, 2));
  ASSERT_TRUE(TensorShape<1>(1) != TensorShape<DynamicDimensions>(1, 2));
  ASSERT_FALSE(TensorShape<2>(1, 2) == TensorShape<DynamicDimensions>(1));
  ASSERT_TRUE(TensorShape<2>(1, 2) != TensorShape<DynamicDimensions>(1));
  ASSERT_FALSE(TensorShape<DynamicDimensions>(1) == TensorShape<2>(1, 2));
  ASSERT_TRUE(TensorShape<DynamicDimensions>(1) != TensorShape<2>(1, 2));
  ASSERT_FALSE(TensorShape<DynamicDimensions>(1, 2) == TensorShape<1>(1));
  ASSERT_TRUE(TensorShape<DynamicDimensions>(1, 2) != TensorShape<1>(1));
}

TEST(TensorShapeTest, RangeLoop) {
  TensorShape<5> ts{0, 1, 2, 3, 4};
  int expected = 0;
  for (auto s : ts) {
    ASSERT_EQ(s, expected);
    expected++;
  }
}

TEST(TensorShapeTest, FirstStaticOnStatic) {
  TensorShape<5> ts(1, 2, 3, 4, 5);
  ASSERT_EQ(ts.first<0>(), TensorShape<0>());
  ASSERT_EQ(ts.first<1>(), TensorShape<1>(1));
  ASSERT_EQ(ts.first<2>(), TensorShape<2>(1, 2));
  ASSERT_EQ(ts.first<3>(), TensorShape<3>(1, 2, 3));
  ASSERT_EQ(ts.first<4>(), TensorShape<4>(1, 2, 3, 4));
  ASSERT_EQ(ts.first<5>(), TensorShape<5>(1, 2, 3, 4, 5));
}

TEST(TensorShapeTest, LastStaticOnStatic) {
  TensorShape<5> ts(1, 2, 3, 4, 5);
  ASSERT_EQ(ts.last<0>(), TensorShape<0>());
  ASSERT_EQ(ts.last<1>(), TensorShape<1>(5));
  ASSERT_EQ(ts.last<2>(), TensorShape<2>(4, 5));
  ASSERT_EQ(ts.last<3>(), TensorShape<3>(3, 4, 5));
  ASSERT_EQ(ts.last<4>(), TensorShape<4>(2, 3, 4, 5));
  ASSERT_EQ(ts.last<5>(), TensorShape<5>(1, 2, 3, 4, 5));
}

TEST(TensorShapeTest, FirstStaticOnDynamic) {
  TensorShape<DynamicDimensions> ts(1, 2, 3, 4, 5);
  ASSERT_EQ(ts.first<0>(), TensorShape<DynamicDimensions>());
  ASSERT_EQ(ts.first<1>(), TensorShape<DynamicDimensions>(1));
  ASSERT_EQ(ts.first<2>(), TensorShape<DynamicDimensions>(1, 2));
  ASSERT_EQ(ts.first<3>(), TensorShape<DynamicDimensions>(1, 2, 3));
  ASSERT_EQ(ts.first<4>(), TensorShape<DynamicDimensions>(1, 2, 3, 4));
  ASSERT_EQ(ts.first<5>(), TensorShape<DynamicDimensions>(1, 2, 3, 4, 5));
}

TEST(TensorShapeTest, LastStaticOnDynamic) {
  TensorShape<DynamicDimensions> ts(1, 2, 3, 4, 5);
  ASSERT_EQ(ts.last<0>(), TensorShape<DynamicDimensions>());
  ASSERT_EQ(ts.last<1>(), TensorShape<DynamicDimensions>(5));
  ASSERT_EQ(ts.last<2>(), TensorShape<DynamicDimensions>(4, 5));
  ASSERT_EQ(ts.last<3>(), TensorShape<DynamicDimensions>(3, 4, 5));
  ASSERT_EQ(ts.last<4>(), TensorShape<DynamicDimensions>(2, 3, 4, 5));
  ASSERT_EQ(ts.last<5>(), TensorShape<DynamicDimensions>(1, 2, 3, 4, 5));
}

TEST(TensorShapeTest, FirstDynamicOnStatic) {
  TensorShape<5> ts(1, 2, 3, 4, 5);
  ASSERT_EQ(ts.first(0), TensorShape<0>());
  ASSERT_EQ(ts.first(1), TensorShape<1>(1));
  ASSERT_EQ(ts.first(2), TensorShape<2>(1, 2));
  ASSERT_EQ(ts.first(3), TensorShape<3>(1, 2, 3));
  ASSERT_EQ(ts.first(4), TensorShape<4>(1, 2, 3, 4));
  ASSERT_EQ(ts.first(5), TensorShape<5>(1, 2, 3, 4, 5));
}

TEST(TensorShapeTest, LastDynamicOnStatic) {
  TensorShape<5> ts(1, 2, 3, 4, 5);
  ASSERT_EQ(ts.last(0), TensorShape<0>());
  ASSERT_EQ(ts.last(1), TensorShape<1>(5));
  ASSERT_EQ(ts.last(2), TensorShape<2>(4, 5));
  ASSERT_EQ(ts.last(3), TensorShape<3>(3, 4, 5));
  ASSERT_EQ(ts.last(4), TensorShape<4>(2, 3, 4, 5));
  ASSERT_EQ(ts.last(5), TensorShape<5>(1, 2, 3, 4, 5));
}

TEST(TensorShapeTest, FirstDynamicOnDynamic) {
  TensorShape<DynamicDimensions> ts(1, 2, 3, 4, 5);
  ASSERT_EQ(ts.first(0), TensorShape<DynamicDimensions>());
  ASSERT_EQ(ts.first(1), TensorShape<DynamicDimensions>(1));
  ASSERT_EQ(ts.first(2), TensorShape<DynamicDimensions>(1, 2));
  ASSERT_EQ(ts.first(3), TensorShape<DynamicDimensions>(1, 2, 3));
  ASSERT_EQ(ts.first(4), TensorShape<DynamicDimensions>(1, 2, 3, 4));
  ASSERT_EQ(ts.first(5), TensorShape<DynamicDimensions>(1, 2, 3, 4, 5));
}

TEST(TensorShapeTest, LastDynamicOnDynamic) {
  TensorShape<DynamicDimensions> ts(1, 2, 3, 4, 5);
  ASSERT_EQ(ts.last(0), TensorShape<DynamicDimensions>());
  ASSERT_EQ(ts.last(1), TensorShape<DynamicDimensions>(5));
  ASSERT_EQ(ts.last(2), TensorShape<DynamicDimensions>(4, 5));
  ASSERT_EQ(ts.last(3), TensorShape<DynamicDimensions>(3, 4, 5));
  ASSERT_EQ(ts.last(4), TensorShape<DynamicDimensions>(2, 3, 4, 5));
  ASSERT_EQ(ts.last(5), TensorShape<DynamicDimensions>(1, 2, 3, 4, 5));
}

TEST(TensorShapeTest, Concatenation) {
  ASSERT_EQ(shape_cat(TensorShape<0>(), TensorShape<0>()), TensorShape<0>());
  ASSERT_EQ(shape_cat(TensorShape<1>(1), TensorShape<0>()), TensorShape<1>(1));
  ASSERT_EQ(shape_cat(TensorShape<0>(), TensorShape<1>(1)), TensorShape<1>(1));
  ASSERT_EQ(shape_cat(TensorShape<2>(1, 2), TensorShape<3>(1, 2, 3)),
            TensorShape<5>(1, 2, 1, 2, 3));

  ASSERT_EQ(shape_cat(TensorShape<DynamicDimensions>(), TensorShape<DynamicDimensions>()),
            TensorShape<DynamicDimensions>());
  ASSERT_EQ(shape_cat(TensorShape<DynamicDimensions>(1), TensorShape<DynamicDimensions>()),
            TensorShape<DynamicDimensions>(1));
  ASSERT_EQ(shape_cat(TensorShape<DynamicDimensions>(), TensorShape<DynamicDimensions>(1)),
            TensorShape<DynamicDimensions>(1));
  ASSERT_EQ(
      shape_cat(TensorShape<DynamicDimensions>(1, 2), TensorShape<DynamicDimensions>(1, 2, 3)),
      TensorShape<DynamicDimensions>(1, 2, 1, 2, 3));

  ASSERT_EQ(shape_cat(TensorShape<DynamicDimensions>(), TensorShape<0>()),
            TensorShape<DynamicDimensions>());
  ASSERT_EQ(shape_cat(TensorShape<DynamicDimensions>(1), TensorShape<0>()),
            TensorShape<DynamicDimensions>(1));
  ASSERT_EQ(shape_cat(TensorShape<DynamicDimensions>(), TensorShape<1>(1)),
            TensorShape<DynamicDimensions>(1));
  ASSERT_EQ(
      shape_cat(TensorShape<DynamicDimensions>(1, 2), TensorShape<DynamicDimensions>(1, 2, 3)),
      TensorShape<DynamicDimensions>(1, 2, 1, 2, 3));

  ASSERT_EQ(shape_cat(TensorShape<0>(), TensorShape<DynamicDimensions>()),
            TensorShape<DynamicDimensions>());
  ASSERT_EQ(shape_cat(TensorShape<1>(1), TensorShape<DynamicDimensions>()),
            TensorShape<DynamicDimensions>(1));
  ASSERT_EQ(shape_cat(TensorShape<0>(), TensorShape<DynamicDimensions>(1)),
            TensorShape<DynamicDimensions>(1));
  ASSERT_EQ(shape_cat(TensorShape<2>(1, 2), TensorShape<DynamicDimensions>(1, 2, 3)),
            TensorShape<DynamicDimensions>(1, 2, 1, 2, 3));
}

TEST(TensorViewTest, Conversions) {
  TensorView<EmptyBackendTag, int, 4> empty_static_dim{};
  ASSERT_EQ(empty_static_dim.data, nullptr);
  ASSERT_EQ(empty_static_dim.shape, TensorShape<4>());

  TensorView<EmptyBackendTag, int, 4> static_dim{static_cast<int*>(nullptr), {1, 2, 3, 4}};
  // Allowed conversions
  TensorView<EmptyBackendTag, int, DynamicDimensions> dynamic_dim{static_dim};
  ASSERT_EQ(dynamic_dim.shape, static_dim.shape);
  TensorView<EmptyBackendTag, int, 4> static_dim_2(dynamic_dim.to_static<4>());
  ASSERT_EQ(static_dim_2.shape, static_dim.shape);
  ASSERT_EQ(static_dim_2.shape, dynamic_dim.shape);
}

TEST(VolumeTest, Result) {
  ASSERT_EQ(volume(std::vector<int64_t>{}), 0);
  ASSERT_EQ(volume(std::vector<int64_t>{1}), 1);
  ASSERT_EQ(volume(std::vector<int64_t>{1, 2}), 2);
  ASSERT_EQ(volume(std::vector<int64_t>{1, 2, 3, 4, 5}), 1 * 2 * 3 * 4 * 5);
}

TEST(FlattenTest, StaticTensorShape) {
  auto shapes_vec = std::vector<TensorShape<3>>{{1, 2, 3}, {2, 3, 4}, {3, 4, 5}};
  auto expected = std::vector<int64_t>{1, 2, 3, 2, 3, 4, 3, 4, 5};
  ASSERT_EQ(flatten_shapes(shapes_vec), expected);
}

TEST(FlattenTest, DynamicTensorShape) {
  auto shapes_vec = std::vector<TensorShape<>>{{1, 2, 3}, {2, 3, 4}, {3, 4, 5}};
  auto vec_vec = std::vector<std::vector<int64_t>>{{1, 2, 3}, {2, 3, 4}, {3, 4, 5}};
  auto expected = std::vector<int64_t>{1, 2, 3, 2, 3, 4, 3, 4, 5};
  ASSERT_EQ(flatten_shapes(shapes_vec), expected);
  ASSERT_EQ(flatten_shapes(vec_vec), expected);
}

TEST(CalculateOffsetsTest, Result) {
  TensorListShape<3> tls_static({{1, 2, 3}, {2, 3, 4}, {3, 4, 5}});
  TensorListShape<> tls_dynamic(
      std::vector<TensorShape<DynamicDimensions>>{{1, 2, 3}, {2, 3, 4}, {3, 4, 5}});
  auto static_offs = calculate_offsets(tls_static);
  auto dynamic_offs = calculate_offsets(tls_dynamic);
  auto expected = std::vector<ptrdiff_t>{0, 6, 30, 90};
  ASSERT_EQ(static_offs, expected);
  ASSERT_EQ(dynamic_offs, expected);
}

TEST(TensorListShape, IsUniform) {
  // TensorListShape
}

TEST(TensorListShape, FirstStatic) {
  TensorListShape<3> tls_static({{1, 2, 3}, {2, 3, 4}, {3, 4, 5}});
  TensorListShape<> tls_dynamic(
      std::vector<TensorShape<DynamicDimensions>>{{1, 2, 3}, {2, 3, 4}, {3, 4, 5}});

  TensorListShape<0> expected_static_0(std::vector<TensorShape<0>>{{}, {}, {}});
  TensorListShape<1> expected_static_1(std::vector<TensorShape<1>>{{1}, {2}, {3}});
  TensorListShape<2> expected_static_2(std::vector<TensorShape<2>>{{1, 2}, {2, 3}, {3, 4}});
  TensorListShape<3> expected_static_3(std::vector<TensorShape<3>>{{1, 2, 3}, {2, 3, 4}, {3, 4, 5}});

  auto first_0 = tls_static.first<0>();
  auto first_1 = tls_static.first<1>();
  auto first_2 = tls_static.first<2>();
  auto first_3 = tls_static.first<3>();
  ASSERT_EQ(first_0, expected_static_0);
  ASSERT_EQ(first_1, expected_static_1);
  ASSERT_EQ(first_2, expected_static_2);
  ASSERT_EQ(first_3, expected_static_3);
}

TEST(TensorListShape, ToStatic) {
  TensorListShape<> tls_dynamic(
      std::vector<TensorShape<DynamicDimensions>>{{1, 2, 3}, {2, 3, 4}, {3, 4, 5}});
  auto tls_static = tls_dynamic.to_static<3>();
  ASSERT_EQ(tls_static, tls_dynamic);
  static_assert(std::is_same<decltype(tls_static), TensorListShape<3>>::value, "Wrong type");
  auto copy(tls_dynamic);
  auto moved_static = std::move(copy).to_static<3>();
  ASSERT_EQ(moved_static, tls_dynamic);
  static_assert(std::is_same<decltype(moved_static), TensorListShape<3>>::value, "Wrong type");
}

TEST(TensorTest, WontCompile) {
  // TensorShape<5> static_shape_less(1, 2, 3, 4);
  // TensorShape<5> static_shape_more(1, 2, 3, 4, 5, 6);
  // TensorShape<DynamicDimensions>().to_static<DynamicDimensions>();
  // TensorShape<5>{TensorShape<DynamicDimensions>()};
  // TensorShape<-2> negative;

  // TensorView<EmptyBackendTag, int8_t, 4>(static_cast<int*>(nullptr), {1, 2, 3, 4});
  // TensorView<EmptyBackendTag, int, 4>{TensorView<EmptyBackendTag, int, DynamicDimensions>{}};
  // TensorView<EmptyBackendTag, int, DynamicDimensions>{TensorView<EmptyBackendTag, int8_t, 4>{}};

}

}  // namespace tensor
