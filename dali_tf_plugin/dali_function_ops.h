// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

#ifndef DALI_TF_PLUGIN_DALI_FUNCTION_OPS_H_
#define DALI_TF_PLUGIN_DALI_FUNCTION_OPS_H_

#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/op_kernel.h"


namespace dali_tf_impl {

class RemoteCallOp : public tensorflow::AsyncOpKernel {
 public:
  explicit RemoteCallOp(tensorflow::OpKernelConstruction* ctx);

  ~RemoteCallOp() override {}

  void ComputeAsync(tensorflow::OpKernelContext* ctx, DoneCallback done) override;

  std::string TraceString(const tensorflow::OpKernelContext& ctx, bool verbose) const override;

 private:
  tensorflow::NameAttrList func_;
  tensorflow::DataTypeVector input_dtypes_;
  tensorflow::DataTypeVector output_dtypes_;

  tensorflow::mutex mu_;
  typedef std::pair<std::string, tensorflow::FunctionLibraryRuntime*> FunctionTarget;
  std::map<FunctionTarget, tensorflow::FunctionLibraryRuntime::Handle> handle_cache_
      TF_GUARDED_BY(mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(RemoteCallOp);
};

}  // namespace dali_tf_impl
#endif  // DALI_TF_PLUGIN_DALI_FUNCTION_OPS_H_
