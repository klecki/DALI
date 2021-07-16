// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

#include "dali_tf_plugin/dali_function_ops.h"

#include <deque>
#include <vector>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/executor.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/gradients.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/memory_types.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/util/device_name_utils.h"

using namespace tensorflow;  // NOLINT(build/namespaces)

namespace dali_tf_impl {

RemoteCallOp::RemoteCallOp(OpKernelConstruction* ctx) : AsyncOpKernel(ctx) {
  OP_REQUIRES_OK(ctx,
                 ctx->GetAttr(FunctionLibraryDefinition::kFuncAttr, &func_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("Tin", &input_dtypes_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("Tout", &output_dtypes_));
}

void RemoteCallOp::ComputeAsync(OpKernelContext* ctx, DoneCallback done) {
  FunctionLibraryRuntime* lib = ctx->function_library();
  OP_REQUIRES_ASYNC(ctx, lib != nullptr,
                    errors::Internal("No function library is provided."), done);

  const string& source_device = lib->device()->name();
  const Tensor* target;
  OP_REQUIRES_OK_ASYNC(ctx, ctx->input("target", &target), done);

  FunctionTarget function_target;
  OP_REQUIRES_OK_ASYNC(
      ctx,
      DeviceNameUtils::CanonicalizeDeviceName(
          target->scalar<tstring>()(), source_device, &function_target.first),
      done);
  function_target.second = lib;

  const string& target_device = function_target.first;
  const string& func_name = func_.name();

  FunctionLibraryRuntime::Handle handle;
  {
    mutex_lock l(mu_);
    auto cached_entry = handle_cache_.find(function_target);
    if (cached_entry != handle_cache_.end()) {
      handle = cached_entry->second;
    } else {
      VLOG(1) << "Instantiating " << func_name << " on " << target_device;
      profiler::TraceMe activity(
          [&] {
            return strings::StrCat("RemoteCall: Instantiate: ", func_name,
                                   " on ", target_device);
          },
          profiler::TraceMeLevel::kInfo);
      FunctionLibraryRuntime::InstantiateOptions instantiate_opts;
      const auto* config = (ctx->function_library())
                               ? ctx->function_library()->config_proto()
                               : nullptr;
      if (config) {
        instantiate_opts.config_proto = *config;
      }
      instantiate_opts.target = target_device;
      OP_REQUIRES_OK_ASYNC(ctx,
                           lib->Instantiate(func_name, AttrSlice(&func_.attr()),
                                            instantiate_opts, &handle),
                           done);
      auto insert_result = handle_cache_.insert({function_target, handle});
      CHECK(insert_result.second) << "Insert unsuccessful.";
      VLOG(1) << "Instantiated " << func_name << " on " << target_device
              << ", resulting in handle: " << handle << " flr: " << lib;
    }
  }

  OpInputList arguments;
  OP_REQUIRES_OK_ASYNC(ctx, ctx->input_list("args", &arguments), done);

  FunctionLibraryRuntime::Options opts;
  opts.runner = nullptr;  // Use default runner at remote device.
  opts.run_all_kernels_inline = ctx->run_all_kernels_inline();
  opts.source_device = source_device;
  std::cout << "SOURCE : " << source_device << " TARGET: " << target_device << std::endl;
  if (opts.source_device != target_device) {
    opts.remote_execution = true;
  }
  opts.create_rendezvous = true;
  std::vector<Tensor> args(arguments.begin(), arguments.end());
  opts.args_alloc_attrs.reserve(input_dtypes_.size());
  for (const auto& dtype : input_dtypes_) {
    AllocatorAttributes arg_alloc_attrs;
    arg_alloc_attrs.set_on_host(DataTypeAlwaysOnHost(dtype));
    opts.args_alloc_attrs.push_back(arg_alloc_attrs);
  }
  opts.rets_alloc_attrs.reserve(output_dtypes_.size());
  for (const auto& dtype : output_dtypes_) {
    AllocatorAttributes ret_alloc_attrs;
    ret_alloc_attrs.set_on_host(true);
    opts.rets_alloc_attrs.push_back(ret_alloc_attrs);
  }
  auto* rets = new std::vector<Tensor>;
  VLOG(1) << "Running " << func_name << " on " << target_device
          << " with handle: " << handle;
  profiler::TraceMe trace_me(
      [&] {
        return absl::StrCat("RemoteCallOp#func_name=", func_name,
                            ",device=", target_device, "#");
      },
      profiler::TraceMeLevel::kInfo);
  lib->Run(
      opts, handle, args, rets,
      [rets, done = std::move(done), func_name, ctx,
       function_step_id = opts.step_id,
       target_device = std::move(function_target.first)](const Status& status) {
        profiler::TraceMe activity(
            [&] {
              return absl::StrCat("RemoteCallOpDone#func_name=", func_name,
                                  ",device=", target_device, "#");
            },
            profiler::TraceMeLevel::kInfo);
        if (!status.ok()) {
          ctx->SetStatus(status);
        } else {
          for (size_t i = 0; i < rets->size(); ++i) {
            ctx->set_output(i, std::move((*rets)[i]));
          }
        }
        delete rets;
        done();
      });
}

string RemoteCallOp::TraceString(const OpKernelContext& ctx,
                                 bool verbose) const {
  string trace_string = profiler::TraceMeOp(
      strings::StrCat(name_view(), "__", func_.name()), type_string_view());
  if (verbose) {
    string shape = ShapeTraceString(ctx);
    if (!shape.empty()) {
      trace_string =
          profiler::TraceMeEncode(std::move(trace_string), {{"shape", shape}});
    }
  }
  return trace_string;
}



// class RemoteCallOp2 : public  OpKernel {
//  public:
//   explicit RemoteCallOp2( OpKernelConstruction* ctx) : OpKernel(ctx) {}

//   ~RemoteCallOp2() override {}

//   void Compute( OpKernelContext* ctx) override {}

//   std::string TraceString(const  OpKernelContext& ctx, bool verbose) const override {}

//  private:

//   TF_DISALLOW_COPY_AND_ASSIGN(RemoteCallOp2);
// };

REGISTER_KERNEL_BUILDER(
    Name("PassCpuRemoteCall").Device(DEVICE_CPU).HostMemory("target").HostMemory("output"), RemoteCallOp);
REGISTER_KERNEL_BUILDER(
    Name("PassCpuRemoteCall").Device(DEVICE_GPU).HostMemory("target").HostMemory("output"), RemoteCallOp);


REGISTER_OP("PassCpuRemoteCall")
    .Input("target: string")
    .Input("args: Tin")
    .Output("output: Tout")
    .Attr("Tin: list(type)")
    .Attr("Tout: list(type)")
    .Attr("f: func")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape);


}  // namespace dali_tf_impl
