// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_PIPELINE_WORKSPACE_WORKSPACE_DATA_FACTORY_H_
#define DALI_PIPELINE_WORKSPACE_WORKSPACE_DATA_FACTORY_H_

#include <memory>

#include "dali/pipeline/graph/op_graph.h"
#include "dali/pipeline/operators/operator.h"
#include "dali/pipeline/workspace/device_workspace.h"
#include "dali/pipeline/workspace/host_workspace.h"
#include "dali/pipeline/workspace/mixed_workspace.h"
#include "dali/pipeline/workspace/support_workspace.h"

#include "dali/kernels/static_switch.h"
#include "dali/kernels/tuple_helpers.h"
#include "dali/pipeline/util/op_type_to_workspace.h"

namespace dali {

/*
 * Mappings from DALIOpType, DALITensorDevice to index in storage_owner_t
 */

constexpr size_t GetStorageIndex(DALIOpType op_type, DALITensorDevice device) {
  return static_cast<size_t>(op_type) * static_cast<size_t>(DALITensorDevice::COUNT) +
         static_cast<size_t>(device);
}

constexpr size_t GetMaxStorageIndex() {
  return GetStorageIndex(DALIOpType::COUNT, static_cast<DALITensorDevice>(0));
}

constexpr DALIOpType GetOpType(size_t storage_idx) {
  return static_cast<DALIOpType>(storage_idx / static_cast<size_t>(DALITensorDevice::COUNT));
}

constexpr DALITensorDevice GetStorageDevice(size_t storage_idx) {
  return static_cast<DALITensorDevice>(storage_idx % static_cast<size_t>(DALITensorDevice::COUNT));
}

// We use a tuple that can hold Output Type from Device, Host, Mixed and Support workspaces,
// so we have a unifided place that can own any of this type
// Additionally, we use order of those types deifned by GetStorageIndex
// We have 4 workspaces with two possible Backends, obtatining 8 types
// TODO(klecki): this can be clearer as
// std::tuple<DeviceOutputType<CPUBackend>, DeviceOutputType<GPUBackend>,
//            HostOutputType<CPUBackend>, ...
// but that way I ensure correct order of types and not use 8 static_asserts
template <size_t storage_idx>
struct storage_type_gen {
  using type = typename workspace_t<GetOpType(
      storage_idx)>::template output_t<storage_backend_t<GetStorageDevice(storage_idx)>>;
};

template <size_t storage_idx>
using storage_gen_t = typename storage_type_gen<storage_idx>::type;

// using storage_owner_t =
//     std::tuple<storage_gen_t<0>, storage_gen_t<1>, storage_gen_t<2>, storage_gen_t<3>,
//                storage_gen_t<4>, storage_gen_t<5>, storage_gen_t<6>, storage_gen_t<7>>;

using storage_owner_t =
    detail::tuple_generator_t<storage_gen_t, detail::build_seq_t<0, GetMaxStorageIndex()>>;

template <size_t op_type>
using workspace_blob_gen_type = std::vector<workspace_t<static_cast<DALIOpType>(op_type)>>;

using workspace_owner_t = detail::tuple_generator_t<
    workspace_blob_gen_type,
    detail::build_seq_t<0, static_cast<int>(DALIOpType::COUNT)>>;

template <DALIOpType op_type, DALITensorDevice device>
struct BatchFactoryImpl {
  static storage_gen_t<GetStorageIndex(op_type, device)> CreateOutputBatch(int batch_size) {
    // Output batch from GPU, MIXED and SUPPORT Ops are shared_ptr<Something>
    using BatchType = typename storage_gen_t<GetStorageIndex(op_type, device)>::element_type;
    return std::make_shared<BatchType>();
  }
  static_assert(op_type == DALIOpType::GPU || op_type == DALIOpType::MIXED,
                "Only GPU and MIXED handled by default case due to use of outermost shared_ptr and "
                "pinned mem usage");
};

// TODO(klecki): Should we use make_shared here as well?
template <DALITensorDevice device>
struct BatchFactoryImpl<DALIOpType::CPU, device> {
  static storage_gen_t<GetStorageIndex(DALIOpType::CPU, device)> CreateOutputBatch(
      int batch_size) {
    DALI_ENFORCE(device == DALITensorDevice::CPU, "Only CPU outputs allowed");
    // Allocate `batch_size` Tensors for this ops
    // results and add them to the workspace.
    storage_gen_t<GetStorageIndex(DALIOpType::CPU, device)> output(batch_size, nullptr);
    for (auto &tensor_ptr : output) {
      tensor_ptr.reset(new Tensor<storage_backend_t<device>>);
      tensor_ptr->set_pinned(false);
    }
    return output;
  }
};

template <DALITensorDevice device>
struct BatchFactoryImpl<DALIOpType::SUPPORT, device> {
  static storage_gen_t<GetStorageIndex(DALIOpType::SUPPORT, device)> CreateOutputBatch(
      int batch_size) {
    DALI_ENFORCE(device == DALITensorDevice::CPU, "Only CPU outputs allowed");
    storage_gen_t<GetStorageIndex(DALIOpType::SUPPORT, device)> output(
        new Tensor<storage_backend_t<device>>);
    output->set_pinned(false);
    return output;
  }
};

template <DALIOpType op_type, DALITensorDevice device>
struct FillStorageOwner {
  storage_owner_t operator()(int batch_size) {
    storage_owner_t result;
    std::get<GetStorageIndex(op_type, device)>(result) =
        BatchFactoryImpl<op_type, device>::CreateOutputBatch(batch_size);
    return result;
  }
};

// inline storage_owner_t BatchFactory(DALIOpType op_type, DALITensorDevice device, int batch_size)
// {
//   storage_owner_t result;
//   // std::get<GetStorageIndex(op_type, device)>(result)
//   //     = BatchFactoryImpl<op_type, device>::CreateOutputBatch(batch_size);
//   VALUE_SWITCH(
//       op_type, op_type_static,
//       (DALIOpType::GPU, DALIOpType::CPU, DALIOpType::MIXED,
//        DALIOpType::SUPPORT),
//       (VALUE_SWITCH(
//           device, device_static, (DALITensorDevice::CPU, DALITensorDevice::GPU),
//           (std::get<GetStorageIndex(op_type_static, device_static)>(result) =
//                BatchFactoryImpl<op_type_static, device_static>::CreateOutputBatch(batch_size)),
//           DALI_FAIL("Unexpected device"))),
//       DALI_FAIL("Unexpected op_type"));
//   return result;
// }

/**
 * @brief Used to go over all cases of DALIOpType and DALITensorDevice and call appropriate
 * template based on runtime values of those enums
 *
 * @tparam ToExecute Functor class parametrized by DALIOpType and DALITensorDevice
 * @tparam Ret return type
 * @tparam T Arguments to operator() of `ToExecute` functor
 * @param op_type runtime value of DALIOpType used to chose Functor
 * @param device runtime value of DALITensorDevice used to chose Functor
 * @param args Runtime arguments to operator()
 * @return Ret
 */
template <template <DALIOpType, DALITensorDevice> class ToExecute, typename Ret, typename... T>
Ret Switch_OpType_Device(DALIOpType op_type, DALITensorDevice device, T &&... args) {
  VALUE_SWITCH(
      op_type, op_type_static,
      (DALIOpType::GPU, DALIOpType::CPU, DALIOpType::MIXED,
       DALIOpType::SUPPORT),
      (VALUE_SWITCH(device, device_static, (DALITensorDevice::CPU, DALITensorDevice::GPU),
                    (return ToExecute<op_type_static, device_static>{}(std::forward<T>(args)...);),
                    DALI_FAIL("Unexpected device"))),
      DALI_FAIL("Unexpected op_type"));
}

template <template <DALIOpType> class ToExecute, typename Ret, typename... T>
Ret Switch_OpType(DALIOpType op_type, T &&... args) {
  VALUE_SWITCH(op_type, op_type_static,
               (DALIOpType::GPU, DALIOpType::CPU, DALIOpType::MIXED,
                DALIOpType::SUPPORT),
               (return ToExecute<op_type_static>{}(std::forward<T>(args)...);),
               DALI_FAIL("Unexpected op_type"));
}

template <template <DALITensorDevice> class ToExecute, typename Ret, typename... T>
Ret Switch_Device(DALITensorDevice device, T &&... args) {
  VALUE_SWITCH(device, device_static, (DALITensorDevice::CPU, DALITensorDevice::GPU),
               (return ToExecute<device_static>{}(std::forward<T>(args)...);),
               DALI_FAIL("Unexpected device"));
}

inline storage_owner_t BatchFactory(DALIOpType op_type, DALITensorDevice device, int batch_size) {
  return Switch_OpType_Device<FillStorageOwner, storage_owner_t>(op_type, device, batch_size);
}

template <DALIOpType op_type, DALITensorDevice device>
auto get_storage(storage_owner_t& owner) -> decltype(std::get<GetStorageIndex(op_type, device)>(owner)) {
  std::get<GetStorageIndex(op_type, device)>(owner);
}

template <DALIOpType op_type, DALITensorDevice device>
typename std::tuple_element<GetStorageIndex(op_type, device), storage_owner_t>::type&
get_storage(storage_owner_t& owner) noexcept {
  return std::get<GetStorageIndex(op_type, device)>(owner);
}

template <DALIOpType op_type, DALITensorDevice device>
typename std::tuple_element<GetStorageIndex(op_type, device), storage_owner_t>::type&&
get_storage(storage_owner_t&& owner) noexcept {
  return std::get<GetStorageIndex(op_type, device)>(owner);
}

template <DALIOpType op_type, DALITensorDevice device>
typename std::tuple_element<GetStorageIndex(op_type, device), storage_owner_t>::type const&
get_storage(const storage_owner_t& owner) noexcept {
  return std::get<GetStorageIndex(op_type, device)>(owner);
}

template <DALIOpType op_type, DALITensorDevice device>
typename std::tuple_element<GetStorageIndex(op_type, device), storage_owner_t>::type const&&
get_storage(const storage_owner_t&& owner) noexcept {
  return std::get<GetStorageIndex(op_type, device)>(owner);
}


}  // namespace dali

#endif  // DALI_PIPELINE_WORKSPACE_WORKSPACE_DATA_FACTORY_H_