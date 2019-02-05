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

#include <algorithm>

#include "dali/pipeline/op_graph.h"

#include "dali/pipeline/operators/op_schema.h"

namespace dali {

namespace {

bool AllInputsCPU(const OpSpec &spec) {
  for (int i = 0; i < spec.NumInput(); ++i) {
    if (spec.InputDevice(i) == "gpu") return false;
  }
  return true;
}

bool AllOutputsCPU(const OpSpec &spec) {
  for (int i = 0; i < spec.NumOutput(); ++i) {
    if (spec.OutputDevice(i) == "gpu") return false;
  }
  return true;
}

bool AllOutputsGPU(const OpSpec &spec) {
  for (int i = 0; i < spec.NumOutput(); ++i) {
    if (spec.OutputDevice(i) == "cpu") return false;
  }
  return true;
}

void CheckOpConstraints(const OpSpec &spec) {
  const OpSchema &schema = SchemaRegistry::GetSchema(spec.name());

  bool allows_multiple_inputs = schema.AllowsMultipleInputSets();
  const int additional_outputs = schema.CalculateAdditionalOutputs(spec);

  int num_input_sets = 1;
  if (allows_multiple_inputs) {
    num_input_sets = spec.GetArgument<int>("num_input_sets");
  } else {
    DALI_ENFORCE(spec.GetArgument<int>("num_input_sets") == 1,
        "Op '" + spec.name() + "' does not support multiple input sets.");
  }

  DALI_ENFORCE(schema.SupportsInPlace(spec) || !spec.GetArgument<bool>("inplace"),
      "Op '" + spec.name() + "' does not support in-place execution.");
  DALI_ENFORCE(spec.NumRegularInput() <= num_input_sets * schema.MaxNumInput(),
      "Operator '" + spec.name() +
      "' supports a maximum of " + std::to_string(schema.MaxNumInput()) + " inputs, "
      "but was passed " + std::to_string(spec.NumRegularInput()) + ".");
  DALI_ENFORCE(spec.NumRegularInput() >= num_input_sets * schema.MinNumInput(),
      "Operator '" + spec.name() +
      "' supports a minimum of " + std::to_string(schema.MinNumInput()) + " inputs, "
      "but was passed " + std::to_string(spec.NumRegularInput()) + ".");
  DALI_ENFORCE(spec.NumOutput() == schema.CalculateOutputs(spec) + additional_outputs,
      "Operator '" + spec.name() + "' supports "
      + std::to_string((schema.CalculateOutputs(spec) + additional_outputs)/num_input_sets)
      + " outputs, but was passed " + std::to_string(spec.NumOutput()/num_input_sets) + ".");
}


DALIOpType DeviceStringToOpType(std::string device) {
  if (device == "gpu") {
    return DALIOpType::DALI_GPU;
  } else if (device == "cpu") {
    return DALIOpType::DALI_CPU;
  } else if (device == "mixed") {
    return DALIOpType::DALI_MIXED;
  } else if (device == "support") {
    return DALIOpType::DALI_SUPPORT;
  }
  DALI_FAIL("Unsupported device type: " + device + ".");
}

DALITensorDevice DeviceStringToTensorStorage(std::string inout_device) {
  if (inout_device == "cpu") {
    return DALITensorDevice::CPU;
  }
  return DALITensorDevice::GPU;
}

}  // namespace

OpNode& OpGraph::PlaceNewOp(DALIOpType op_type, OpSpec op_spec, std::string instance_name) {
  op_nodes_.push_back(OpNode());
  auto &node = op_nodes_.back();
  node.id = op_nodes_.size() - 1;
  node.spec = op_spec;
  node.instance_name = std::move(instance_name);
  node.op_type = op_type;
  auto new_partition_id = NumOp(op_type); //node_partitions_[op_type].size();
  node.partition_index = new_partition_id;
  node_partitions_[static_cast<int>(op_type)].push_back(node.id);
  // id_to_node_map_.push_back({op_type, new_partition_id});
  return node;
}

TensorNode& OpGraph::PlaceNewTensor() {
  tensor_nodes_.push_back(TensorNode());
  tensor_nodes_.back().id = tensor_nodes_.size() - 1;
  return tensor_nodes_.back();
}

void OpGraph::AddOp(const OpSpec &spec, const std::string& name) {
  // Validate the op specification
  CheckOpConstraints(spec);

  string device = spec.GetArgument<string>("device");
  auto op_type = DeviceStringToOpType(device);
  switch (op_type) {
    case DALIOpType::DALI_CPU: {
      // Enforce graph constraints
      DALI_ENFORCE(AllInputsCPU(spec), "CPU ops cannot receive GPU input data.");
      DALI_ENFORCE(AllOutputsCPU(spec), "CPU ops can only produce CPU output data.");
      break;
    }
    case DALIOpType::DALI_GPU: {
      break;
    }
    case DALIOpType::DALI_MIXED: {
      // Enforce graph constraints
      DALI_ENFORCE(AllInputsCPU(spec), "Mixed ops cannot receive GPU input data.");
      break;
    }
    case DALIOpType::DALI_SUPPORT: {
      // Enforce graph constraints
      DALI_ENFORCE(AllInputsCPU(spec), "Support ops cannot receive GPU input data.");
      break;
    }
    default:
      DALI_FAIL("Invalid device argument \"" + device +
          "\". Valid options are \"cpu\", \"gpu\" or \"mixed\"");
      break;
  }
  // Add node meta-data and add to the list of nodes
  auto &new_node = PlaceNewOp(op_type, spec, name);

  // Setup references between nodes. We require that the
  // ops are added to the graph in a topological ordering.
  // This loop will verify this by ensuring that all inputs
  // to the new op have already been created.
  for (int i = 0; i < spec.NumInput(); ++i) {
    // Get the tensor id we are consuming by its name
    auto it = tensor_name_to_id_.find(spec.Input(i));
    DALI_ENFORCE(it != tensor_name_to_id_.end(), "Tensor with name \"" +
        name + "\" has no known source.");
    auto consumed_tensor_id = it->second;

    // Add parent node id, checks if parent node exists in graph
    auto parent_id = tensor_nodes_[consumed_tensor_id].producer_edge.node;

    // Note: We don't care if the parent has already
    // been added to this nodes set of parents, so
    // we don't check the return value.
    new_node.parents.insert(parent_id);
    // new_node.parent_ops.push_back(parent_id);

    // Add new node as child
    auto &parent_node = this->Node(parent_id);
    parent_node.children.insert(new_node.id);

    // Place it as a parent tensor
    new_node.parent_tensors.push_back(consumed_tensor_id);

    // Update the consumer info for this tensor
    TensorMeta meta;
    meta.node = new_node.id;
    meta.tensor = consumed_tensor_id;
    meta.index = i;
    meta.is_support = spec.IsArgumentInput(i);
    meta.storage_device = DeviceStringToTensorStorage(spec.InputDevice(i));

    // Insert new tensor consumer
    tensor_nodes_[consumed_tensor_id].consumer_edges.push_back(meta);
  }

  // Mark this op as the source of its output tensors
  // Create TensorNodes for outputs and respective edges for this OpNode -> TensorNodes
  for (int i = 0; i < spec.NumOutput(); ++i) {
    // Set the producer info for this tensor
    TensorMeta meta;
    meta.node = new_node.id;
    meta.index = i;
    meta.is_support = spec.GetArgument<string>("device") == "support";
    meta.storage_device = DeviceStringToTensorStorage(spec.OutputDevice(i));

    string name = spec.Output(i);

    // Place new Tensor with producer info and add edge to OpNode.
    auto &new_tensor = PlaceNewTensor();
    meta.tensor = new_tensor.id;
    new_tensor.producer_edge = meta;
    new_tensor.name = name;
    new_node.children_tensors.push_back(new_tensor.id);

    auto it_inserted = tensor_name_to_id_.insert({name, new_tensor.id});
    DALI_ENFORCE(it_inserted.second, "Operator '" + spec.name() +
        "' has output with name " + name + ", but output "
        "with this name already exists as output of op '" +
        this->Node(TensorSourceID(name)).spec.name() + "'");
  }
}

void OpGraph::InstantiateOperators() {
  // traverse devices by topological order (support, cpu, mixed, gpu)
  DALIOpType order[] = { DALIOpType::DALI_SUPPORT, DALIOpType::DALI_CPU,
                         DALIOpType::DALI_MIXED,   DALIOpType::DALI_GPU };

  for (auto op_type : order) {
    for (auto op_id : node_partitions_[static_cast<int>(op_type)]) {
      op_nodes_[op_id].InstantiateOperator();
    }
  }
}

void OpGraph::SwapTensorNodes(TensorNodeId left_id, TensorNodeId right_id) {
  auto &left = tensor_nodes_[left_id];
  auto &right = tensor_nodes_[right_id];
  // Change ids in producers (there is only one), and on edges
  {
    auto &left_prod = Node(left.producer_edge.node);
    left_prod.children_tensors[left.producer_edge.index] = right_id;
    left.producer_edge.tensor = right_id;
    auto &right_prod = Node(right.producer_edge.node);
    right_prod.children_tensors[right.producer_edge.index] = left_id;
    right.producer_edge.tensor = left_id;
  }
  // Change ids in consumers
  {
    for (auto &cons_edge : left.consumer_edges) {
      auto &cons = Node(cons_edge.node);
      cons.parent_tensors[cons_edge.index] = right_id;
      cons_edge.tensor = right_id;
    }
    for (auto &cons_edge : right.consumer_edges) {
      auto &cons = Node(cons_edge.node);
      cons.parent_tensors[cons_edge.index] = left_id;
      cons_edge.tensor = left_id;
    }
  }
  // Clean up names mapping
  tensor_name_to_id_[left.name] = right_id;
  tensor_name_to_id_[right.name] = left_id;
  // Swap the actual nodes
  std::swap(left, right);
}

void OpGraph::RemoveTensorNode(TensorNodeId id) {
  DALI_ENFORCE_VALID_INDEX(id, (Index)tensor_nodes_.size());
  DALI_ENFORCE(tensor_nodes_[id].consumer_edges.empty(), "Removed tensor cannot have any consumers.");
  // Swap it out
  for (TensorNodeId i = id + 1; i < (int)tensor_nodes_.size(); i++) {
    // Move from i to i - 1
    SwapTensorNodes(i, i - 1);
  }
  // We remove the last element
  tensor_nodes_.pop_back();
  // There is no option to remove from positional array of tensor produced by parent op
}

void OpGraph::SwapOpNodes(OpNodeId left_id, OpNodeId right_id) {
  auto &left = op_nodes_[left_id];
  auto &right = op_nodes_[right_id];
  // Swap all references in tensor edges
  // Produced tensors (children)
  {
    auto &tensor_nodes_ref = tensor_nodes_;
    auto swap_ids_in_parent_tensor = [&tensor_nodes_ref](OpNode &node, OpNodeId new_id) {
      for (auto tid : node.children_tensors) {
        auto &tensor = tensor_nodes_ref[tid];
        // edges from node to tensor
        tensor.producer_edge.node = new_id;
      }
    };
    swap_ids_in_parent_tensor(left, right_id);
    swap_ids_in_parent_tensor(right, left_id);
  }
  // Consumed tensors (parents). As we can have overlapping parents, we do this in two steps
  // otherwise we could overwrite twice.
  {
    auto &tensor_nodes_ref = tensor_nodes_;
    auto swap_ids_in_child_tensor = [&tensor_nodes_ref](OpNode &node, OpNodeId old_id, OpNodeId new_id) {
      for (auto tid : node.parent_tensors) {
        auto &tensor = tensor_nodes_ref[tid];
        // edges from tensor to node
        for (auto &edge : tensor.consumer_edges) {
          if (edge.node == old_id) {
            edge.node = new_id;
          }
        }
      }
    };
    constexpr OpNodeId dummy_id = -1;
    swap_ids_in_child_tensor(left, left_id, dummy_id);
    swap_ids_in_child_tensor(right, right_id, left_id);
    swap_ids_in_child_tensor(left, dummy_id, right_id);
  }
  // Swap all references in parent and children ops
  {
    auto &op_nodes_ref = op_nodes_;
    auto remove_id_in_family_op = [&op_nodes_ref](OpNode &node, OpNodeId old_id) {
      for (auto oid : node.parents) {
        op_nodes_ref[oid].children.erase(old_id);
      }
      for (auto oid : node.children) {
        op_nodes_ref[oid].parents.erase(old_id);
      }
    };
    auto add_id_in_family_op = [&op_nodes_ref](OpNode &node, OpNodeId new_id) {
      for (auto oid : node.parents) {
        op_nodes_ref[oid].children.insert(new_id);
      }
      for (auto oid : node.children) {
        op_nodes_ref[oid].parents.insert(new_id);
      }
    };
    remove_id_in_family_op(left, left_id);
    remove_id_in_family_op(right, right_id);
    add_id_in_family_op(left, right_id);
    add_id_in_family_op(right, left_id);
  }

  std::swap(left, right);
}

void OpGraph::RemoveOpNode(OpNodeId id) {
  DALI_ENFORCE_VALID_INDEX(id, (Index)op_nodes_.size());
  auto &target_op = op_nodes_[id];
  DALI_ENFORCE(target_op.children.empty(), "Overwritten ops cannot have any children.");
  DALI_ENFORCE(target_op.children_tensors.empty(),
               "All produced tensors should be removed before removing op"
               " and list of children tensors should be invalidated.");
  for (OpNodeId i = id + 1; i < (int)op_nodes_.size(); i++) {
    // Move from i to i - 1
    SwapOpNodes(i - 1, i);
  }
  // assume that we removed one element
  op_nodes_.pop_back();
}

template <typename T>
void RemoveVectorElement(T& vector, typename T::iterator it) {
  std::swap(*it, vector.back());
  vector.pop_back();
}


// TODO: adjust for unified node indexing
// Op Removal Process:
// 1. Validate we can remove it (it has no children & no consumers for produced tensors)
// 2. Remove tensors it produces
// 3. Remove the OpNode
// 4. Correct the partitions as we changed the ids while removing OpNode
void OpGraph::RemoveOp(OpNodeId id) {
  OpNode &target = this->Node(id);

  // If the node has any children, we cannot remove it
  DALI_ENFORCE(target.children.empty(), "Node '" + target.spec.name() +
      "' has " + std::to_string(target.children.size()) +
      ". Cannot remove");
  for (auto t : target.children_tensors) {
    DALI_ENFORCE(tensor_nodes_[t].consumer_edges.empty(), "Node '" + target.spec.name() +
      "' produces a tensor that has " + std::to_string(tensor_nodes_[t].consumer_edges.size()) +
      " consumers. Cannot remove");
  }

  // Remove all tensors produced by this node and invalidate list of children tensors
  for (auto t : target.children_tensors) {
    RemoveTensorNode(t);
  }
  target.children_tensors.clear();

  // TODO(klecki): not sure if we can consume some tensors more than once in one op,
  // so we would need to remove it few times
  for (auto t : target.parent_tensors) {
    auto &sibling_consumers = tensor_nodes_[t].consumer_edges;
    auto result = std::find_if(sibling_consumers.begin(), sibling_consumers.end(), [id](TensorMeta &meta)  {
      return meta.node == id;
    });
    RemoveVectorElement(sibling_consumers, result);
  }

  RemoveOpNode(id);
  Repartition();
}

void OpGraph::Repartition() {
  for (auto & p : node_partitions_) {
    p.clear();
  }
  for (auto &node : op_nodes_) {
    auto new_partition_id = NumOp(node.op_type); //node_partitions_[op_type].size();
    node.partition_index = new_partition_id;
    node_partitions_[static_cast<int>(node.op_type)].push_back(node.id);
  }
}

// TODO(klecki)
OpNode& OpGraph::Node(const std::string& name) {
  for (auto &node : op_nodes_) {
    if (node.instance_name == name) {
      return node;
    }
  }
  DALI_FAIL("Operator node with name " + name + " not found.");
}

template <>
bool OpGraph::TensorIsType<CPUBackend>(const string &name) {
  return TensorSourceMeta(name).storage_device == DALITensorDevice::CPU;
}

template <>
bool OpGraph::TensorIsType<GPUBackend>(const string &name) {
  return TensorSourceMeta(name).storage_device == DALITensorDevice::GPU;
}

}  // namespace dali
