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

#ifndef DALI_PIPELINE_OP_GRAPH_H_
#define DALI_PIPELINE_OP_GRAPH_H_

#include <map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <string>
#include <memory>
#include <fstream>
#include <set>

#include "dali/common.h"
#include "dali/error_handling.h"
#include "dali/pipeline/operators/operator.h"

namespace dali {

using OpNodeId = int64_t;
using OpPartitionId = int64_t;
using TensorNodeId = int64_t;
// using producer_edge_t = std::pair<OpNodeId, Index>;
// using consumer_edge_t = std::pair<OpNodeId, Index>;


// What device is this tensor stored on
enum class DALITensorDevice {
  CPU = 0,
  GPU = 1,
};

struct OpNode {
  inline OpNode() {}
  virtual ~OpNode() = default;
  OpNode& operator=(const OpNode&) = delete;

  OpNode(OpNode &&) = default;
  OpNode& operator=(OpNode &&) = default;

  inline OperatorBase &InstantiateOperator() {
    if (!op) op = dali::InstantiateOperator(spec);
    return *op;
  }

  std::unique_ptr<OperatorBase> op;
  OpNodeId id;
  OpSpec spec;
  std::set<OpNodeId> parents, children;

  // parent and children tensors indexed by our inputs and outputs
  std::vector<TensorNodeId> parent_tensors, children_tensors;

  // parent ops indexed by our inputs,
  // parent_ops[i] === parent_tensors[i].producer_edge.node;
  // std::vector<OpNodeId> parent_ops;

  std::string instance_name;
};

// Stores meta-data about a tensor and how it
// is used by a producer/consumer node.
struct TensorMeta {
  OpNodeId node;
  TensorNodeId tensor;
  Index index;
  DALITensorDevice storage_device;
  bool is_support;
};

using producer_edge_t = TensorMeta;
using consumer_edge_t = TensorMeta;

// Second type of graph nodes.
struct TensorNode {
  TensorNodeId id;
  std::string name; // TODO(klecki): not happy about all the strings
  producer_edge_t producer_edge;
  // order of consumers is arbitrary
  std::vector<consumer_edge_t> consumer_edges;
};


/**
 * @brief Stores all meta-data about a graph of operations to be run
 * keeps track of useful meta-data about consumers/producers of
 * different intermediates.
 *
 * Operators in the graph have a global OpNodeId that is assigned in
 * the order ops are added to the graph. Operators also have an
 * index within the set of ops of its type (cpu, mixed, gpu).
 * This enables us to iterate over select portions of the graph, or
 * the entire graph.
 */
class DLL_PUBLIC OpGraph {
 public:
  DLL_PUBLIC inline OpGraph() {
    node_partitions_.resize(static_cast<int>(DALIOpType::DALI_OP_TYPE_COUNT));
  }
  DLL_PUBLIC inline ~OpGraph() = default;

  /**
   * @brief Adds an op with the input specification to the graph.
   */
  DLL_PUBLIC void AddOp(const OpSpec &spec, const std::string& name);

  /**
   * @brief Removes the node with the specified OpNodeId from
   * the graph. Fails if the removal would produce an invalid
   * graph.
   */
  DLL_PUBLIC void RemoveOp(OpNodeId id);

  /**
   * @brief Returns the total number of ops in the graph.
   */
  DLL_PUBLIC inline Index NumOp() const {
    return NumCPUOp() + NumGPUOp() + NumMixedOp() + NumSupportOp();
  }

  DLL_PUBLIC inline Index NumOp(DALIOpType op_type) const {
    return node_partitions_[static_cast<int>(op_type)].size();
  }

  /**
   * @brief Returns the number of cpu ops in the graph.
   */
  DLL_PUBLIC inline Index NumCPUOp() const { return NumOp(DALIOpType::DALI_CPU); }

  /**
   * @brief Returns the number of gpu ops in the graph.
   */
  DLL_PUBLIC inline Index NumGPUOp() const { return NumOp(DALIOpType::DALI_GPU); }

  /**
   * @brief Returns the number of mixed ops in the graph.
   */
  DLL_PUBLIC inline Index NumMixedOp() const { return NumOp(DALIOpType::DALI_MIXED); }

  /**
   * @brief Returns the number of support ops in the graph.
   */
  DLL_PUBLIC inline Index NumSupportOp() const { return NumOp(DALIOpType::DALI_SUPPORT); }

  /**
   * @brief Returns the unique NodeId for partition_id among nodes of op_type
   */
  DLL_PUBLIC inline OpNodeId NodeId(OpPartitionId partition_id, DALIOpType op_type) {
    DALI_ENFORCE_VALID_INDEX(partition_id, NumOp(op_type));
    return node_partitions_[static_cast<int>(op_type)][partition_id];
  }

  // TODO(klecki) return a copy/const& to disallow modification
  DLL_PUBLIC inline OpNode& Node(OpPartitionId partition_id, DALIOpType op_type) {
    auto node_id = NodeId(partition_id, op_type);
    return op_nodes_[node_id];
  }

  DLL_PUBLIC inline std::pair<DALIOpType, OpPartitionId> PartitionId(OpNodeId op_id) {
    // TODO(klecki) return {nodes_[op_id].op_type} ??
    return id_to_node_map_[op_id];
  }

  /**
   * @brief Returns the node object for the `idx`-th cpu op that
   * was added to the graph.
   */
  DLL_PUBLIC inline OpNode& cpu_node(Index idx) {
    return Node(idx, DALIOpType::DALI_CPU);
  }

  /**
   * @brief Returns the node object for the `idx`-th gpu op that
   * was added to the graph.
   */
  DLL_PUBLIC inline OpNode& gpu_node(Index idx) {
    return Node(idx, DALIOpType::DALI_GPU);
  }

  /**
   * @brief Returns the node object for the `idx`-th mixed op that
   * was added to the graph.
   */
  DLL_PUBLIC inline OpNode& mixed_node(Index idx) {
    return Node(idx, DALIOpType::DALI_MIXED);
  }

  /**
   * @brief Returns the node object for the `idx`-th support op that
   * was added to the graph.
   */
  DLL_PUBLIC inline OpNode& support_node(Index idx) {
    return Node(idx, DALIOpType::DALI_SUPPORT);
  }

  /**
   * @brief Returns the graph node with the given name.
   * This function is much slower than the version taking
   * index as argument so should not be used in performance
   * critical section of the code.
   */
  DLL_PUBLIC OpNode& Node(const std::string& name);

  /**
   * @brief Returns the graph node with the given index in the graph.
   */
  DLL_PUBLIC OpNode& Node(OpNodeId id) {
    DALI_ENFORCE_VALID_INDEX(id, op_nodes_.size());
    return op_nodes_[id];
  }

  /**
   * @brief Returns the graph node with the given index in the graph.
   */
  DLL_PUBLIC const OpNode& Node(OpNodeId id) const {
    DALI_ENFORCE_VALID_INDEX(id, op_nodes_.size());
    return op_nodes_[id];
  }

  DLL_PUBLIC TensorNode& Tensor(TensorNodeId id) {
    DALI_ENFORCE_VALID_INDEX(id, tensor_nodes_.size());
    return tensor_nodes_[id];
  }

  DLL_PUBLIC const TensorNode& Tensor(TensorNodeId id) const {
    DALI_ENFORCE_VALID_INDEX(id, tensor_nodes_.size());
    return tensor_nodes_[id];
  }

  // DLL_PUBLIC OpNode& ProdNode(TensorNodeId id) {
  //   auto prod_id = Tensor(id).producer_edge.node;
  //   return op_nodes_[prod_id];
  // }

  /**
   * @brief Returns the type (cpu, gpu, mixed) of the node
   * at the given index.
   */
  DLL_PUBLIC inline DALIOpType NodeType(OpNodeId id) const {
    DALI_ENFORCE_VALID_INDEX(id, (Index)id_to_node_map_.size());
    return id_to_node_map_[id].first;
  }

  /**
   * @brief Returns the index of the node with the specified id
   * among nodes of its type.
   */
  DLL_PUBLIC inline Index NodeIdx(OpNodeId id) const {
    DALI_ENFORCE_VALID_INDEX(id, (Index)id_to_node_map_.size());
    return id_to_node_map_[id].second;
  }

  /**
   * @brief Returns the TensorMeta objects for the tensor
   * with the given name and its producer node.
   */
  DLL_PUBLIC inline TensorMeta TensorSourceMeta(const string &name) const {
    auto it = tensor_name_to_id_.find(name);
    DALI_ENFORCE(it != tensor_name_to_id_.end(), "Tensor with name \"" +
        name + "\" has no known source.");
    return tensor_nodes_[it->second].producer_edge;
  }

  /**
   * @brief Checks if given Tensor already exists in the graph
   */
  DLL_PUBLIC inline bool TensorExists(const string &name) {
    auto it = tensor_name_to_id_.find(name);
    return it != tensor_name_to_id_.end();
  }

  /**
   * @brief Returns the id of the op that produces the tensor with
   * the given name.
   */
  DLL_PUBLIC inline OpNodeId TensorSourceID(const string &name) {
    return TensorSourceMeta(name).node;
  }

  /**
   * @brief Returns the output idx of the input tensor in
   * its source.
   */
  DLL_PUBLIC inline Index TensorIdxInSource(const string &name) {
    return TensorSourceMeta(name).index;
  }

  /**
   * @brief Returns true if the tensor with the given name
   * has a backend type that matches the calling type.
   */
  template <typename Backend>
  DLL_PUBLIC bool TensorIsType(const string &name);

  /**
   * @brief Returns a vector of meta-data about the nodes that
   * consume the tensor with the input name.
   */
  DLL_PUBLIC inline vector<TensorMeta> TensorConsumerMeta(const string &name) const {
    auto it = tensor_name_to_id_.find(name);
    if (it == tensor_name_to_id_.end()) {
      // If we have no entries for this tensors consumers,
      // we just return an empty vector
      return vector<TensorMeta>{};
    }
    return tensor_nodes_[it->second].consumer_edges;
  }

  /**
   * @brief Helper function for saving graph to DOT file
   */
  DLL_PUBLIC void GenerateDOTFromGraph(const OpNode& current_node, std::ofstream& ofs) {
    if (current_node.children.empty()
        || visited_nodes_.find(current_node.id) != visited_nodes_.end()) {
      ofs << current_node.instance_name << "\n";
      return;
    }
    visited_nodes_.insert(current_node.id);
    for (auto node_id : current_node.children) {
        ofs << current_node.instance_name;
        ofs << " -> ";
        OpNode& child_node = Node(node_id);
        GenerateDOTFromGraph(child_node, ofs);
    }
  }

  /**
   * @brief Instantiates the operators based on OpSpecs in nodes
   */
  DLL_PUBLIC void InstantiateOperators();

  /**
   * @brief Save graph in DOT directed graph format
   * in filename.
   */
  DLL_PUBLIC void SaveToDotFile(const string filename) {
    std::ofstream ofs(filename);
    ofs << "digraph graphname {\n";
    const OpNode& current_node = Node(0);
    GenerateDOTFromGraph(current_node, ofs);
    ofs << "}\n";
    visited_nodes_.clear();
  }

  DISABLE_COPY_MOVE_ASSIGN(OpGraph);

 private:

  OpNode& PlaceNewOp(DALIOpType op_type, OpSpec op_spec, std::string instance_name);
  TensorNode& PlaceNewTensor();

  // vector<OpNode> cpu_nodes_;
  // vector<OpNode> gpu_nodes_;
  // vector<OpNode> mixed_nodes_;
  // vector<OpNode> support_nodes_;

  std::vector<OpNode> op_nodes_;
  std::vector<TensorNode> tensor_nodes_;
  std::vector<std::vector<OpPartitionId>> node_partitions_;

  // Overwrite target_id TensorNode with source_id TensorNode fixing all references to
  // TensorNode source_id.
  // target_id tensor will be invalidate and cannot be used
  // void OverwriteTensorNode(TensorNodeId source_id, TensorNodeId target_id);
  void SwapTensorNodes(TensorNodeId left_id, TensorNodeId right_id);

  void RemoveTensorNode(TensorNodeId id);

  void OverwriteOpNode(OpNodeId source_id, OpNodeId target_id);
  void SwapOpNodes(OpNodeId left_id, OpNodeId right_id);
  void RemoveOpNode(OpNodeId id);

  // Stores a mapping from NodeIDs to a pair where the first
  // element indicates what type of node it is,  and the second
  // is the index of the op within the specified vector.
  vector<std::pair<DALIOpType, Index>> id_to_node_map_;

  // std::map<string, TensorMeta> tensor_producers_;
  // std::map<string, vector<TensorMeta>> tensor_consumers_;

  std::map<std::string, TensorNodeId> tensor_name_to_id_;

  // For the graph traversal
  std::unordered_set<OpNodeId> visited_nodes_;
  TensorNodeId next_tensor_id_ = 0;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OP_GRAPH_H_
