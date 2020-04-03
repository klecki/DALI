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

#ifndef DALI_PIPELINE_OPERATOR_OP_SCHEMA_H_
#define DALI_PIPELINE_OPERATOR_OP_SCHEMA_H_

#include <functional>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "dali/core/common.h"
#include "dali/core/copy_vector_helper.h"
#include "dali/core/format.h"
#include "dali/core/traits.h"
#include "dali/core/error_handling.h"
#include "dali/pipeline/data/types.h"
#include "dali/pipeline/operator/argument.h"

namespace dali {

class OpSpec;

struct RequiredArgumentDef {
  std::string doc;
  DALIDataType dtype;
  std::string shape;
};

struct DefaultedArgumentDef {
  std::string doc;
  DALIDataType dtype;
  Value *default_value;
  std::string shape;
};

class DLL_PUBLIC OpSchema {
 public:
  typedef std::function<int(const OpSpec &spec)> SpecFunc;

  OpSchema(OpSchema &&) = default;
  OpSchema(const OpSchema &) = delete;
  OpSchema &operator=(const OpSchema &) = delete;
  OpSchema &operator=(OpSchema &&) = default;

  DLL_PUBLIC explicit inline OpSchema(const std::string &name): name_(name) {
    // Fill internal arguments
    AddInternalArg("num_threads", "Number of CPU threads in a thread pool", -1);
    AddInternalArg("batch_size", "Batch size", -1);
    AddInternalArg("device", "Device on which the Op is run", std::string("cpu"));
    AddInternalArg("inplace", "Whether Op can be run in place", false);
    AddInternalArg("default_cuda_stream_priority", "Default cuda stream priority", 0);

    AddOptionalArg("seed", "Random seed (If not provided it will be populated based "
      "on the global seed of the pipeline)", -1);

    AddOptionalArg("bytes_per_sample_hint", "Output size hint (bytes), "
      "per sample. The memory will be preallocated if it uses GPU or page-locked memory", 0);

    AddOptionalArg("preserve", "Do not remove the Op from the "
                               "graph even if its outputs are unused.", false);
  }



  DLL_PUBLIC inline ~OpSchema() = default;

  /**
   * @brief Returns the name of this operator.
   */
  DLL_PUBLIC inline const std::string& name() const {
    return name_;
  }

  /**
   * @brief Sets the doc string for this operator.
   */
  DLL_PUBLIC inline OpSchema& DocStr(const string &dox) {
    dox_ = dox;
    return *this;
  }

  /**
   * @brief Allows to set a docstring for __call__ method of Operator.
   *
   * The first line of the string can contain the signature that will be used
   * in the sphinx-generated documentation, for example:
   * "__call__(input0, input1, optional_input = None, **kwargs)\n"
   *
   * The doc should follow numpydoc format, and can either contain
   * `Args`, `Returns` sections or they can be provided separately using InputDocStr
   * and OutputDocStr respectively.
   *
   * If the `append_kwargs_section` is true, the docstring generator will append the Keyword args
   * section at the end of this doc
   *
   * @param doc
   * @param append_kwargs_section
   */
  DLL_PUBLIC OpSchema &CallDocStr(const std::string &doc, bool append_kwargs_section = true);

  /**
   * @brief Sets the docstring for input.
   *
   * Set the documentation for intput at given `index`.
   *
   * If the operator specifies some range of allowed inputs with NumInput(int min, int max)
   * only the first `min` inputs are considered mandatory, the rest are optional
   *
   * Python bindings will generate entry in `Args` section using numpydoc style:
   * name: type_doc
   *     doc
   */
  DLL_PUBLIC OpSchema &InputDoc(int index, const string &name, const string &type_doc,
                                const string &doc);

  /**
   * @brief Sets the docstring for output.
   *
   * Set the documentation for output at given `index`.
   *
   * Works only with operators that have static number of outputs, that is
   * use only the NumOutput() and not OutputFn() or AdditionalOutputFn().
   *
   * name can be empty
   *
   * Python bindings will generate entry in `Returns` section using numpydoc style:
   * name: type_doc
   *     doc
   */
  DLL_PUBLIC OpSchema &OutputDoc(int index, const string &name, const string &type_doc,
                                 const string &doc);

  /**
   * @brief Set the `Args` section of `__call__` doc explicitly
   *
   * Incompatible with InputDoc.
   *
   * The arguments should be described using numpydoc syntax,
   * with comments indented by 4 spaces, for example:
   * """
   * input0: sample shape: [X, Y], element_type
   *     This is the first input
   * input1: sample shape: any shape, element_type
   *     This is second input
   * optional_input:  sample shape: [X], element_type, optional
   *     This is optional input
   * """
   *
   */
  DLL_PUBLIC OpSchema &InputDocStr(const std::string &doc);

  /**
   * @brief Set the `Returns` section of `__call__` doc explicitly
   *
   * Incompatible with OutputDoc.
   *
   * The arguments should be described using numpydoc syntax,
   * with comments indented by 4 spaces, for example:
   * """
   * output0: TensorList of type
   *     This is the first output
   * output1: TensorList of some kind
   *     This is second output
   * optional_output: TensorList, optional
   *     This is optional output
   * """
   */
  DLL_PUBLIC OpSchema &OutputDocStr(const std::string &doc);

  /**
   * @brief Sets a function that infers the number of outputs this
   * op will produce from the ops specification. This is required
   * to expose the op to the python interface.
   *
   * If the ops has a fixed number of outputs, this function
   * does not need to be added to the schema
   */
  DLL_PUBLIC inline OpSchema& OutputFn(SpecFunc f) {
    output_fn_ = std::move(f);
    return *this;
  }

  /**
   * @brief Sets a function to determine the number of
   * additional outputs (independent of output sets) from an
   * op from the op's specification.
   *
   * If this function is not set it will be assumed that no
   * additional outputs can be returned
   *
   * Use case is to expose additional information (such as random
   * numbers used within operators) to the user
   */
  DLL_PUBLIC inline OpSchema& AdditionalOutputsFn(SpecFunc f) {
    additional_outputs_fn_ = std::move(f);
    return *this;
  }

  /**
   * @brief Sets the number of inputs that the op can receive.
   */
  DLL_PUBLIC inline OpSchema& NumInput(int n) {
    DALI_ENFORCE(n >= 0);
    max_num_input_ = n;
    min_num_input_ = n;
    input_dox_.resize(n);
    input_layouts_.resize(n);
    return *this;
  }

  /**
   * @brief Sets the min and max number of inputs the op can receive.
   */
  DLL_PUBLIC inline OpSchema& NumInput(int min, int max) {
    DALI_ENFORCE(min <= max);
    DALI_ENFORCE(min >= 0);
    DALI_ENFORCE(max >= 0);
    min_num_input_ = min;
    max_num_input_ = max;
    input_layouts_.resize(max);
    input_dox_.resize(max);
    return *this;
  }

  /**
   * @brief Sets the number of outputs that the op can receive.
   */
  DLL_PUBLIC inline OpSchema& NumOutput(int n) {
    DALI_ENFORCE(n >= 0);
    num_output_ = n;
    output_dox_.resize(n);
    return *this;
  }

  /**
   * @brief Indicates that this operator should not use auto-generated documentation
   *        of inputs and `__call__` operator with custom signature.
   */
  DLL_PUBLIC inline OpSchema& DisableAutoInputDox() {
    disable_auto_input_dox_ = true;
    return *this;
  }

  /**
   * @brief Indicates that multiple instances of this operator cannot share a logical ID to achieve
   * uniform processing of multiple input sets
   */
  DLL_PUBLIC inline OpSchema& DisallowInstanceGrouping() {
    allow_instance_grouping_ = false;
    return *this;
  }

  /**
   * @brief Notes that this operator expects sequence inputs exclusively
   */
  DLL_PUBLIC inline OpSchema& SequenceOperator() {
    is_sequence_operator_ = true;
    return *this;
  }

  /**
   * @brief Notes that sequences can be used with this op
   */
  DLL_PUBLIC inline OpSchema& AllowSequences() {
    allow_sequences_ = true;
    return *this;
  }

  /**
   * Notes that the operator can process 3D data.
   * @return
   */
  DLL_PUBLIC inline OpSchema& SupportVolumetric() {
    support_volumetric_ = true;
    return *this;
  }

  /**
   * @brief Notes that this operator is internal to DALI backend (and shouldn't be exposed in Python API)
   */
  DLL_PUBLIC inline OpSchema& MakeInternal() {
    is_internal_ = true;
    return *this;
  }

  /**
   * @brief Notes that this operator is deprecated and optionally specifies the operator to be used instead
   */
  DLL_PUBLIC inline OpSchema& Deprecate(const std::string &in_favor_of) {
    is_deprecated_ = true;
    deprecated_in_favor_of_ = in_favor_of;
    return *this;
  }

  /**
   * @brief Notes that this operator cannot be serialized
   */
  DLL_PUBLIC inline OpSchema& Unserializable() {
    serializable_ = false;
    return *this;
  }

  /**
   * @brief Adds a required argument to op with its type
   */

  /**
   * @brief  Adds a required argument to op with its type
   *
   * @param s Argument name
   * @param doc docstring for given argument
   * @param dtype type of the argument (used for docstring generation)
   * @param enable_tensor_input If given argument accepts tensors, which means it is also
   *                            an Argument Input.
   * @param shape If this is Argument Input optional shape can be provided.
   */
  DLL_PUBLIC inline OpSchema& AddArg(const std::string &s,
                                     const std::string &doc,
                                     const DALIDataType dtype,
                                     bool enable_tensor_input = false,
                                     const std::string &shape = "") {
    DALI_ENFORCE(enable_tensor_input || shape.empty(),
                 make_string("Shape cannot be provided for required argument ", s,
                             " which doesn't accept tensor inputs."));
    CheckArgument(s);
    arguments_[s] = {doc, dtype, shape};
    if (enable_tensor_input) {
      tensor_arguments_.insert(s);
    }
    return *this;
  }

  /**
   * @brief Sets input layout constraints and default for given input.
   *
   * At run-time, when the operator encounters a tensor(list) with specified
   * layout, but different than one provided to this function, error is raised.
   *
   * If the input tensor has no layout, the one provided to this function is assumed
   * if number of dimensions matches. Otherswise, error is raised.
   */
  DLL_PUBLIC inline OpSchema& InputLayout(int index, TensorLayout layout) {
    return InputLayout(index, { layout });
  }

  /**
   * @brief Sets input layout constraints and default for given input.
   *
   * At run-time, when the operator encounters a tensor(list) with specified
   * layout, but not one of those provided to this function, error is raised.
   *
   * If the input tensor has no layout, the layouts specified by call to this function
   * are searched for the first matching the number of dimensions of the input -
   * it will be the default value for this input. If number of dimensions doesn't
   * match any of the layouts provided here, an error is raised.
   */
  DLL_PUBLIC inline OpSchema& InputLayout(int index, std::initializer_list<TensorLayout> layouts) {
    CheckInputIndex(index);
    DALI_ENFORCE(input_layouts_[index].empty(), "Layouts for input " + std::to_string(index) +
                 " already specified");
    for (auto &l : layouts) {
      DALI_ENFORCE(!l.empty(), "Cannot specify an empty layout for an input");
    }
    input_layouts_[index] = layouts;
    return *this;
  }

  /**
   * @brief Sets input layout constraint and default for all inputs.
   * @see InputLayout(int index, TensorLayout layout)
   */
  DLL_PUBLIC inline OpSchema& InputLayout(TensorLayout layout) {
    return InputLayout({ layout });
  }

  /**
   * @brief Sets input layout constraint and default for all inputs.
   * @see InputLayout(int index, TensorLayout layout)
   */
  DLL_PUBLIC inline OpSchema& InputLayout(std::initializer_list<TensorLayout> layouts) {
    for (int i = 0; i < max_num_input_; i++)
      InputLayout(i, layouts);
    return *this;
  }

  /**
   * @brief Verifies that the layout is valid for given input index and number of dimensions
   *        or returns a default layout if the layout parameter is empty.
   */
  DLL_PUBLIC inline const TensorLayout &GetInputLayout(int index, int sample_ndim,
                                                       const TensorLayout &layout = {}) const {
    CheckInputIndex(index);
    DALI_ENFORCE(layout.empty() || layout.ndim() == sample_ndim, make_string(
      "The layout '", layout, "' is not valid for ", sample_ndim, "-dimensional tensor"));

    if (input_layouts_[index].empty()) {
      return layout;
    }

    if (layout.empty()) {
      for (auto &l : input_layouts_[index])
        if (l.ndim() == sample_ndim)
          return l;
      std::stringstream ss;
      ss << "The number of dimensions " << sample_ndim << " does not match any of the allowed"
        " layouts for input " << index << ". Valid layouts are:\n";
      for (auto &l : input_layouts_[index])
        ss << l.c_str() << "\n";
      DALI_FAIL(ss.str());
    } else {
      for (auto &l : input_layouts_[index])
        if (l == layout)
          return l;
      std::stringstream ss;
      ss << "The layout \"" << layout << "\" does not match any of the allowed"
        " layouts for input " << index << ". Valid layouts are:\n";
      for (auto &l : input_layouts_[index])
        ss << l.c_str() << "\n";
      DALI_FAIL(ss.str());
    }
  }

  DLL_PUBLIC inline const std::vector<TensorLayout> &GetSupportedLayouts(int input_idx) const {
    CheckInputIndex(input_idx);
    return input_layouts_[input_idx];
  }

  /**
   * @brief Adds an optional non-vector argument without default to op
   */
  template <typename T>
  DLL_PUBLIC inline OpSchema& AddOptionalArg(const std::string &s,
                                     const std::string &doc,
                                     std::nullptr_t,
                                     bool enable_tensor_input = false,
                                     const std::string &shape = "") {
    CheckArgument(s);
    optional_arguments_[s] = {doc, type2id<T>::value, nullptr, shape};
    if (enable_tensor_input) {
      tensor_arguments_.insert(s);
    }
    return *this;
  }


  /**
   * @brief Adds an optional non-vector argument to op
   */
  template <typename T>
  DLL_PUBLIC inline std::enable_if_t<!is_vector<T>::value && !is_std_array<T>::value, OpSchema&>
  AddOptionalArg(const std::string &s,
                 const std::string &doc,
                 T default_value,
                 bool enable_tensor_input = false,
                 const std::string &shape = "") {
    CheckArgument(s);
    auto to_store = Value::construct(default_value);
    optional_arguments_[s] = {doc, type2id<T>::value, to_store.get(), shape};
    optional_arguments_unq_.push_back(std::move(to_store));
    if (enable_tensor_input) {
      tensor_arguments_.insert(s);
    }
    return *this;
  }

  DLL_PUBLIC inline OpSchema &AddOptionalArg(const std::string &s,
                                             const std::string &doc,
                                             const char *default_value) {
    return AddOptionalArg(s, doc, std::string(default_value), false);
  }

  /**
   * @brief Adds an optional vector argument to op
   */
  template <typename T>
  DLL_PUBLIC inline OpSchema& AddOptionalArg(const std::string &s, const std::string &doc,
                                             std::vector<T> default_value,
                                             bool enable_tensor_input = false) {
    CheckArgument(s);
    using S = argument_storage_t<T>;
    auto to_store = Value::construct(detail::convert_vector<S>(default_value));
    optional_arguments_[s] = {doc, type2id<std::vector<T>>::value, to_store.get()};
    optional_arguments_unq_.push_back(std::move(to_store));
    if (enable_tensor_input) {
      tensor_arguments_.insert(s);
    }
    return *this;
  }

  /**
   * @brief Sets a function that infers whether the op can
   * be executed in-place depending on the ops specification.
   */
  DLL_PUBLIC inline OpSchema& InPlaceFn(SpecFunc f) {
    (void)f;
    REPORT_FATAL_PROBLEM("In-place op support not yet implemented.");
    return *this;
  }

  /**
   * @brief Sets a parent (which could be used as a storage of default parameters)
   * Does not support cyclic dependency. There can be multiple parents
   * and the lookup is transitive.
   * Only arguments are inherited, inputs and outputs are not.
   */
  DLL_PUBLIC inline OpSchema& AddParent(const std::string &parentName) {
    parents_.push_back(parentName);
    return *this;
  }

  DLL_PUBLIC inline OpSchema& SetName(const std::string &name) {
    name_ = name;
    return *this;
  }

  /**
   * @brief Notes that this operator should not be pruned from
   * a graph even if its outputs are unused.
   */
  DLL_PUBLIC inline OpSchema& NoPrune() {
    no_prune_ = true;
    return *this;
  }

  /**
   * @brief Informs that the data passes though this operator unchanged, only
   *        the metadata is affected.
   *
   * If the operator _can_ pass an input buffer as-is to the output (possibly
   * changing the associated metadata), this property of should be set accordingly
   * in the operator's OpSchema. The purpose of this property is to inform the
   * pipeline and the executor that a particular output of the operator doesn't
   * own the storage and the associated input should be included in double-buffering
   * whenever the output should.
   *
   * @param inout - tells which inputs are passed through to which outputs.
   *                Multiple inputs can be passed through to one output (at
   *                least potentially, e.g. when conditionally forwarding
   *                one of inputs to the output), but not vice versa.
   */
  DLL_PUBLIC inline OpSchema &PassThrough(const std::map<int, int> &inout) {
    passthrough_map_ = inout;
    return *this;
  }

  DLL_PUBLIC inline const vector<std::string>& GetParents() const {
    return parents_;
  }

  DLL_PUBLIC string DocStr() const;

  /**
   * @brief Return true wether the default input docs can be used
   */
  DLL_PUBLIC bool CanUseAutoInputDoc() {
    return !disable_auto_input_dox_ && MaxNumInput() <= 1;
  }

  DLL_PUBLIC bool AppendKwargsSection() {
    return append_kwargs_section_;
  }

  /**
   * @brief Return true when `__call__` docstring was explicitly set
   *
   * Should be considered as highest preference
   */
  DLL_PUBLIC bool HasCallDocStr() {
    return !call_dox_str_.empty();
  }

  DLL_PUBLIC std::string GetCallDocStr() {
    DALI_ENFORCE(HasCallDocStr(), "__call__ docstring was not set");
    return call_dox_str_;
  }


  DLL_PUBLIC bool HasInputDocStr() {
    return !input_dox_str_.empty();
  }

  DLL_PUBLIC std::string GetInputDocStr() {
    DALI_ENFORCE(HasCallDocStr(), "Input docstring was not set");
    return input_dox_str_;
  }

  DLL_PUBLIC bool HasOutputDocStr() {
    return !output_dox_str_.empty();
  }

  DLL_PUBLIC std::string GetOutputDocStr() {
    DALI_ENFORCE(HasOutputDocStr(), "Output docstring was not set");
    return output_dox_str_;
  }

  /**
   * @brief Check if this operator has per-input docstrings provided
   */
  DLL_PUBLIC bool HasPerInputDoc() {
    return input_dox_set_;
  }

  /**
   * @brief Check if this operator has per-output docstrings provided
   */
  DLL_PUBLIC bool HasPerOutputDoc() {
    return output_dox_set_;
  }

  /**
   * @brief List all the inputs that should appear in `__call__` signature based on the input
   *        docs that were specified. Requires HasPerInputDoc() to return true
   *
   */
  DLL_PUBLIC std::string GetCallSignatureInputs();

  struct InOutDoc {
    std::string name = {};
    std::string type_doc = {};
    std::string doc = {};
  };

  /**
   * @brief Get the docs for given input, must be specified with InputDoc
   */
  DLL_PUBLIC InOutDoc GetPerInputDoc(int input_idx);

  /**
   * @brief Get the docs for given output, must be specified with OutputDoc
   */
  DLL_PUBLIC InOutDoc GetPerOutputDoc(int output_idx);

  DLL_PUBLIC inline int MaxNumInput() const {
    return max_num_input_;
  }

  DLL_PUBLIC inline int MinNumInput() const {
    return min_num_input_;
  }

  DLL_PUBLIC inline int NumOutput() const {
    return num_output_;
  }

  DLL_PUBLIC inline bool AllowsAutoInputDox() const {
    return allow_instance_grouping_;
  }

  DLL_PUBLIC inline bool AllowsInstanceGrouping() const {
    return allow_instance_grouping_;
  }

  DLL_PUBLIC inline bool IsSequenceOperator() const {
    return is_sequence_operator_;
  }

  DLL_PUBLIC inline bool AllowsSequences() const {
    return allow_sequences_;
  }

  DLL_PUBLIC inline bool SupportsVolumetric() const {
    return support_volumetric_;
  }

  DLL_PUBLIC inline bool IsInternal() const {
    return is_internal_;
  }

  DLL_PUBLIC inline bool IsDeprecated() const {
    return is_deprecated_;
  }

  DLL_PUBLIC inline const std::string& DeprecatedInFavorOf() const {
    return deprecated_in_favor_of_;
  }

  DLL_PUBLIC inline bool HasOutputFn() const {
    return static_cast<bool>(output_fn_);
  }

  DLL_PUBLIC inline bool IsNoPrune() const {
    return no_prune_;
  }

  DLL_PUBLIC inline bool IsSerializable() const {
    return serializable_;
  }

  /**
   * @brief Returns the index of the output to which the input is passed.
   * @return Output index or -1 if given input is not passed through.
   */
  DLL_PUBLIC inline int GetPassThroughOutputIdx(int input_idx) const {
    auto it = passthrough_map_.find(input_idx);
    if (it == passthrough_map_.end())
      return -1;
    return it->second;
  }

  DLL_PUBLIC inline bool HasPassThrough() const {
    return !passthrough_map_.empty();
  }

  DLL_PUBLIC int CalculateOutputs(const OpSpec &spec) const;

  DLL_PUBLIC int CalculateAdditionalOutputs(const OpSpec &spec) const {
    if (!additional_outputs_fn_) return 0;
    return additional_outputs_fn_(spec);
  }

  DLL_PUBLIC inline bool SupportsInPlace(const OpSpec &spec) const {
    if (!in_place_fn_) return false;
    return in_place_fn_(spec);
  }

  DLL_PUBLIC void CheckArgs(const OpSpec &spec) const;

  /**
   * @brief Get default value of optional or internal argument. The default value must be declared
   */
  template<typename T>
  DLL_PUBLIC inline T GetDefaultValueForArgument(const std::string &s) const;

  DLL_PUBLIC bool HasRequiredArgument(const std::string &name, const bool local_only = false) const;

  DLL_PUBLIC bool HasOptionalArgument(const std::string &name, const bool local_only = false) const;

  DLL_PUBLIC bool HasInternalArgument(const std::string &name, const bool local_only = false) const;
  /**
   * @brief Finds default value for a given argument
   * @return A pair of the defining schema and the value
   */
  DLL_PUBLIC std::pair<const OpSchema *, const Value *>
  FindDefaultValue(const std::string &arg_name,
                   bool local_only = false,
                   bool include_internal = true) const;

  DLL_PUBLIC inline bool HasArgument(const std::string &name, bool include_internal = false) const {
    return HasRequiredArgument(name) || HasOptionalArgument(name)
        || (include_internal && HasInternalArgument(name, true));
  }

  /**
   * @brief Get docstring for operator argument of given name (Python Operator Kwargs).
   */
  DLL_PUBLIC std::string GetArgumentDox(const std::string &name) const;

  /**
   * @brief Get enum representing type of argument of given name.
   */
  DLL_PUBLIC DALIDataType GetArgumentType(const std::string &name) const;

  /**
   * @brief Get string representing a shape of argument input of given name.
   *
   * If the input is not accepting tensor inputs (argument input) the returned string is empty.
   */
  DLL_PUBLIC std::string GetArgumentShape(const std::string &name) const;

  /**
   * @brief Check if the argument has a default value.
   *        Required arguments always return false.
   *        Internal arguments always return true.
   */
  DLL_PUBLIC bool HasArgumentDefaultValue(const std::string &name) const;

  /**
   * @brief Get default value of optional argument represented as python-compatible repr string.
   *        Not allowed for internal arguments.
   */
  DLL_PUBLIC std::string GetArgumentDefaultValueString(const std::string &name) const;

  /**
   * @brief Get names of all required and optional arguments
   */
  DLL_PUBLIC std::vector<std::string> GetArgumentNames() const;
  DLL_PUBLIC bool IsTensorArgument(const std::string &name) const;

 private:
  inline void CheckArgument(const std::string &s) {
    DALI_ENFORCE(!HasArgument(s),
                 "Argument \"" + s + "\" already added to the schema");
    DALI_ENFORCE(internal_arguments_.find(s) == internal_arguments_.end(),
                 "Argument name \"" + s + "\" is reserved for internal use");
  }

  inline void CheckInputIndex(int index) const {
    DALI_ENFORCE(index >= 0 && index < max_num_input_,
                 make_string("Input index `", index, "` out of range [0..", max_num_input_,
                             "). Was NumInput called?"));
  }

  inline void CheckOutputIndex(int index) const {
    DALI_ENFORCE(index >= 0 && index < num_output_,
                 make_string("Output index `", index, "` out of range [0..", num_output_,
                             "). Was NumOutput called?"));
  }

  /**
   * @brief Add internal argument to schema. It always has a value.
   */
  template <typename T>
  void AddInternalArg(const std::string &name, const std::string &doc, T value) {
    auto v = Value::construct(value);
    internal_arguments_[name] = {doc, type2id<T>::value, v.get()};
    internal_arguments_unq_.push_back(std::move(v));
  }

  std::map<std::string, RequiredArgumentDef> GetRequiredArguments() const;


  std::map<std::string, DefaultedArgumentDef> GetOptionalArguments() const;

  string dox_;
  string name_;

  bool disable_auto_input_dox_ = false;

  std::vector<InOutDoc> input_dox_ = {};
  std::vector<InOutDoc> output_dox_ = {};
  bool input_dox_set_ = false;
  bool output_dox_set_ = false;

  // Custom docstring, if not empty should be used in place of input_dox_ descriptions
  std::string call_dox_str_ = {};

  // Custom docstrings for `Args` and `Returns` sections
  std::string input_dox_str_ = {};
  std::string output_dox_str_ = {};

  // Whether to append kwargs section to __call__ docstring. On by default,
  // can be turned off for call_dox_str_ specified manually
  bool append_kwargs_section_ = true;

  SpecFunc output_fn_, in_place_fn_, additional_outputs_fn_;

  int min_num_input_ = 0, max_num_input_ = 0;
  int num_output_ = 0;

  bool allow_instance_grouping_ = true;
  vector<string> parents_;

  bool support_volumetric_ = false;

  bool allow_sequences_ = false;
  bool is_sequence_operator_ = false;

  bool is_internal_ = false;

  bool no_prune_ = false;

  bool serializable_ = true;

  std::map<int, int> passthrough_map_;

  bool is_deprecated_ = false;
  string deprecated_in_favor_of_;

  std::map<std::string, RequiredArgumentDef> arguments_;
  std::map<std::string, DefaultedArgumentDef> optional_arguments_;
  std::map<std::string, DefaultedArgumentDef> internal_arguments_;
  std::vector<std::unique_ptr<Value> > optional_arguments_unq_;
  std::vector<std::unique_ptr<Value> > internal_arguments_unq_;
  std::vector<std::vector<TensorLayout>> input_layouts_;

  std::set<std::string> tensor_arguments_;
};

class SchemaRegistry {
 public:
  DLL_PUBLIC static OpSchema& RegisterSchema(const std::string &name);
  DLL_PUBLIC static const OpSchema& GetSchema(const std::string &name);
  DLL_PUBLIC static const OpSchema* TryGetSchema(const std::string &name);

 private:
  inline SchemaRegistry() {}

  DLL_PUBLIC static std::map<string, OpSchema>& registry();
};

template<typename T>
inline T OpSchema::GetDefaultValueForArgument(const std::string &s) const {
  const Value *v = FindDefaultValue(s, false, true).second;
  DALI_ENFORCE(v != nullptr,
               make_string("The argument \"", s, "\" doesn't have a default value in schema \"",
                           name(), "\"."));

  const ValueInst<T> * vT = dynamic_cast<const ValueInst<T>*>(v);
  DALI_ENFORCE(vT != nullptr, "Unexpected type of the default value for argument \"" + s +
        "\" of schema \"" + this->name() + "\"");
  return vT->Get();
}

#define DALI_SCHEMA_REG(OpName)      \
  int DALI_OPERATOR_SCHEMA_REQUIRED_FOR_##OpName() {        \
    return 42;                                              \
  }                                                         \
  static ::dali::OpSchema* ANONYMIZE_VARIABLE(OpName) =             \
    &::dali::SchemaRegistry::RegisterSchema(#OpName)

#define DALI_SCHEMA(OpName)                            \
      DALI_SCHEMA_REG(OpName)

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATOR_OP_SCHEMA_H_

