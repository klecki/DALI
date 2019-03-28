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

#ifndef DALI_PIPELINE_EXECUTOR_QUEUE_POLICY_H_
#define DALI_PIPELINE_EXECUTOR_QUEUE_POLICY_H_

#include <condition_variable>
#include <mutex>
#include <queue>

#include "dali/pipeline/executor/queue_metadata.h"

namespace dali {

// Policy that passes Queueing/Buffering indexes between stages, handling required synchronization
// struct QueuePolicy {
//   // Return sizes of stage queues based of Pipeline arguments
//   static StageQueues GetQueueSizes(QueueSizes init_sizes);
//   // Initialize the policy during Executor::Build();
//   void InitializeQueues(const StageQueues &stage_queue_depths);
//   // Acquire Queue indexes for given stage
//   QueueIdxs AcquireIdxs(OpType stage);
//   // Finish stage and release the indexes. Not called by the last stage, as it "returns" outputs
//   void ReleaseIdxs(OpType stage, QueueIdxs idxs);
//   // Called by the last stage - mark the Queue idxs as ready to be used as output
//   void QueueOutputIdxs(QueueIdxs idxs);
//   // Get the indexes of ready outputs and mark them as in_use by the user
//   OutputIdxs UseOutputIdxs();
//   // Release currently used output
//   void ReleaseOutputIdxs();
//   // Wake all waiting threads and skip further execution due to stop signaled
//   void SignalStop();
//   // Returns true if we signaled stop previously
//   bool IsStopSignaled();
// };


// Each stage requires ready buffers from previous stage and free buffers from current stage
struct UniformQueuePolicy {
  static StageQueues GetQueueSizes(QueueSizes init_sizes) {
    DALI_ENFORCE(init_sizes.cpu_size == init_sizes.gpu_size,
                 "Queue sizes should be equal for UniformQueuePolicy");
    return StageQueues(init_sizes.cpu_size);
  }

  void InitializeQueues(const StageQueues &stage_queue_depths) {
    DALI_ENFORCE(
        stage_queue_depths[OpType::CPU] == stage_queue_depths[OpType::MIXED] &&
            stage_queue_depths[OpType::MIXED] == stage_queue_depths[OpType::GPU],
        "This policy does not support splited queues");

    // All buffers start off as free
    for (int i = 0; i < stage_queue_depths[OpType::CPU]; ++i) {
      free_queue_.push(i);
    }
  }

  QueueIdxs AcquireIdxs(OpType stage) {
    if (stage == OpType::SUPPORT) {
      // Block until there is a free buffer to use
      std::unique_lock<std::mutex> lock(free_mutex_);
      free_cond_.wait(lock, [stage, this]() {
        return !free_queue_.empty() || stage_work_stop_[static_cast<int>(stage)];
      });
      if (stage_work_stop_[static_cast<int>(stage)]) {
        return QueueIdxs{-1};  // We return anything due to exec error
      }
      int queue_idx = free_queue_.front();
      free_queue_.pop();
      return QueueIdxs{queue_idx};
    }

    std::lock_guard<std::mutex> lock(stage_work_mutex_[static_cast<int>(stage)]);
    if (stage_work_stop_[static_cast<int>(stage)]) {
      return QueueIdxs{-1};
    }
    auto queue_idx = stage_work_queue_[static_cast<int>(stage)].front();
    stage_work_queue_[static_cast<int>(stage)].pop();
    return QueueIdxs{queue_idx};
  }

  void ReleaseIdxs(OpType stage, QueueIdxs idxs) {
    if (HasNextStage(stage)) {
      auto next_stage = NextStage(stage);
      std::lock_guard<std::mutex> lock(stage_work_mutex_[static_cast<int>(next_stage)]);
      stage_work_queue_[static_cast<int>(next_stage)].push(idxs[stage]);
    }
  }

  void QueueOutputIdxs(QueueIdxs idxs) {
    // We have to give up the elements to be occupied
    {
      std::lock_guard<std::mutex> lock(ready_mutex_);
      ready_queue_.push(idxs[OpType::GPU]);
    }
    ready_cond_.notify_all();
  }

  OutputIdxs UseOutputIdxs() {
    // Block until the work for a batch has been issued.
    // Move the queue id from ready to in_use
    std::unique_lock<std::mutex> lock(ready_mutex_);
    ready_cond_.wait(lock, [this]() {
      return !ready_queue_.empty() || ready_stop_;
    });
    if (ready_stop_) {
      return OutputIdxs{-1};
    }
    int output_idx = ready_queue_.front();
    ready_queue_.pop();
    in_use_queue_.push(output_idx);
    lock.unlock();
    return OutputIdxs{output_idx};
  }

  void ReleaseOutputIdxs() {
    // Mark the last in-use buffer as free and signal
    // to waiting threads
    // TODO(klecki): in_use_queue should be guarded, but we assume it is used only in synchronous
    // python calls
    if (!in_use_queue_.empty()) {
      {
        std::lock_guard<std::mutex> lock(free_mutex_);
        free_queue_.push(in_use_queue_.front());
        in_use_queue_.pop();
      }
      free_cond_.notify_one();
    }
  }

  void NotifyAll() {
    ready_cond_.notify_all();
    free_cond_.notify_all();
  }

  void SignalStop() {
    {
      std::lock_guard<std::mutex> lock(ready_mutex_);
      ready_stop_ = true;
    }
    for (int i = 0; i < static_cast<int>(OpType::COUNT); ++i) {
      std::lock_guard<std::mutex> l(stage_work_mutex_[i]);
      stage_work_stop_[i] = true;
    }
    NotifyAll();
  }

  bool IsStopSignaled() {
    // We only need to check the first one, since they're
    // always set in the same time
    std::lock_guard<std::mutex> l(ready_mutex_);
    return ready_stop_;
  }

 private:
  std::queue<int> ready_queue_, free_queue_, in_use_queue_;
  std::mutex ready_mutex_, free_mutex_;
  std::condition_variable ready_cond_, free_cond_;

  static const int kOpCount = static_cast<int>(OpType::COUNT);
  std::array<std::queue<int>, kOpCount> stage_work_queue_;
  std::array<std::mutex, kOpCount> stage_work_mutex_;
  // We use a dedicated stop flag for every mutex & condition_varialbe pair,
  // so when using them in cond_var predicate,
  // we know the changes are propagated properly and we won't miss a notify.
  std::array<bool, kOpCount> stage_work_stop_ = {{false, false, false, false}};
  bool ready_stop_ = false;
};

// Ready buffers from previous stage imply that we can process corresponding buffers from current
// stage
struct SeparateQueuePolicy {
  static StageQueues GetQueueSizes(QueueSizes init_sizes) {
    StageQueues result(0);
     // For non-uniform case we buffer for CPU x GPU pair.
    result[OpType::SUPPORT] = init_sizes.cpu_size * init_sizes.gpu_size;
    result[OpType::CPU] = init_sizes.cpu_size;
    // Mixed and GPU are bound together due to being outputs
    result[OpType::MIXED] = init_sizes.gpu_size;
    result[OpType::GPU] = init_sizes.gpu_size;
    return result;
  }

  void InitializeQueues(const StageQueues &stage_queue_depths) {
    for (int stage = 0; stage < static_cast<int>(OpType::COUNT); stage++) {
      for (int i = 0; i < stage_queue_depths[static_cast<OpType>(stage)]; i++) {
        stage_free_[stage].push(i);
      }
    }
  }

  QueueIdxs AcquireIdxs(OpType stage) {
    QueueIdxs result(0);
    // We dine with the philosophers

    int current_stage = static_cast<int>(stage);
    // We actually have a previous stage
    if (HasPreviousStage(stage)) {
      int previous_stage = static_cast<int>(PreviousStage(stage));
      std::unique_lock<std::mutex> ready_previous_lock(stage_ready_mutex_[previous_stage]);
      stage_ready_cv_[previous_stage].wait(ready_previous_lock, [previous_stage, this]() {
        return !stage_ready_[previous_stage].empty() || stage_ready_stop_[previous_stage];
      });
      if (stage_ready_stop_[previous_stage]) {
        return QueueIdxs{-1};
      }
      // We fill the information about all the previous stages herew
      result = stage_ready_[previous_stage].front();
      stage_ready_[previous_stage].pop();
      // We are the only ones waiting for the lock, so we do not try to wake anyone
    }
    // There always is a current stage
    {
      std::unique_lock<std::mutex> free_current_lock(stage_free_mutex_[current_stage]);
      stage_free_cv_[current_stage].wait(free_current_lock, [current_stage, this]() {
        return !stage_free_[current_stage].empty() || stage_free_stop_[current_stage];
      });
      if (stage_free_stop_[current_stage]) {
        return QueueIdxs{-1};
      }
      // We add info about current stage
      result[stage] = stage_free_[current_stage].front();
      stage_free_[current_stage].pop();
    }
    return result;
  }

  void ReleaseIdxs(OpType stage, QueueIdxs idxs) {
    int current_stage = static_cast<int>(stage);
    // Do not release
    // // We have a special case for Support ops - they are set free by a GPU stage,
    // // during QueueOutputIdxs
    // if (stage != OpType::CPU) {
    //   if (HasPreviousStage(stage)) {
    //     ReleaseStageIdx(PreviousStage(stage), idxs);
    //   }
    // }
    {
      std::lock_guard<std::mutex> ready_current_lock(stage_ready_mutex_[current_stage]);
      // stage_ready_[current_stage].push(idxs[stage]);
      // Store the idxs up to the point of stage that we processed
      stage_ready_[current_stage].push(idxs);
    }
    stage_ready_cv_[current_stage].notify_one();
  }

  void QueueOutputIdxs(QueueIdxs idxs) {
    {
      std::lock_guard<std::mutex> ready_output_lock(ready_output_mutex_);
      ready_output_queue_.push(idxs);
    }
    ready_output_cv_.notify_all();

    // In case of GPU we release also the Support Op
    // ReleaseStageIdx(OpType::SUPPORT, idxs);
    // ReleaseStageIdx(OpType::CPU, idxs);
  }

  OutputIdxs UseOutputIdxs() {
    // Block until the work for a batch has been issued.
    // Move the queue id from ready to in_use
    std::unique_lock<std::mutex> ready_lock(ready_output_mutex_);
    ready_output_cv_.wait(ready_lock, [this]() {
      return !ready_output_queue_.empty() || ready_stop_;
    });
    if (ready_stop_) {
      return OutputIdxs{-1, -1};
    }
    auto output_idx = ready_output_queue_.front();
    ready_output_queue_.pop();
    // TODO(klecki): in_use_queue should be guarded, but we assume it is used only in synchronous
    // python calls
    in_use_queue_.push(output_idx);
    ready_lock.unlock();
    return {output_idx[OpType::MIXED], output_idx[OpType::GPU]};
  }

  void ReleaseOutputIdxs() {
    // Mark the last in-use buffer as free and signal
    // to waiting threads
    if (!in_use_queue_.empty()) {
      auto mixed_idx = static_cast<int>(OpType::MIXED);
      auto gpu_idx = static_cast<int>(OpType::GPU);
      // TODO(klecki): in_use_queue should be guarded, but we assume it is used only in synchronous
      // python calls
      auto processed = in_use_queue_.front();
      in_use_queue_.pop();
      ReleaseStageIdx(OpType::SUPPORT, processed);
      ReleaseStageIdx(OpType::CPU, processed);
      ReleaseStageIdx(OpType::MIXED, processed);
      ReleaseStageIdx(OpType::GPU, processed);
      // {
      //   std::lock_guard<std::mutex> lock(stage_free_mutex_[mixed_idx]);
      //   stage_free_[mixed_idx].push(processed.mixed);
      // }
      // stage_free_cv_[mixed_idx].notify_one();
      // {
      //   std::lock_guard<std::mutex> lock(stage_free_mutex_[gpu_idx]);
      //   stage_free_[gpu_idx].push(processed.gpu);
      // }
      // stage_free_cv_[gpu_idx].notify_one();
    }
  }

  void NotifyAll() {
    ready_output_cv_.notify_all();
    free_cond_.notify_all();
    for (int i = 0; i < static_cast<int>(OpType::COUNT); ++i) {
      stage_ready_cv_[i].notify_all();
      stage_free_cv_[i].notify_all();
    }
  }

  void SignalStop() {
    {
      std::lock_guard<std::mutex> lock(ready_output_mutex_);
      ready_stop_ = true;
    }
    for (int i = 0; i < static_cast<int>(OpType::COUNT); ++i) {
      {
        std::lock_guard<std::mutex> l(stage_free_mutex_[i]);
        stage_free_stop_[i] = true;
      }
      {
        std::lock_guard<std::mutex> l(stage_ready_mutex_[i]);
        stage_ready_stop_[i] = true;
      }
    }
    NotifyAll();
  }

  bool IsStopSignaled() {
    // We only need to check the first one, since they're
    // always set in the same time
    std::lock_guard<std::mutex> l(ready_output_mutex_);
    return ready_stop_;
  }

 private:
  void ReleaseStageIdx(OpType stage, QueueIdxs idxs) {
    auto released_stage = static_cast<int>(stage);
    // We release the consumed buffer
    {
      std::lock_guard<std::mutex> free_lock(stage_free_mutex_[released_stage]);
      stage_free_[released_stage].push(idxs[stage]);
    }
    // We freed buffer, so we notfiy the released stage it can continue it's work
    stage_free_cv_[released_stage].notify_one();
  }

  static const int kOpCount = static_cast<int>(OpType::COUNT);
  // For syncing free and ready buffers between stages
  std::array<std::mutex, kOpCount> stage_free_mutex_;
  std::array<std::mutex, kOpCount> stage_ready_mutex_;
  // We use a dedicated stop flag for every mutex & condition_varialbe pair,
  // so when using them in cond_var predicate,
  // we know the changes are propagated properly and we won't miss a notify.
  std::array<bool, kOpCount> stage_free_stop_ = {{false, false, false, false}};
  std::array<bool, kOpCount> stage_ready_stop_ = {{false, false, false, false}};
  bool ready_stop_ = false;
  std::array<std::condition_variable, kOpCount> stage_free_cv_;
  std::array<std::condition_variable, kOpCount> stage_ready_cv_;

  // Buffers are rotated between being 'free', where the
  // pipeline is ok to fill them with data, 'ready', where
  // they are already full of prepared data, and 'in-use',
  // where the user currently owns that buffer. A buffer
  // is marked as in-use when it is returned as and output.
  // The buffer is then returned the the ready queue the
  // next time Ouputs() is called.
  std::array<std::queue<int>, kOpCount> stage_free_;
  std::array<std::queue<QueueIdxs>, kOpCount> stage_ready_;

  std::condition_variable ready_output_cv_, free_cond_;
  // Output ready and in_use mutexes and queues
  std::mutex ready_output_mutex_, in_use_mutex_;

  std::queue<QueueIdxs> ready_output_queue_;
  std::queue<QueueIdxs> in_use_queue_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_EXECUTOR_QUEUE_POLICY_H_
