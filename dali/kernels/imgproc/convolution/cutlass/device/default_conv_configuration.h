/***************************************************************************************************
 * Copyright (c) 2017-2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Definitions for GEMM structures
*/


#ifndef DALI_KERNELS_IMGPROC_CONVOLUTION_CUTLASS_DEVICE_DEFAULT_CONV_CONFIGURATION_H_
#define DALI_KERNELS_IMGPROC_CONVOLUTION_CUTLASS_DEVICE_DEFAULT_CONV_CONFIGURATION_H_

#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/arch/wmma.h"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"

#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/thread/linear_combination_clamp.h"
#include "cutlass/gemm/device/default_gemm_configuration.h"
#include "cutlass/gemm/gemm.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace device {

////////////////////////////////////////////////////////////////////////////////

template <typename OperatorClass, typename ArchTag, typename ElementA, typename ElementB,
          typename ElementC, typename ElementAccumulator>
struct DefaultConvConfiguration;

////////////////////////////////////////////////////////////////////////////////

template <typename OperatorClass, typename ArchTag, typename ElementA, typename ElementB,
          typename ElementC, typename ElementAccumulator>
struct DefaultConvConfiguration {
  using UnderlyingConv = DefaultGemmConfiguration<OperatorClass, ArchTag, ElementA, ElementB,
                                                  ElementC, ElementAccumulator>;

  static int const kAlignmentA = UnderlyingConv::kAlignmentA;
  static int const kAlignmentB = UnderlyingConv::kAlignmentB;

  using ThreadblockShape = typename UnderlyingConv::ThreadblockShape;
  using WarpShape = typename UnderlyingConv::WarpShape;
  using InstructionShape = typename UnderlyingConv::InstructionShape;
  static int const kStages = UnderlyingConv::kStages;

  using EpilogueOutputOp = typename UnderlyingConv::EpilogueOutputOp;

  using Operator = typename UnderlyingConv::Operator;
};

template <typename ArchTag, typename ElementA, typename ElementB, typename ElementC,
          typename ElementAccumulator>
struct DefaultConvConfiguration<arch::OpClassSimt, ArchTag, ElementA, ElementB, ElementC,
                                ElementAccumulator> {
  static int const kAlignmentA = 1;
  static int const kAlignmentB = 1;
  using ThreadblockShape = GemmShape<128, 128, 8>;
  using WarpShape = GemmShape<32, 64, 8>;
  using InstructionShape = GemmShape<1, 1, 1>;
  static int const kStages = 2;

  using EpilogueOutputOp =
      epilogue::thread::LinearCombination<ElementC, 1, ElementAccumulator, ElementAccumulator>;

  using Operator = arch::OpMultiplyAdd;
};

template <typename ElementA, typename ElementB, typename ElementC, typename ElementAccumulator>
struct DefaultConvConfiguration<arch::OpClassTensorOp, arch::Sm70, ElementA, ElementB, ElementC,
                                ElementAccumulator> {
  // This requires alignment of inputs and outputs
  // TODO(klecki): Maybe we can limit alignment to the conv matrix and see how it works
  static int const kAlignmentA = 128 / sizeof_bits<ElementA>::value;
  static int const kAlignmentB = 128 / sizeof_bits<ElementB>::value;

  using ThreadblockShape = GemmShape<128, 256, 32>;
  using WarpShape = GemmShape<64, 64, 32>;
  using InstructionShape = GemmShape<8, 8, 4>;
  static int const kStages = 2;

  using EpilogueOutputOp =
      epilogue::thread::LinearCombination<ElementC, 128 / sizeof_bits<ElementC>::value,
                                          ElementAccumulator, ElementAccumulator>;

  using Operator = arch::OpMultiplyAdd;
};

////////////////////////////////////////////////////////////////////////////////
}  // namespace device
}  // namespace gemm
}  // namespace cutlass

////////////////////////////////////////////////////////////////////////////////


#endif  // DALI_KERNELS_IMGPROC_CONVOLUTION_CUTLASS_DEVICE_DEFAULT_CONV_CONFIGURATION_H_