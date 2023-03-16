// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <c10/util/hash.h>
#include <mma_type.h>
#include <scheduler/heuristic.h>
#include <utils.h>
#include <functional>

#include <sstream>
#include "type.h"

namespace nvfuser {

// Parameters of the matmul heuristic to describe the optimial schedule.
class MatmulParams : public HeuristicParams {
 public:
  struct DoubleBufferOptions {
    bool double_buffer_smem_write = false;
    bool double_buffer_smem_read = false;
    int smem_double_buffer_stage = 2;

    bool operator==(const DoubleBufferOptions& other) const {
      return other.double_buffer_smem_write == double_buffer_smem_write &&
          other.double_buffer_smem_read == double_buffer_smem_read &&
          other.smem_double_buffer_stage == smem_double_buffer_stage;
    }

    std::string toString() const {
      std::stringstream ss;
      ss << "DoubleBufferOptions:\n"
         << "  double_buffer_smem_write: "
         << (double_buffer_smem_write ? "true" : "false") << "\n"
         << "  double_buffer_smem_read: "
         << (double_buffer_smem_read ? "true" : "false") << "\n"
         << "  smem_double_buffer_stage: " << smem_double_buffer_stage;
      return ss.str();
    }

    size_t hash() const {
      return (std::hash<size_t>{}(static_cast<size_t>(double_buffer_smem_write))
              << 0) ^
          (std::hash<size_t>{}(static_cast<size_t>(double_buffer_smem_read))
           << 1) ^
          (std::hash<size_t>{}(static_cast<size_t>(smem_double_buffer_stage))
           << 2);
    };
  };

  //! Whether to rotate the ldmatrix out of the main loop
  bool rotate_ldmatrix_out_of_main_loop = true;

  //! (Ampere+) Use cp.async to load operands.
  bool async_gmem_load_operands = false;

  //! Specifies the tiling hierarchy on block,
  //!  warp, and instruction levels.
  MatMulTileOptions tile_sizes = {};

  //! Specify the type of MMA op to be used in generated kernel.
  MmaOptions::MacroType mma_op = MmaOptions::MacroType::NoMMA;

  //! Specify the input layout of input tensors.
  MmaOptions::MmaInputLayout layout =
      static_cast<MmaOptions::MmaInputLayout>(-1);

  //! Specify which tensor we double buffer.
  DoubleBufferOptions double_buffer_options = {};

  std::string toString() const override {
    std::stringstream ss;
    ss << "\n===== Matmul Parameters ========\n"
       << (tag == "" ? "" : "Tag: ") << tag << "\n"
       << "MMA op: " << nvfuser::toString(mma_op, true) << "\n"
       << "Layout: " << nvfuser::toString(layout) << "\n"
       << double_buffer_options.toString() << "\n"
       << nvfuser::toString(tile_sizes) << "\n"
       << "Rotate ldmatrix out of main loop: "
       << (rotate_ldmatrix_out_of_main_loop ? "true" : "false") << "\n"
       << "Async global mem load: "
       << (async_gmem_load_operands ? "true" : "false") << "\n"
       << "Indexing mode: "
       << (cparams.index_type.has_value()
               ? (cparams.index_type.value() == PrimDataType::Int ? "int64_t"
                                                                  : "int32_t")
               : "unavailable")
       << "\n"
       << "====================================\n";
    return ss.str();
  }

  size_t hash() const override {
    const size_t attr_hash = (nvfuser::hash(mma_op) << 0) //
        ^ (nvfuser::hash(layout) << 1) //
        ^ (double_buffer_options.hash() << 2) //
        ^ (nvfuser::hash(tile_sizes) << 3) //
        ^ (std::hash<size_t>{}(rotate_ldmatrix_out_of_main_loop) << 4) //
        ^ (std::hash<size_t>{}(async_gmem_load_operands) << 5);
    return attr_hash;
  }

  bool sameAs(
      const std::shared_ptr<HeuristicParams>& other_base) const override {
    auto other_casted = std::dynamic_pointer_cast<MatmulParams>(other_base);
    if (other_casted == nullptr) {
      return false;
    }

    if (other_casted->layout != layout || //
        other_casted->mma_op != mma_op || //
        other_casted->async_gmem_load_operands != async_gmem_load_operands || //
        other_casted->rotate_ldmatrix_out_of_main_loop !=
            rotate_ldmatrix_out_of_main_loop) {
      return false;
    }
    if (!(other_casted->tile_sizes == tile_sizes) || //
        !(other_casted->double_buffer_options == double_buffer_options)) {
      return false;
    }

    return true;
  }

  std::shared_ptr<HeuristicParams> clone() const override {
    return std::make_shared<MatmulParams>(*this);
  }
};

} // namespace nvfuser
