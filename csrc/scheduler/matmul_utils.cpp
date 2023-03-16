// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <scheduler/matmul_heuristic.h>
#include <scheduler/matmul_utils.h>
#include <scheduler/registry.h>

// NOTE: included to avoid compilation error caused by missing destructor in
// 'SchedulerRuntimeInfo'
#include <executor_utils.h>
#include <iostream>
#include <type_traits>
#include <utility>
#include "ATen/cuda/CUDAContext.h"
#include "c10/util/Optional.h"
#include "ir_base_nodes.h"
#include "ir_interface_nodes.h"
#include "ir_internal_nodes.h"
#include "ir_utils.h"
#include "mma_type.h"
#include "type.h"
#include "utils.h"

namespace nvfuser {

using MatmulLayout = MmaOptions::MmaInputLayout;
using LayoutData =
    std::pair<c10::optional<MatmulLayout>, c10::optional<std::string>>;
using TensorShape = std::vector<int64_t>;
using ProblemShape = TensorShape;

//! A constant with position of M value (a number of columns in A tensor for TT
//! layout) in problem in ProblemShape type.
constexpr size_t M_POS = 0;
//! A constant with position of N value (a number of rows in B tensor for TT
//! layout) in problem in ProblemShape type.
constexpr size_t N_POS = 1;
//! A constant with position of K value (a number of rows in A tensor for TT
//! layout) in problem in ProblemShape type.
constexpr size_t K_POS = 2;
//! A constant with expected number of dimensions in ProblemShape type.
constexpr size_t PROBLEM_DIMS = 3;

namespace {

//! A collection of labels for supported expressions that are supported by
//! matmul scheduler.
enum class MatmulType {
  Unsupported = 0,
  // basic formula: C = A x B
  // A, B and C have the same number of dimensions
  Basic,
};

//! A wrapper for printing debug details.
void printMsg(const std::string& msg) {
  std::cout << msg << std::endl;
}

//! A helper for deciding what kernel indexing mode use (int32_t or int64_t).
PrimDataType getIndexType(const ProblemShape& problem_shape) {
  // based on collectIndexMode function
  constexpr int64_t most_positive_int32_index =
      std::numeric_limits<int>::max() / 2;

  const bool use_i64_index = //
      problem_shape[M_POS] > most_positive_int32_index || //
      problem_shape[N_POS] > most_positive_int32_index || //
      problem_shape[K_POS] > most_positive_int32_index;

  return use_i64_index ? PrimDataType::Int : PrimDataType::Int32;
}

//! A helper for deciding the type of MMA op for given fusion and problem shape.
inline c10::optional<MmaOptions::MacroType> getMmaOp(
    const int dev_version,
    const ProblemShape& problem) {
  using MacroType = MmaOptions::MacroType;

  TORCH_INTERNAL_ASSERT(
      problem.size() == PROBLEM_DIMS,
      "Invalid size of problem shape (number of dimensions)");

  // NOTE: A temp condition
  const bool use_small_n =
      ((problem[N_POS] % 8) == 0) && ((problem[N_POS] % 16) != 0);

  switch (dev_version) {
    case 70:
      return MacroType::Volta_16_16_4;
    case 75:
      return (use_small_n) ? MacroType::Turing_16_8_16
                           : MacroType::Turing_16_16_16;
    case 80:
      return (use_small_n) ? MacroType::Ampere_16_8_16
                           : MacroType::Ampere_16_16_16;
    default:
      break;
  }
  return c10::nullopt;
}

//! A helper for deciding warp tile shape, based on MMA op.
inline GemmTile getWarpTileShape(
    const MmaOptions::MacroType& op,
    const GemmTile& instruction_tile) {
  using DimType = decltype(GemmTile::m);

  if (isAmpere(op)) {
    // Initial target:
    // - 2 MMA ops per thread in a warp (32 threads), warp tile should be 64x
    // instruction tile
    constexpr DimType target_dim = 64;

    const DimType m_ratio = target_dim / instruction_tile.m;
    const DimType n_ratio = target_dim / instruction_tile.n;
    const DimType k_ratio = target_dim / (m_ratio * n_ratio);

    return {
        instruction_tile.m * m_ratio,
        instruction_tile.n * n_ratio,
        instruction_tile.k * k_ratio};
  }

  // TODO: support for Volta and Turing
  TORCH_INTERNAL_ASSERT(false, "unsupported architecture");
  return instruction_tile;
}

//! A helper for deciding GTA tile shape, based on warp tile and problem shapes.
inline GemmTile getCtaTileShape(
    const GemmTile& warp_tile,
    const ProblemShape& problem) {
  // Initial target:
  // - 4 warp tiles per CTA
  // - CTA k-dim should be same as warp tile k-dim

  using DimType = decltype(GemmTile::m);

  DimType m_ratio = 2;
  DimType n_ratio = 2;

  const auto mn_ratio = problem[M_POS] / (double)problem[N_POS];
  if (mn_ratio < 0.5) {
    m_ratio = 1;
    n_ratio = 4;
  } else if (mn_ratio > 2) {
    m_ratio = 4;
    n_ratio = 1;
  }

  return {warp_tile.m * m_ratio, warp_tile.n * n_ratio, warp_tile.k};
}

//! A helper for checking if layout of MMA op's inputs. It will return optional
//! message if check fails.
LayoutData getLayout(const MmaOp* mma_expr) {
  std::stringstream ss;
  const auto& mmaExprInputs = mma_expr->inputs();

  const auto* in_A = mmaExprInputs[0]->as<TensorView>();
  const auto* in_B = mmaExprInputs[1]->as<TensorView>();

  // The number of IterDomains of MMA inputs must be the same
  if (in_A->nDims() != in_B->nDims()) {
    ss << "Mma op inputs don't have the same number of IterDomains, 1st input("
       << std::to_string(in_A->nDims()) << "), 2nd input("
       << std::to_string(in_B->nDims()) + ")";
    return {c10::nullopt, ss.str()};
  }

  // The currently supported number of IterDomains per MMA op input is 3
  const size_t supportedDims = 3;
  if (in_A->nDims() != supportedDims) {
    ss << "Mma op inputs have unsupported number of IterDomains, got: "
       << std::to_string(in_A->nDims()) << ", expected "
       << std::to_string(supportedDims);
    return {c10::nullopt, ss.str()};
  }

  using AxisPos = decltype(std::declval<TensorView>().nDims());
  const AxisPos unInitPos = -1;
  AxisPos bcastInApos = unInitPos;
  AxisPos bcastInBpos = unInitPos;

  // The first and the second input of MMA have the same number of
  // IterDomains
  for (AxisPos pos = 0; pos < in_A->nDims(); ++pos) {
    if (in_A->axis(pos)->isBroadcast()) {
      if (bcastInApos != unInitPos) {
        ss << "Mma op first input has more than one broadcast IterDomain: "
           << std::to_string(bcastInApos) << " and " << std::to_string(pos);
        return {c10::nullopt, ss.str()};
      }
      bcastInApos = pos;
    }
    if (in_B->axis(pos)->isBroadcast()) {
      if (bcastInBpos != unInitPos) {
        ss << "Mma op second input has more than one broadcast IterDomain: "
           << std::to_string(bcastInBpos) << " and " << std::to_string(pos);
        return {c10::nullopt, ss.str()};
      }
      bcastInBpos = pos;
    }
  }

  // MMA inputs need to have broadcast IterDomains
  if (bcastInApos == unInitPos || bcastInBpos == unInitPos) {
    ss << "The " << (bcastInApos == unInitPos ? "first" : "second")
       << " mma op doesn't have any broadcast IterDomain";
    return {c10::nullopt, ss.str()};
  }

  // MMA inputs must have supported data layout, defined in MatmulLayout
  // MatmulLayout::TT
  if (bcastInApos == static_cast<size_t>(2) &&
      bcastInBpos == static_cast<size_t>(0)) {
    return {MatmulLayout::TT, c10::nullopt};
  }
  // MatmulLayout::TN
  if (bcastInApos == static_cast<size_t>(1) &&
      bcastInBpos == static_cast<size_t>(0)) {
    return {MatmulLayout::TN, c10::nullopt};
  }
  // MatmulLayout::NT
  if (bcastInApos == static_cast<size_t>(2) &&
      bcastInBpos == static_cast<size_t>(1)) {
    return {MatmulLayout::NT, c10::nullopt};
  }

  ss << "Unsupported layout, broadcasts: inputA(" << bcastInApos << "), inputB("
     << bcastInBpos << ")";
  return {c10::nullopt, ss.str()};
}

//! A helper for making a quick check if fusion inputs and outputs match any of
//! supported matmul patterns.
MatmulType getMatmulType(Fusion* fusion) {
  const auto& inputs = fusion->inputs();
  const auto& outputs = fusion->outputs();

  using InputsSizeType = std::decay<decltype(inputs)>::type::size_type;
  using OutputsSizeType = std::decay<decltype(outputs)>::type::size_type;

  // MatmulType::Basic
  if (inputs.size() == 2 && outputs.size() == 1) {
    const InputsSizeType inputs_expected_dims = 2;
    // output tensor has additional reduction dim
    const OutputsSizeType output_expected_dims = 3;
    const auto* in_A = inputs[0]->as<TensorView>();
    const auto* in_B = inputs[1]->as<TensorView>();
    const auto* output = outputs[0]->as<TensorView>();

    if (inputs_expected_dims == in_A->nDims() &&
        in_A->nDims() == in_B->nDims() && //
        output_expected_dims == output->nDims()) {
      return MatmulType::Basic;
    }
  }

  return MatmulType::Unsupported;
}

//! A helper for getting problem shape from fusion and runtime info. Operation
//! can fail and nullopt object is returned.
c10::optional<ProblemShape> getProblemShape(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    const MatmulType matmul_type,
    const MatmulLayout matmul_layout) {
  const auto getShape = [&runtime_info](const TensorView* tv) {
    const auto& root_dom = tv->getMaybeRFactorDomain();
    TensorShape tv_shape;

    for (size_t i = root_dom.size(); i > 0; i--) {
      auto id = root_dom[i - 1];
      if (id->isBroadcast()) {
        continue;
      }
      if (id->isReduction()) {
        continue;
      }
      auto dim_size = runtime_info.expressionEvaluator().evaluate(id->extent());
      if (dim_size) {
        tv_shape.push_back(dim_size->as<int64_t>());
      }
    }
    return tv_shape;
  };

  switch (matmul_type) {
    case MatmulType::Basic: {
      const auto& in_A = getShape(fusion->inputs()[0]->as<TensorView>());
      const auto& in_B = getShape(fusion->inputs()[1]->as<TensorView>());
      const auto& output = getShape(fusion->outputs()[0]->as<TensorView>());

      constexpr size_t expected_dims = 2;
      if (in_A.size() != expected_dims || //
          in_B.size() != expected_dims || //
          output.size() != expected_dims) {
        return c10::nullopt;
      }

      switch (matmul_layout) {
        case MatmulLayout::TT: {
          // in_A := [K, M]
          // in_B := [N, K]
          // output := [N, M]
          const bool check_k = in_A[0] == in_B[1];
          const bool check_m = in_A[1] == output[1];
          const bool check_n = in_B[0] == output[0];
          if (!(check_k && check_m && check_n)) {
            return c10::nullopt;
          }
          // [M, N, K]
          return TensorShape{in_A[1], in_B[0], in_A[0]};
        }
        case MatmulLayout::NT: {
          // in_A := [M, K]
          // in_B := [N, K]
          // output := [N, M]
          const bool check_k = in_A[1] == in_B[1];
          const bool check_m = in_A[0] == output[1];
          const bool check_n = in_B[0] == output[0];
          if (!(check_k && check_m && check_n)) {
            return c10::nullopt;
          }
          // [M, N, K]
          return TensorShape{in_A[0], in_B[0], in_A[1]};
        }
        case MatmulLayout::TN: {
          // in_A := [K, M]
          // in_B := [K, N]
          // output := [N, M]
          const bool check_k = in_A[0] == in_B[0];
          const bool check_m = in_A[1] == output[1];
          const bool check_n = in_B[1] == output[0];
          if (!(check_k && check_m && check_n)) {
            return c10::nullopt;
          }
          // [M, N, K]
          return TensorShape{in_A[1], in_B[1], in_A[0]};
        }
        default:
          return c10::nullopt;
      }
      break;
    }
    default:
      break;
  }
  return c10::nullopt;
}

//! A helper for checking in depth the basic pattern (C=AxB) in provided fusion
//! and MMA op defined in it.
std::string checkMatmulPatternBasic(const MmaOp* mma_expr, Fusion* fusion) {
  const auto fusion_inputs = fusion->inputs();
  const auto fusion_outputs = fusion->outputs();
  const auto mma_inputs = mma_expr->inputs();
  const auto mma_outputs = mma_expr->outputs();

  // Fusion input A: fusion_inputs[0]
  // Fusion input B: fusion_inputs[1]
  // Fusion output C: fusion_outputs[0]
  // MMA input A: mma_inputs[0]
  // MMA input B: mma_inputs[1]
  // MMA output C: mma_outputs[0]

  // Check dependencies:
  // #1 - fusion [A,B] -> mma A
  // #2 - fusion [B,A] -> mma B
  // #3 - mma C -> fusion C
  // #4 - fusion A, B, C -> global memory
  // #5 - mma A, B -> local memory
  // #6 - mma C -> global memory

  // #1
  auto path_to_mma_A =
      DependencyCheck::getAllDependencyChains(fusion_inputs[0], mma_inputs[0]);
  if (path_to_mma_A.empty()) {
    // in case fusion B -> mma A
    path_to_mma_A = DependencyCheck::getAllDependencyChains(
        fusion_inputs[1], mma_inputs[0]);
  }
  if (path_to_mma_A.empty()) {
    return "No dependency between MMA A input and any of fusion inputs";
  }

  // #2
  auto path_to_mma_B =
      DependencyCheck::getAllDependencyChains(fusion_inputs[1], mma_inputs[1]);
  if (path_to_mma_B.empty()) {
    // in case fusion A -> mma B
    path_to_mma_B = DependencyCheck::getAllDependencyChains(
        fusion_inputs[0], mma_inputs[1]);
  }
  if (path_to_mma_B.empty()) {
    return "No dependency between MMA B input and any of fusion inputs";
  }

  // #3
  auto path_from_mma_C = DependencyCheck::getAllDependencyChains(
      fusion_outputs[0], mma_outputs[0]);
  if (path_from_mma_C.empty()) {
    return "No dependency between MMA B input and any of fusion inputs";
  }

  auto is_tensor_expected_location =
      [](const Val* val, const MemoryType mem_type) -> std::string {
    if (nullptr == val) {
      return "Val is a nullptr";
    }
    if (const auto* tv = dynamic_cast<const TensorView*>(val)) {
      return (tv->getMemoryType() == mem_type)
          ? ""
          : "TensorView is located in different location than expeced";
    } else {
      return "Val is not a tensor view";
    }
  };

  // #4
  auto tv_status =
      is_tensor_expected_location(fusion_inputs[0], MemoryType::Global);
  if (!tv_status.empty()) {
    return tv_status;
  }
  tv_status = is_tensor_expected_location(fusion_inputs[1], MemoryType::Global);
  if (!tv_status.empty()) {
    return tv_status;
  }
  // #5
  tv_status =
      is_tensor_expected_location(fusion_outputs[0], MemoryType::Global);
  if (!tv_status.empty()) {
    return tv_status;
  }
  tv_status = is_tensor_expected_location(mma_inputs[0], MemoryType::Local);
  if (!tv_status.empty()) {
    return tv_status;
  }
  // #6
  tv_status = is_tensor_expected_location(mma_inputs[1], MemoryType::Local);
  if (!tv_status.empty()) {
    return tv_status;
  }

  return "";
}

//! A dispatcher for checking supported matmul patterns.
std::string checkMatmulPattern(
    const MmaOp* mma_expr,
    const MatmulType matmul_type,
    Fusion* fusion) {
  switch (matmul_type) {
    case MatmulType::Basic:
      return checkMatmulPatternBasic(mma_expr, fusion);
    case MatmulType::Unsupported:
      return "Unsupported matmul operations pattern in provided fusion object";
  }
  return "Unknwon matmul opreations pattern in provided fusion object";
}

} // anonymous namespace

std::string getMatmulRunTimeRejectReason(
    Fusion* fusion,
    HeuristicSummary* data_cache,
    SchedulerRuntimeInfo& runtime_info) {
  // TODO: add proper set of checks
  return "";
}

std::string getMatmulCompileTimeRejectReason(Fusion* fusion) {
  // The plan:
  // 1. check if there is exactly one MmaOp defined in the fusion
  // 2. check if fusion represents expressions that can be recognized as matmul
  // 3. check if MmaOp inputs match any of supported layout
  // 4. evaluate relationship between fusion inputs and outputs through MmaOp

  // #1
  const auto fusion_exprs = fusion->exprs();
  auto mmaExprs = ir_utils::filterByType<MmaOp>(fusion_exprs).vector();
  if (mmaExprs.size() != 1) {
    std::stringstream ss;
    ss << "Matmul scheduler supports only a single MMA op in the fusion, got: "
       << mmaExprs.size();
    return ss.str();
  }

  // #2
  const auto matmul_type = getMatmulType(fusion);
  if (matmul_type == MatmulType::Unsupported) {
    return "Provided set of expressions cannot be matched with known matmul type";
  }

  // #3
  {
    for (const auto* mma_expr : mmaExprs) {
      const auto layout_data = getLayout(mma_expr);
      if (layout_data.second) {
        return layout_data.second.value();
      }
    }
  }

  // #4
  {
    for (auto expr : mmaExprs) {
      const auto msg = checkMatmulPattern(expr, matmul_type, fusion);
      if (!msg.empty()) {
        return msg;
      }
    }
  }

  return "";
}

std::shared_ptr<MatmulParams> getMatmulHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache) {
  FusionGuard fg(fusion);
  (void)data_cache;
  (void)runtime_info;
  auto params = std::make_shared<MatmulParams>();

  const auto fusion_exprs = fusion->exprs();
  auto mma_exprs = ir_utils::filterByType<MmaOp>(fusion_exprs).vector();
  if (mma_exprs.size() != 1) {
    // Support only for fusion with a single mma op
    return nullptr;
  }

  const auto matmul_type = getMatmulType(fusion);
  if (matmul_type == MatmulType::Unsupported) {
    // Unsupported matmul type, based on input/output tensors
    if (isDebugDumpEnabled(DebugDumpOption::MatmulChecks)) {
      printMsg("Unsupported matmul type");
    }
    return nullptr;
  }

  const auto layout = getLayout(mma_exprs.front());
  if (layout.second) {
    // no heuristics if layout request returned an error message
    if (isDebugDumpEnabled(DebugDumpOption::MatmulChecks)) {
      printMsg(layout.second.value());
    }
    return nullptr;
  }

  const auto problem_shape =
      getProblemShape(fusion, runtime_info, matmul_type, layout.first.value());
  if (!problem_shape) {
    // Failed to acquire problem shape
    return nullptr;
  }

  const auto device_prop = at::cuda::getCurrentDeviceProperties();
  const auto mma_op = getMmaOp(
      device_prop->major * 10 + device_prop->minor, problem_shape.value());
  if (!mma_op) {
    // no heuristics if mma op request is empty
    return nullptr;
  }

  // initialize heuristic parameters
  const GemmTile instruction_tile = getMmaOpShape(mma_op.value());
  const GemmTile warp_tile = getWarpTileShape(mma_op.value(), instruction_tile);
  const GemmTile cta_tile = getCtaTileShape(warp_tile, problem_shape.value());
  const int stages = 3;

  params->mma_op = mma_op.value();
  params->layout = layout.first.value();
  params->tile_sizes = {cta_tile, warp_tile, instruction_tile};
  params->async_gmem_load_operands = true;
  params->double_buffer_options.double_buffer_smem_write = true;
  params->double_buffer_options.double_buffer_smem_read = true;
  params->double_buffer_options.smem_double_buffer_stage = stages;

  // set kernel index mode
  params->cparams.index_type = getIndexType(problem_shape.value());

  if (isDebugDumpEnabled(DebugDumpOption::MatmulChecks)) {
    printMsg(params->toString());
  }

  return params;
}

} // namespace nvfuser
