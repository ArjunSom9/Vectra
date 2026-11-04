# Copyright 2025 Texas A&M University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Selects the instruction set the distance kernels are vectorized for.
#
#   VECTRA_ARCH=native     the build host (development default)
#   VECTRA_ARCH=znver5     AMD EPYC 9005 production nodes (AVX-512)
#   VECTRA_ARCH=x86-64-v4  any AVX-512 x86-64 node
#   VECTRA_ARCH=x86-64-v3  any AVX2 x86-64 node
#   VECTRA_ARCH=none       compiler default (portable baseline)
#
# The kernels in src/index/simd_distance.cc are plain C++ with fixed-width
# accumulator lanes, so the choice here is the only SIMD switch.

include(CheckCXXCompilerFlag)

function(vectra_apply_simd target)
  if(VECTRA_ARCH STREQUAL "none" OR VECTRA_ARCH STREQUAL "")
    return()
  endif()
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64|ARM64)$")
    if(VECTRA_ARCH STREQUAL "native")
      set(flag "-mcpu=native")
    else()
      set(flag "-mcpu=${VECTRA_ARCH}")
    endif()
  else()
    set(flag "-march=${VECTRA_ARCH}")
  endif()
  string(MAKE_C_IDENTIFIER "VECTRA_HAS_${flag}" cache_var)
  check_cxx_compiler_flag("${flag}" ${cache_var})
  if(${cache_var})
    target_compile_options(${target} PRIVATE ${flag})
  else()
    message(WARNING "Compiler rejects ${flag}; using its default target.")
  endif()
endfunction()
