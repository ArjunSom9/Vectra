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

# Link-time optimization for optimized builds, when the toolchain supports it.

include(CheckIPOSupported)

function(vectra_apply_lto target)
  if(NOT VECTRA_ENABLE_LTO)
    return()
  endif()
  check_ipo_supported(RESULT supported OUTPUT reason LANGUAGES CXX)
  if(supported)
    set_property(TARGET ${target} PROPERTY
      INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    set_property(TARGET ${target} PROPERTY
      INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO TRUE)
  else()
    message(STATUS "LTO unavailable for ${target}: ${reason}")
  endif()
endfunction()
