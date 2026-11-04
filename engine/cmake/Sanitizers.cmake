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

# -DVECTRA_SANITIZE=address,undefined (or thread) instruments first-party
# targets. Required before submitting changes to memory ownership.

function(vectra_apply_sanitizers target)
  if(NOT VECTRA_SANITIZE)
    return()
  endif()
  set(flags "-fsanitize=${VECTRA_SANITIZE}" -fno-omit-frame-pointer
            -fno-sanitize-recover=all)
  target_compile_options(${target} PUBLIC ${flags})
  target_link_options(${target} PUBLIC "-fsanitize=${VECTRA_SANITIZE}")
endfunction()
