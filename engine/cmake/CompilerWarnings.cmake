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

# Warning policy for first-party targets. -Wold-style-cast enforces the
# Google style rule against C-style casts at compile time.

function(vectra_set_warnings target)
  target_compile_options(${target} PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Wshadow
    -Wnon-virtual-dtor
    -Wold-style-cast
    -Woverloaded-virtual
    -Wimplicit-fallthrough
    -Wformat=2)
  if(VECTRA_WARNINGS_AS_ERRORS)
    target_compile_options(${target} PRIVATE -Werror)
  endif()
endfunction()
