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

# vectra_grpc_proto_library(<target> PROTO_ROOT <dir> PROTOS <file>...)
#
# Generates message (*.pb.{h,cc}) and service (*.grpc.pb.{h,cc}) sources for
# every listed .proto (paths relative to PROTO_ROOT) into the build tree and
# compiles them into a static library. Stubs are never committed; they are
# regenerated from contracts/proto, the single source of truth.

function(vectra_grpc_proto_library target)
  cmake_parse_arguments(ARG "" "PROTO_ROOT" "PROTOS" ${ARGN})
  set(out_dir "${CMAKE_CURRENT_BINARY_DIR}/gen")
  file(MAKE_DIRECTORY "${out_dir}")
  set(protoc $<TARGET_FILE:protobuf::protoc>)
  set(grpc_plugin $<TARGET_FILE:gRPC::grpc_cpp_plugin>)
  set(sources)
  foreach(proto IN LISTS ARG_PROTOS)
    string(REGEX REPLACE "\\.proto$" "" stem "${proto}")
    set(generated
      "${out_dir}/${stem}.pb.cc" "${out_dir}/${stem}.pb.h"
      "${out_dir}/${stem}.grpc.pb.cc" "${out_dir}/${stem}.grpc.pb.h")
    add_custom_command(
      OUTPUT ${generated}
      COMMAND ${protoc}
        --proto_path=${ARG_PROTO_ROOT}
        --cpp_out=${out_dir}
        --grpc_out=${out_dir}
        --plugin=protoc-gen-grpc=${grpc_plugin}
        ${ARG_PROTO_ROOT}/${proto}
      DEPENDS "${ARG_PROTO_ROOT}/${proto}" protobuf::protoc
              gRPC::grpc_cpp_plugin
      COMMENT "Generating C++ for ${proto}"
      VERBATIM)
    list(APPEND sources "${out_dir}/${stem}.pb.cc" "${out_dir}/${stem}.grpc.pb.cc")
  endforeach()
  add_library(${target} STATIC ${sources})
  target_include_directories(${target} PUBLIC "${out_dir}")
  target_link_libraries(${target} PUBLIC protobuf::libprotobuf gRPC::grpc++)
endfunction()
