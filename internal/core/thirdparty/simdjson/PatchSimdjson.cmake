# Copyright (C) 2026 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# simdjson 5.0.2 defines these header functions with
# SIMDJSON_CONSTEXPR_STRING, which is empty on older standard libraries.
# Explicit inline preserves constexpr support while avoiding ODR violations.
# Apply after population so existing build directories are repaired too.
set(_simdjson_header
    "${simdjson_SOURCE_DIR}/include/simdjson/generic/ondemand/key_selector.h")
file(READ "${_simdjson_header}" _simdjson_original)
set(_simdjson_patched "${_simdjson_original}")
foreach(_simdjson_function IN ITEMS append_uint append_byte)
    set(_simdjson_before "SIMDJSON_CONSTEXPR_STRING void ${_simdjson_function}(")
    set(_simdjson_after "inline ${_simdjson_before}")
    string(FIND "${_simdjson_patched}" "${_simdjson_after}" _simdjson_already_patched)
    if(_simdjson_already_patched EQUAL -1)
        string(FIND "${_simdjson_patched}" "${_simdjson_before}" _simdjson_found)
        if(_simdjson_found EQUAL -1)
            message(FATAL_ERROR
                "simdjson inline compatibility patch no longer matches "
                "${_simdjson_function}; review the dependency patch.")
        endif()
        string(REPLACE "${_simdjson_before}" "${_simdjson_after}"
            _simdjson_patched "${_simdjson_patched}")
    endif()
endforeach()
if(NOT _simdjson_original STREQUAL _simdjson_patched)
    file(WRITE "${_simdjson_header}" "${_simdjson_patched}")
    message(STATUS "Applied simdjson 5.0.2 inline compatibility patch")
endif()
