cmake_minimum_required(VERSION 3.16)
project(p4_synth LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Where we cloned the EPFL libs
set(EPFL_EXTERNAL_DIR "${CMAKE_SOURCE_DIR}/external")

# Basic sanity: warn if libs are missing, but don't be "smart"
foreach(lib mockturtle kitty percy)
  if(NOT EXISTS "${EPFL_EXTERNAL_DIR}/${lib}")
    message(WARNING "Expected ${EPFL_EXTERNAL_DIR}/${lib} (did you run install_epfl_libs.sh?)")
  endif()
endforeach()

# Executable
add_executable(p4_synth
  src/main.cpp
)

# Define LIN64 for ABC (needed for percy's ABC headers)
target_compile_definitions(p4_synth PRIVATE LIN64)

# Header-only wiring:
#  - mockturtle/include for main headers
#  - mockturtle/lib/parallel_hashmap for <parallel_hashmap/phmap.h>
#  - mockturtle/lib/fmt for fmt library
#  - mockturtle/lib/bill for bill/sat headers
#  - mockturtle/lib/abcesop for eabc headers
#  - mockturtle/lib/lorina for lorina headers
#  - mockturtle/lib/json for nlohmann/json headers
#  - kitty/include & percy/include per their docs
#  - percy/nauty for nauty.h (needs configure to be run first)
#  - percy/abc for ABC headers like <abc/satSolver.h>
target_include_directories(p4_synth PRIVATE
  "${EPFL_EXTERNAL_DIR}/mockturtle/include"
  "${EPFL_EXTERNAL_DIR}/mockturtle/lib/parallel_hashmap"
  "${EPFL_EXTERNAL_DIR}/mockturtle/lib/fmt"
  "${EPFL_EXTERNAL_DIR}/mockturtle/lib/bill"
  "${EPFL_EXTERNAL_DIR}/mockturtle/lib/abcesop"
  "${EPFL_EXTERNAL_DIR}/mockturtle/lib/lorina"
  "${EPFL_EXTERNAL_DIR}/mockturtle/lib/json"
  "${EPFL_EXTERNAL_DIR}/kitty/include"
  "${EPFL_EXTERNAL_DIR}/percy/include"
  "${EPFL_EXTERNAL_DIR}/percy/nauty"
  "${EPFL_EXTERNAL_DIR}/percy/abc"
)

