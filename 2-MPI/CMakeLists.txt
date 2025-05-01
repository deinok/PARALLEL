cmake_minimum_required(VERSION 3.28.3)
project(OMP_MPI C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 1) Try to find OpenMP via CMake’s built-in module
find_package(OpenMP QUIET)

if(OpenMP_C_FOUND)
  message(STATUS "Found OpenMP via CMake.")
else()
  if(APPLE)
    message(STATUS "OpenMP module not found; trying Homebrew libomp…")
    # look for libomp headers and library in common Homebrew locations
    find_path(HOMEBREW_OMP_INCLUDE omp.h
      PATHS /opt/homebrew/opt/libomp/include /usr/local/include)
    find_library(HOMEBREW_OMP_LIB omp
      PATHS /opt/homebrew/opt/libomp/lib /usr/local/lib)
    if(NOT HOMEBREW_OMP_INCLUDE OR NOT HOMEBREW_OMP_LIB)
      message(FATAL_ERROR "libomp not found. Install it with: brew install libomp")
    endif()
    # set flags for Apple Clang + libomp
    set(OpenMP_C_FLAGS        "-Xpreprocessor -fopenmp")
    set(OpenMP_C_INCLUDE_DIRS "${HOMEBREW_OMP_INCLUDE}")
    set(OpenMP_C_LIBRARIES    "${HOMEBREW_OMP_LIB}")
    # create an imported target so we can link with OpenMP::OpenMP_C
    add_library(OpenMP::OpenMP_C UNKNOWN IMPORTED)
    set_target_properties(OpenMP::OpenMP_C PROPERTIES
      IMPORTED_LOCATION "${HOMEBREW_OMP_LIB}"
      INTERFACE_INCLUDE_DIRECTORIES "${HOMEBREW_OMP_INCLUDE}"
      INTERFACE_COMPILE_OPTIONS "${OpenMP_C_FLAGS}"
    )
  else()
    message(FATAL_ERROR "OpenMP not available. On Linux install libgomp (e.g. via GCC).")
  endif()
endif()

# 2) Require MPI
find_package(MPI REQUIRED)

# 3) Define executable
add_executable(OMP_MPI heat_parallel.c)

# 4) Link OpenMP (via imported target or manual flags)
if(TARGET OpenMP::OpenMP_C)
  target_link_libraries(OMP_MPI PRIVATE OpenMP::OpenMP_C)
else()
  target_include_directories(OMP_MPI PRIVATE ${OpenMP_C_INCLUDE_DIRS})
  target_compile_options(OMP_MPI PRIVATE ${OpenMP_C_FLAGS})
  target_link_libraries(OMP_MPI PRIVATE ${OpenMP_C_LIBRARIES})
endif()

# 5) Link MPI
target_link_libraries(OMP_MPI PRIVATE MPI::MPI_C)

# 6) Show MPI settings
message(STATUS "MPI include dir: ${MPI_C_INCLUDE_PATH}")
message(STATUS "MPI libraries: ${MPI_C_LIBRARIES}")
