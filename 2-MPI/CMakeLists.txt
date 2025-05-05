cmake_minimum_required(VERSION 2.8.12.2...3.5)
project(OMP_MPI C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 1) Find OpenMP
find_package(OpenMP QUIET)
if(OpenMP_C_FOUND)
  message(STATUS "Found OpenMP via CMake.")
else()
  if(APPLE)
    message(STATUS "OpenMP module not found; trying Homebrew libomp…")
    find_path(HOMEBREW_OMP_INCLUDE omp.h
      PATHS /opt/homebrew/opt/libomp/include /usr/local/include)
    find_library(HOMEBREW_OMP_LIB omp
      PATHS /opt/homebrew/opt/libomp/lib /usr/local/lib)
    if(NOT HOMEBREW_OMP_INCLUDE OR NOT HOMEBREW_OMP_LIB)
      message(FATAL_ERROR "libomp not found. Install it with: brew install libomp")
    endif()
    set(OpenMP_C_FLAGS        "-Xpreprocessor -fopenmp")
    set(OpenMP_C_INCLUDE_DIRS "${HOMEBREW_OMP_INCLUDE}")
    set(OpenMP_C_LIBRARIES    "${HOMEBREW_OMP_LIB}")
    add_library(OpenMP::OpenMP_C UNKNOWN IMPORTED)
    set_target_properties(OpenMP::OpenMP_C PROPERTIES
      IMPORTED_LOCATION "${HOMEBREW_OMP_LIB}"
      INTERFACE_INCLUDE_DIRECTORIES "${HOMEBREW_OMP_INCLUDE}"
      INTERFACE_COMPILE_OPTIONS "${OpenMP_C_FLAGS}")
  else()
    message(WARNING "OpenMP module not detected; falling back to -fopenmp and -lgomp")
    set(OpenMP_C_FLAGS "-fopenmp")
    set(OpenMP_C_LIBRARIES "gomp")
  endif()
endif()

# 2) Find MPI
find_package(MPI REQUIRED)
if(NOT MPI_C_FOUND)
  message(FATAL_ERROR "MPI not found. Install MPI development libraries.")
endif()

# 3) Define executable
add_executable(OMP_MPI heat_parallel.c)

# 4) Link OpenMP
if(TARGET OpenMP::OpenMP_C)
  target_link_libraries(OMP_MPI PRIVATE OpenMP::OpenMP_C)
else()
  target_compile_options(OMP_MPI PRIVATE ${OpenMP_C_FLAGS})
  target_link_libraries(OMP_MPI PRIVATE ${OpenMP_C_LIBRARIES})
endif()

# 5) Include MPI headers and link
target_include_directories(OMP_MPI PRIVATE ${MPI_C_INCLUDE_PATH})
if(TARGET MPI::MPI_C)
  target_link_libraries(OMP_MPI PRIVATE MPI::MPI_C)
else()
  target_link_libraries(OMP_MPI PRIVATE ${MPI_C_LIBRARIES})
endif()

# 6) Show configuration
message(STATUS "MPI include dir: ${MPI_C_INCLUDE_PATH}")
message(STATUS "MPI libraries: ${MPI_C_LIBRARIES}")
