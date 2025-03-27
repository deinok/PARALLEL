cmake_minimum_required(VERSION 3.28.3)
project(OMP)

set(CMAKE_C_STANDARD 99)

#find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(OMP main.c)
#target_link_libraries(untitled PRIVATE MPI::MPI_CXX)
target_link_libraries(OMP PRIVATE OpenMP::OpenMP_C)
