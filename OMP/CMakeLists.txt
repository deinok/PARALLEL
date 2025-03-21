cmake_minimum_required(VERSION 3.30)
project(untitled)

set(CMAKE_CXX_STANDARD 26)

#find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(untitled main.c)
#target_link_libraries(untitled PRIVATE MPI::MPI_CXX)
target_link_libraries(untitled PRIVATE OpenMP::OpenMP_CXX)
