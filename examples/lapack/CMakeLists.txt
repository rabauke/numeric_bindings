cmake_minimum_required(VERSION 2.8.9)

set(CMAKE_CXX_EXTENSIONS off)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(../../ /usr/include/eigen3)

project(Lapack_tests)

add_executable(gesv gesv.cc)
target_link_libraries(gesv blas lapack dl)

add_executable(gbsv gbsv.cc)
target_link_libraries(gbsv blas lapack dl)
