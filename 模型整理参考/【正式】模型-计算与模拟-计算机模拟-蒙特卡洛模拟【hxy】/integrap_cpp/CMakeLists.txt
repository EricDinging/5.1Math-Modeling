cmake_minimum_required(VERSION 3.21)
project(integrap_cpp)

set(CMAKE_CXX_STANDARD 14)

add_executable(integrap_cpp integral.cpp monte_carlo.h)
