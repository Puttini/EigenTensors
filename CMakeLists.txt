cmake_minimum_required(VERSION 3.5)
project(EigenTensors)

set( EIGEN_PATH "E:/eigen" )
include_directories( ${EIGEN_PATH} )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp)
add_executable(EigenTensors ${SOURCE_FILES})