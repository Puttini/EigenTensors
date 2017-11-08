cmake_minimum_required(VERSION 3.5)
project(EigenTensors)

set( EIGEN_PATH "E:/eigen" CACHE PATH "Eigen Path" )
include_directories( ${EIGEN_PATH} )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp ConstScheme1.h ConstScheme2.h InheritScheme1.h InheritScheme2.h)
add_executable(EigenTensors ${SOURCE_FILES})
