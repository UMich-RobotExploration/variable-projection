cmake_minimum_required(VERSION 3.5)

project(VARPRO LANGUAGES C CXX VERSION 0.1.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the default build type to debug
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
    # set(CMAKE_BUILD_TYPE Debug)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
    message(STATUS "Setting build type to '${CMAKE_BUILD_TYPE}' as none was specified.")
else()
    message(STATUS "Build type set to '${CMAKE_BUILD_TYPE}'.")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    # Add the -O3 optimization flag
    add_compile_options(-O3)
endif()

# Directory for built libraries
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib CACHE PATH "Output directory for libraries")
# Directory for built executables
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin CACHE PATH "Output directory for executables")

# BUILD CONFIGURATIONS
# Enable faster instruction sets (SIMD/AVX)
set(ENABLE_VECTORIZATION ON CACHE BOOL "Enable vectorized instruction sets (SIMD/AVX)? [enabled by default]")
# Enable OMP parallelization
set(ENABLE_OPENMP OFF CACHE BOOL "Enable OMP parallelization? [enabled by default]")
# Enable code profiling using gperftools
set(ENABLE_PROFILING OFF CACHE BOOL "Enable code profiling using gperftools")
# Run the experiments from the paper
set(PERFORM_EXPERIMENTS ON CACHE BOOL "Run the experiments from the paper.")
# Build C++ examples
set(BUILD_EXAMPLES ON CACHE BOOL "Build C++ examples.")
# Build Python bindings
set(BUILD_PYTHON_BINDINGS OFF CACHE BOOL "Build Python bindings.")

# Add the .cmake files that ship with Eigen3 to the CMake module path (useful for finding other stuff)
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" CACHE STRING "The CMake module path used for this project")

# Find (system) Eigen library
find_package(Eigen3 3.3.7 REQUIRED)
if(EIGEN3_FOUND)
message(STATUS "Found Eigen3 library: ${EIGEN3_INCLUDE_DIR}\n")
endif()

if(${ENABLE_VECTORIZATION})
message(STATUS "Enabling SIMD/AVX instruction sets")
add_definitions(-march=native)
endif()

if(${ENABLE_OPENMP})
message(STATUS "Enabling OMP parallelization")
find_package(OpenMP)
if(OPENMP_FOUND)
message(STATUS "Found OpenMP library: ${OPENMP_INCLUDES}")
endif()
endif()

if(${ENABLE_PROFILING})
message(STATUS "Enabling code profiling using Google Performance Tools")
set(CMAKE_EXE_LINKER_FLAGS ${CMAKE_EXE_LINKER_FLAGS} -lprofiler)
endif()

if(${BUILD_PYTHON_BINDINGS})
message(STATUS "Building Python bindings")
endif()

# DEPENDENCIES

# FIND ADDITIONAL LIBRARIES
# These next operations make use of the .cmake files shipped with Eigen3
find_package(SPQR REQUIRED)
find_package(BLAS REQUIRED)

# print out the current_source_dir
message(STATUS "Current source directory: ${CMAKE_CURRENT_SOURCE_DIR}")

# set the libs directory
set(LIBS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/libs)


# Find Optimization library
set(OPTIMIZATION_DIR ${LIBS_DIR}/Optimization/ CACHE PATH "Path to top-level directory of Optimization library (the one containing CMakeLists.txt)")
# Run the Optimization package's CMake file
add_subdirectory(${OPTIMIZATION_DIR})
get_property(OPTIMIZATION_INCLUDE_DIR TARGET Optimization PROPERTY INTERFACE_INCLUDE_DIRECTORIES)

# Find Preconditioners library
set(PRECONDITIONERS_DIR ${LIBS_DIR}/Preconditioners/ CACHE PATH "Path to top-level directory of Preconditioners library (the one containing CMakeLists.txt)")
# Run the Preconditioners package's CMake file
add_subdirectory(${PRECONDITIONERS_DIR})

# BUILD VARPRO LIBRARY

set(VARPRO_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(VARPRO_HDR_DIR ${VARPRO_INCLUDE_DIR}/VarPro)
set(VARPRO_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(VARPRO_EXAMPLES_DIR ${CMAKE_CURRENT_SOURCE_DIR}/examples)

# The VARPRO headers and and Eigen 3, SPQR, and Cholmod are all referenced by the
# header files of the VARPRO library, hence must be PUBLICLY included (i.e.
# clients using the VARPRO headers must also include these headers)
set(VARPRO_INCLUDES ${VARPRO_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIR} ${SPQR_INCLUDES} CACHE INTERNAL "")

# The VARPRO header and source files
set(VARPRO_HDRS
${VARPRO_HDR_DIR}/Solver.h
${VARPRO_HDR_DIR}/Types.h
${VARPRO_HDR_DIR}/Utils.h
${VARPRO_HDR_DIR}/Problem.h
${VARPRO_HDR_DIR}/Preconditioners.h
${VARPRO_HDR_DIR}/Symbol.h
${VARPRO_HDR_DIR}/Measurements.h
${VARPRO_HDR_DIR}/PyfgTextParser.h
${VARPRO_HDR_DIR}/MatrixManifold.h
${VARPRO_HDR_DIR}/StiefelProduct.h
${VARPRO_HDR_DIR}/ScaledStiefelProduct.h
${VARPRO_HDR_DIR}/ObliqueManifold.h
)

set(VARPRO_SRCS
${VARPRO_SOURCE_DIR}/Solver.cpp
${VARPRO_SOURCE_DIR}/Problem.cpp
${VARPRO_SOURCE_DIR}/Utils.cpp
${VARPRO_SOURCE_DIR}/Preconditioners.cpp
${VARPRO_SOURCE_DIR}/Symbol.cpp
${VARPRO_SOURCE_DIR}/PyfgTextParser.cpp
${VARPRO_SOURCE_DIR}/ScaledStiefelProduct.cpp
${VARPRO_SOURCE_DIR}/StiefelProduct.cpp
${VARPRO_SOURCE_DIR}/ObliqueManifold.cpp
)

# The VARPRO library
add_library(${PROJECT_NAME} SHARED ${VARPRO_HDRS} ${VARPRO_SRCS})
target_include_directories(${PROJECT_NAME} PUBLIC ${VARPRO_INCLUDES} ${OPTIMIZATION_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME} PUBLIC ILDL ${SPQR_LIBRARIES} ${BLAS_LIBRARIES} ${OPTIMIZATION_LIBRARIES} ${PRECONDITIONERS_LIBRARIES})

if(OPENMP_FOUND)
# Add additional compilation flags to enable OpenMP support
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS ${OpenMP_CXX_FLAGS})
set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS "-fopenmp")
endif()

# If python bindings are enabled, set warning that this is not implemented yet
if(${BUILD_PYTHON_BINDINGS})
message(WARNING "Python bindings are not implemented yet.")
endif()

# Examples
if(${BUILD_EXAMPLES})
message(STATUS "Building c++ examples")
  add_subdirectory(${VARPRO_EXAMPLES_DIR})
endif()

# export VARPRO library

set(CMAKE_EXPORT_PACKAGE_REGISTRY TRUE)
export(PACKAGE ${PROJECT_NAME})

# configure and install VARPROConfig.cmake
message(WARNING "Currently not installing VARPROConfig.cmake -- this is not implemented yet.")
