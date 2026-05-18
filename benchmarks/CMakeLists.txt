cmake_minimum_required(VERSION 3.18)

set(BENCH_INCLUDES
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
  ${VARPRO_INCLUDES}
)

macro(add_varpro_bench name source)
  add_executable(${name} ${source})
  target_include_directories(${name} PRIVATE ${BENCH_INCLUDES})
  target_link_libraries(${name} PRIVATE VARPROGPU)
  if(ENABLE_GPU)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(${name} PRIVATE
      CUDA::cudart CUDA::cublas CUDA::cusparse)
    target_compile_definitions(${name} PRIVATE VARPRO_HAVE_CUDA)
  endif()
endmacro()

add_varpro_bench(benchmark_schur     benchmark_schur.cpp)
add_varpro_bench(benchmark_solver    benchmark_solver.cpp)
add_varpro_bench(benchmark_precompute benchmark_precompute.cpp)

message(STATUS "Configured benchmarks: benchmark_schur, benchmark_solver, benchmark_precompute")
