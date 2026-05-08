cmake_minimum_required(VERSION 3.18)

# Common include dirs for all tests
set(TEST_INCLUDES
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
  ${VARPRO_INCLUDES}
)

macro(add_varpro_test name source)
  add_executable(${name} ${source})
  target_include_directories(${name} PRIVATE ${TEST_INCLUDES})
  target_link_libraries(${name} PRIVATE VARPROGPU)
  if(ENABLE_GPU)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(${name} PRIVATE
      CUDA::cudart CUDA::cublas CUDA::cusparse)
    target_compile_definitions(${name} PRIVATE VARPRO_HAVE_CUDA)
  endif()
endmacro()

add_varpro_test(test_schur_vs_explicit  test_schur_vs_explicit.cpp)
add_varpro_test(test_fd_gradient        test_fd_gradient.cpp)
add_varpro_test(test_convergence        test_convergence.cpp)
add_varpro_test(test_gauge_handling     test_gauge_handling.cpp)

message(STATUS "Configured tests: test_schur_vs_explicit, test_fd_gradient, test_convergence, test_gauge_handling")
add_executable(test_init_mrclam test_init_mrclam.cpp)
target_link_libraries(test_init_mrclam VARPRO VARPROGPU CUDA::cudart CUDA::cublas CUDA::cusparse)
target_include_directories(test_init_mrclam PRIVATE ${CMAKE_SOURCE_DIR}/examples)
target_compile_definitions(test_init_mrclam PRIVATE VARPRO_HAVE_CUDA)
add_executable(test_init_mrclam2 test_init_mrclam2.cpp)
target_link_libraries(test_init_mrclam2 VARPRO VARPROGPU CUDA::cudart CUDA::cublas CUDA::cusparse)
target_include_directories(test_init_mrclam2 PRIVATE ${CMAKE_SOURCE_DIR}/examples)
target_compile_definitions(test_init_mrclam2 PRIVATE VARPRO_HAVE_CUDA)
add_executable(test_explicit_mrclam test_explicit_mrclam.cpp)
target_link_libraries(test_explicit_mrclam VARPRO VARPROGPU CUDA::cudart CUDA::cublas CUDA::cusparse)
target_include_directories(test_explicit_mrclam PRIVATE ${CMAKE_SOURCE_DIR}/examples)
target_compile_definitions(test_explicit_mrclam PRIVATE VARPRO_HAVE_CUDA)
add_executable(test_mrclam2_direct test_mrclam2_direct.cpp)
target_link_libraries(test_mrclam2_direct VARPRO VARPROGPU CUDA::cudart CUDA::cublas CUDA::cusparse)
target_include_directories(test_mrclam2_direct PRIVATE ${CMAKE_SOURCE_DIR}/examples)
target_compile_definitions(test_mrclam2_direct PRIVATE VARPRO_HAVE_CUDA)
