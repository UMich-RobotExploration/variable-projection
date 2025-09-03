add_executable(varpro_example main.cpp)
target_link_libraries(varpro_example VARPRO)

if(${ENABLE_PROFILING})
  target_compile_definitions(varpro_example PRIVATE GPERFTOOLS)
  target_link_libraries(varpro_example profiler)
endif()

if(${PERFORM_EXPERIMENTS})
  message(STATUS "Building paper_experiments executable")
  add_executable(paper_experiments paper_experiments.cpp)
  target_link_libraries(paper_experiments VARPRO)
  target_include_directories(paper_experiments PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  if(${ENABLE_PROFILING})
    target_compile_definitions(paper_experiments PRIVATE GPERFTOOLS)
    target_link_libraries(paper_experiments profiler)
  endif()
endif()

set(EXAMPLE_PYFG_FILES
"config.json"
# raslam datasets
"data/raslam/factor_graph_small.pyfg"
"data/raslam/mrclam/mrclam2/mrclam2.pyfg"
"data/raslam/mrclam/mrclam3a/mrclam3a.pyfg"
"data/raslam/mrclam/mrclam3b/mrclam3b.pyfg"
"data/raslam/mrclam/mrclam4/mrclam4.pyfg"
"data/raslam/mrclam/mrclam5a/mrclam5a.pyfg"
"data/raslam/mrclam/mrclam5b/mrclam5b.pyfg"
"data/raslam/mrclam/mrclam5c/mrclam5c.pyfg"
"data/raslam/mrclam/mrclam6/mrclam6.pyfg"
"data/raslam/mrclam/mrclam7/mrclam7.pyfg"
"data/raslam/plaza1.pyfg"
"data/raslam/plaza2.pyfg"
"data/raslam/single_drone.pyfg"
"data/raslam/tiers.pyfg"
# pgo datasets
# sfm datasets
# snl datasets
)

message(STATUS "Building main varpro_example command-line executable in directory ${EXECUTABLE_OUTPUT_PATH}\n")

## Copy example PyFG files to build directory
foreach(PYFG_FILE ${EXAMPLE_PYFG_FILES})
# try to copy and if not, print a warning
    configure_file(${PYFG_FILE} ${EXECUTABLE_OUTPUT_PATH}/${PYFG_FILE} COPYONLY)
endforeach()
