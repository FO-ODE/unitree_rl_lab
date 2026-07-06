cmake_minimum_required(VERSION 3.12)
project(go2_controller)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(UNITREE_SDK2_ROOT "$ENV{HOME}/SDK/unitree_sdk2" CACHE PATH "Path to unitree_sdk2")
set(UNITREE_SDK2_ARCH "x86_64" CACHE STRING "unitree_sdk2 library architecture")

execute_process(
  COMMAND python3 -c "import torch; print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX_PATH})

find_package(Boost REQUIRED COMPONENTS program_options)
find_package(yaml-cpp REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(Torch REQUIRED)

include_directories(
  /usr/include/eigen3
  /usr/local/include/ddscxx # dds
  /usr/local/include/iceoryx/v2.0.2 # dds
  ${UNITREE_SDK2_ROOT}/include
  ${UNITREE_SDK2_ROOT}/thirdparty/include
  ${UNITREE_SDK2_ROOT}/thirdparty/include/ddsc
  ${UNITREE_SDK2_ROOT}/thirdparty/include/ddscxx
  include
  ${PROJECT_SOURCE_DIR}/../../thirdparty/onnxruntime-linux-x64-1.22.0/include
  ${PROJECT_SOURCE_DIR}/../../include/
)

link_libraries(
  ${UNITREE_SDK2_ROOT}/lib/${UNITREE_SDK2_ARCH}/libunitree_sdk2.a
  ${UNITREE_SDK2_ROOT}/thirdparty/lib/${UNITREE_SDK2_ARCH}/libddsc.so
  ${UNITREE_SDK2_ROOT}/thirdparty/lib/${UNITREE_SDK2_ARCH}/libddscxx.so
  rt pthread # dds
  libboost_program_options.a libyaml-cpp.a fmt
  ${PROJECT_SOURCE_DIR}/../../thirdparty/onnxruntime-linux-x64-1.22.0/lib/libonnxruntime.so.1.22.0
)

set(CMAKE_BUILD_RPATH "${UNITREE_SDK2_ROOT}/thirdparty/lib/${UNITREE_SDK2_ARCH};${PROJECT_SOURCE_DIR}/../../thirdparty/onnxruntime-linux-x64-1.22.0/lib")
set(CMAKE_INSTALL_RPATH "${UNITREE_SDK2_ROOT}/thirdparty/lib/${UNITREE_SDK2_ARCH};${PROJECT_SOURCE_DIR}/../../thirdparty/onnxruntime-linux-x64-1.22.0/lib")

file(GLOB_RECURSE ADD_SRC_LIST
  src/*.cpp
)
list(REMOVE_ITEM ADD_SRC_LIST ${PROJECT_SOURCE_DIR}/src/jit_runner.cpp)

add_library(go2_jit_runner src/jit_runner.cpp)
target_link_libraries(go2_jit_runner PRIVATE ${TORCH_LIBRARIES})
target_compile_options(go2_jit_runner PRIVATE ${TORCH_CXX_FLAGS})

add_library(${PROJECT_NAME}_lib ${ADD_SRC_LIST})
ament_target_dependencies(${PROJECT_NAME}_lib rclcpp std_msgs)
link_libraries(${PROJECT_NAME}_lib)

add_executable(go2_ctrl main.cpp)
ament_target_dependencies(go2_ctrl rclcpp std_msgs)
target_link_libraries(go2_ctrl go2_jit_runner)
target_link_options(go2_ctrl PRIVATE "-Wl,--disable-new-dtags")
