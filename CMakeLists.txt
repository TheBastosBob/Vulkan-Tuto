cmake_minimum_required(VERSION 3.24)
project(TestCube)

set(CMAKE_CXX_STANDARD 23)

find_package(Vulkan REQUIRED FATAL_ERROR)
find_package(glfw3 REQUIRED FATAL_ERROR)
find_library(Cocoa NAMES Cocoa)
find_package (glog 0.6.0 REQUIRED)
find_package(glm REQUIRED)

add_executable(TestCube main.cpp stb_image.h tiny_obj_loader.h)

if (VULKAN_FOUND)
    message(STATUS "Found Vulkan. Including and linking...")
    target_link_libraries(${PROJECT_NAME} glm::glm  Vulkan::Vulkan glfw ${Cocoa} glog::glog)
endif(VULKAN_FOUND)