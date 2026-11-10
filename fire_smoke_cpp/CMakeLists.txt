cmake_minimum_required(VERSION 3.20)
project(fire_smoke_cpp VERSION 1.2 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(OpenGL REQUIRED)
find_package(SDL2 REQUIRED)
find_package(SDL2_ttf REQUIRED)
find_package(GLEW REQUIRED)
add_executable(fire_smoke src/main.cpp)
target_link_libraries(fire_smoke PRIVATE SDL2::SDL2 SDL2_ttf::SDL2_ttf OpenGL::GL GLEW::GLEW)
target_compile_options(fire_smoke PRIVATE -Wall -Wextra -Wpedantic)
add_custom_command(TARGET fire_smoke POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_SOURCE_DIR}/shaders" "$<TARGET_FILE_DIR:fire_smoke>/shaders"
  COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_SOURCE_DIR}/assets" "$<TARGET_FILE_DIR:fire_smoke>/assets")
