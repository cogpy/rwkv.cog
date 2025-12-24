cmake_minimum_required(VERSION 3.21)

# Example: Cognitive Reasoning Demo
add_executable(cognitive_reasoning_demo cognitive_reasoning_demo.c)
target_link_libraries(cognitive_reasoning_demo PRIVATE ggml rwkv)
target_include_directories(cognitive_reasoning_demo PRIVATE 
    ${CMAKE_SOURCE_DIR} 
    ${CMAKE_SOURCE_DIR}/ggml/include 
    ${CMAKE_SOURCE_DIR}/ggml/src
)