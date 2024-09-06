cmake_minimum_required(VERSION 3.10)
project(NeuralCPP)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(
    include
)

set(SOURCES
    src/main.cpp
    src/activation_functions.cpp
    src/layer.cpp
    src/neural_network.cpp
    src/neuron.cpp
    src/weight_utils.cpp
)

add_executable(${PROJECT_NAME} ${SOURCES})
