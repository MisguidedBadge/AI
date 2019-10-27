cmake_minimum_required (VERSION 2.6)
project (NeuralNetworkStuff)
# The version number.
set (NEURALNETWORK_VERSION_MAJOR 1)
set (NEURALNETWORK_VERSION_MINOR 0)
 
# add the binary tree to the search path for include files
include_directories("${PROJECT_BINARY_DIR}")
 
# add the executable
add_executable(Driver Main.cpp Layer.cpp Activation.cpp)
