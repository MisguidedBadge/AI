# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.15

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/varunyadav/GitHubRepos/AI/NeuralNetworkStuff/NeuralNetworkStuff

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/varunyadav/GitHubRepos/AI/NeuralNetworkStuff/build

# Include any dependencies generated for this target.
include CMakeFiles/Driver.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/Driver.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Driver.dir/flags.make

CMakeFiles/Driver.dir/Main.cpp.o: CMakeFiles/Driver.dir/flags.make
CMakeFiles/Driver.dir/Main.cpp.o: /Users/varunyadav/GitHubRepos/AI/NeuralNetworkStuff/NeuralNetworkStuff/Main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/varunyadav/GitHubRepos/AI/NeuralNetworkStuff/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/Driver.dir/Main.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Driver.dir/Main.cpp.o -c /Users/varunyadav/GitHubRepos/AI/NeuralNetworkStuff/NeuralNetworkStuff/Main.cpp

CMakeFiles/Driver.dir/Main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Driver.dir/Main.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/varunyadav/GitHubRepos/AI/NeuralNetworkStuff/NeuralNetworkStuff/Main.cpp > CMakeFiles/Driver.dir/Main.cpp.i

CMakeFiles/Driver.dir/Main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Driver.dir/Main.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/varunyadav/GitHubRepos/AI/NeuralNetworkStuff/NeuralNetworkStuff/Main.cpp -o CMakeFiles/Driver.dir/Main.cpp.s

CMakeFiles/Driver.dir/Layer.cpp.o: CMakeFiles/Driver.dir/flags.make
CMakeFiles/Driver.dir/Layer.cpp.o: /Users/varunyadav/GitHubRepos/AI/NeuralNetworkStuff/NeuralNetworkStuff/Layer.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/varunyadav/GitHubRepos/AI/NeuralNetworkStuff/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/Driver.dir/Layer.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Driver.dir/Layer.cpp.o -c /Users/varunyadav/GitHubRepos/AI/NeuralNetworkStuff/NeuralNetworkStuff/Layer.cpp

CMakeFiles/Driver.dir/Layer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Driver.dir/Layer.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/varunyadav/GitHubRepos/AI/NeuralNetworkStuff/NeuralNetworkStuff/Layer.cpp > CMakeFiles/Driver.dir/Layer.cpp.i

CMakeFiles/Driver.dir/Layer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Driver.dir/Layer.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/varunyadav/GitHubRepos/AI/NeuralNetworkStuff/NeuralNetworkStuff/Layer.cpp -o CMakeFiles/Driver.dir/Layer.cpp.s

CMakeFiles/Driver.dir/Activation.cpp.o: CMakeFiles/Driver.dir/flags.make
CMakeFiles/Driver.dir/Activation.cpp.o: /Users/varunyadav/GitHubRepos/AI/NeuralNetworkStuff/NeuralNetworkStuff/Activation.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/varunyadav/GitHubRepos/AI/NeuralNetworkStuff/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/Driver.dir/Activation.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Driver.dir/Activation.cpp.o -c /Users/varunyadav/GitHubRepos/AI/NeuralNetworkStuff/NeuralNetworkStuff/Activation.cpp

CMakeFiles/Driver.dir/Activation.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Driver.dir/Activation.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/varunyadav/GitHubRepos/AI/NeuralNetworkStuff/NeuralNetworkStuff/Activation.cpp > CMakeFiles/Driver.dir/Activation.cpp.i

CMakeFiles/Driver.dir/Activation.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Driver.dir/Activation.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/varunyadav/GitHubRepos/AI/NeuralNetworkStuff/NeuralNetworkStuff/Activation.cpp -o CMakeFiles/Driver.dir/Activation.cpp.s

# Object files for target Driver
Driver_OBJECTS = \
"CMakeFiles/Driver.dir/Main.cpp.o" \
"CMakeFiles/Driver.dir/Layer.cpp.o" \
"CMakeFiles/Driver.dir/Activation.cpp.o"

# External object files for target Driver
Driver_EXTERNAL_OBJECTS =

Driver: CMakeFiles/Driver.dir/Main.cpp.o
Driver: CMakeFiles/Driver.dir/Layer.cpp.o
Driver: CMakeFiles/Driver.dir/Activation.cpp.o
Driver: CMakeFiles/Driver.dir/build.make
Driver: CMakeFiles/Driver.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/varunyadav/GitHubRepos/AI/NeuralNetworkStuff/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable Driver"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Driver.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Driver.dir/build: Driver

.PHONY : CMakeFiles/Driver.dir/build

CMakeFiles/Driver.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Driver.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Driver.dir/clean

CMakeFiles/Driver.dir/depend:
	cd /Users/varunyadav/GitHubRepos/AI/NeuralNetworkStuff/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/varunyadav/GitHubRepos/AI/NeuralNetworkStuff/NeuralNetworkStuff /Users/varunyadav/GitHubRepos/AI/NeuralNetworkStuff/NeuralNetworkStuff /Users/varunyadav/GitHubRepos/AI/NeuralNetworkStuff/build /Users/varunyadav/GitHubRepos/AI/NeuralNetworkStuff/build /Users/varunyadav/GitHubRepos/AI/NeuralNetworkStuff/build/CMakeFiles/Driver.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Driver.dir/depend
