# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.20

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
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
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /media/mattiascarpa/Archivio/Programming_Workspace/Computer_Vision_UniPD/00-ScarpaMattia_2005826_FinalProject

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /media/mattiascarpa/Archivio/Programming_Workspace/Computer_Vision_UniPD/00-ScarpaMattia_2005826_FinalProject/build

# Include any dependencies generated for this target.
include include/CMakeFiles/include.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include include/CMakeFiles/include.dir/compiler_depend.make

# Include the progress variables for this target.
include include/CMakeFiles/include.dir/progress.make

# Include the compile flags for this target's objects.
include include/CMakeFiles/include.dir/flags.make

include/CMakeFiles/include.dir/labeltxt.cpp.o: include/CMakeFiles/include.dir/flags.make
include/CMakeFiles/include.dir/labeltxt.cpp.o: ../include/labeltxt.cpp
include/CMakeFiles/include.dir/labeltxt.cpp.o: include/CMakeFiles/include.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/mattiascarpa/Archivio/Programming_Workspace/Computer_Vision_UniPD/00-ScarpaMattia_2005826_FinalProject/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object include/CMakeFiles/include.dir/labeltxt.cpp.o"
	cd /media/mattiascarpa/Archivio/Programming_Workspace/Computer_Vision_UniPD/00-ScarpaMattia_2005826_FinalProject/build/include && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT include/CMakeFiles/include.dir/labeltxt.cpp.o -MF CMakeFiles/include.dir/labeltxt.cpp.o.d -o CMakeFiles/include.dir/labeltxt.cpp.o -c /media/mattiascarpa/Archivio/Programming_Workspace/Computer_Vision_UniPD/00-ScarpaMattia_2005826_FinalProject/include/labeltxt.cpp

include/CMakeFiles/include.dir/labeltxt.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/include.dir/labeltxt.cpp.i"
	cd /media/mattiascarpa/Archivio/Programming_Workspace/Computer_Vision_UniPD/00-ScarpaMattia_2005826_FinalProject/build/include && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/mattiascarpa/Archivio/Programming_Workspace/Computer_Vision_UniPD/00-ScarpaMattia_2005826_FinalProject/include/labeltxt.cpp > CMakeFiles/include.dir/labeltxt.cpp.i

include/CMakeFiles/include.dir/labeltxt.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/include.dir/labeltxt.cpp.s"
	cd /media/mattiascarpa/Archivio/Programming_Workspace/Computer_Vision_UniPD/00-ScarpaMattia_2005826_FinalProject/build/include && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/mattiascarpa/Archivio/Programming_Workspace/Computer_Vision_UniPD/00-ScarpaMattia_2005826_FinalProject/include/labeltxt.cpp -o CMakeFiles/include.dir/labeltxt.cpp.s

include/CMakeFiles/include.dir/dataugmentation.cpp.o: include/CMakeFiles/include.dir/flags.make
include/CMakeFiles/include.dir/dataugmentation.cpp.o: ../include/dataugmentation.cpp
include/CMakeFiles/include.dir/dataugmentation.cpp.o: include/CMakeFiles/include.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/mattiascarpa/Archivio/Programming_Workspace/Computer_Vision_UniPD/00-ScarpaMattia_2005826_FinalProject/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object include/CMakeFiles/include.dir/dataugmentation.cpp.o"
	cd /media/mattiascarpa/Archivio/Programming_Workspace/Computer_Vision_UniPD/00-ScarpaMattia_2005826_FinalProject/build/include && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT include/CMakeFiles/include.dir/dataugmentation.cpp.o -MF CMakeFiles/include.dir/dataugmentation.cpp.o.d -o CMakeFiles/include.dir/dataugmentation.cpp.o -c /media/mattiascarpa/Archivio/Programming_Workspace/Computer_Vision_UniPD/00-ScarpaMattia_2005826_FinalProject/include/dataugmentation.cpp

include/CMakeFiles/include.dir/dataugmentation.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/include.dir/dataugmentation.cpp.i"
	cd /media/mattiascarpa/Archivio/Programming_Workspace/Computer_Vision_UniPD/00-ScarpaMattia_2005826_FinalProject/build/include && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/mattiascarpa/Archivio/Programming_Workspace/Computer_Vision_UniPD/00-ScarpaMattia_2005826_FinalProject/include/dataugmentation.cpp > CMakeFiles/include.dir/dataugmentation.cpp.i

include/CMakeFiles/include.dir/dataugmentation.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/include.dir/dataugmentation.cpp.s"
	cd /media/mattiascarpa/Archivio/Programming_Workspace/Computer_Vision_UniPD/00-ScarpaMattia_2005826_FinalProject/build/include && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/mattiascarpa/Archivio/Programming_Workspace/Computer_Vision_UniPD/00-ScarpaMattia_2005826_FinalProject/include/dataugmentation.cpp -o CMakeFiles/include.dir/dataugmentation.cpp.s

# Object files for target include
include_OBJECTS = \
"CMakeFiles/include.dir/labeltxt.cpp.o" \
"CMakeFiles/include.dir/dataugmentation.cpp.o"

# External object files for target include
include_EXTERNAL_OBJECTS =

include/libinclude.a: include/CMakeFiles/include.dir/labeltxt.cpp.o
include/libinclude.a: include/CMakeFiles/include.dir/dataugmentation.cpp.o
include/libinclude.a: include/CMakeFiles/include.dir/build.make
include/libinclude.a: include/CMakeFiles/include.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/media/mattiascarpa/Archivio/Programming_Workspace/Computer_Vision_UniPD/00-ScarpaMattia_2005826_FinalProject/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX static library libinclude.a"
	cd /media/mattiascarpa/Archivio/Programming_Workspace/Computer_Vision_UniPD/00-ScarpaMattia_2005826_FinalProject/build/include && $(CMAKE_COMMAND) -P CMakeFiles/include.dir/cmake_clean_target.cmake
	cd /media/mattiascarpa/Archivio/Programming_Workspace/Computer_Vision_UniPD/00-ScarpaMattia_2005826_FinalProject/build/include && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/include.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
include/CMakeFiles/include.dir/build: include/libinclude.a
.PHONY : include/CMakeFiles/include.dir/build

include/CMakeFiles/include.dir/clean:
	cd /media/mattiascarpa/Archivio/Programming_Workspace/Computer_Vision_UniPD/00-ScarpaMattia_2005826_FinalProject/build/include && $(CMAKE_COMMAND) -P CMakeFiles/include.dir/cmake_clean.cmake
.PHONY : include/CMakeFiles/include.dir/clean

include/CMakeFiles/include.dir/depend:
	cd /media/mattiascarpa/Archivio/Programming_Workspace/Computer_Vision_UniPD/00-ScarpaMattia_2005826_FinalProject/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /media/mattiascarpa/Archivio/Programming_Workspace/Computer_Vision_UniPD/00-ScarpaMattia_2005826_FinalProject /media/mattiascarpa/Archivio/Programming_Workspace/Computer_Vision_UniPD/00-ScarpaMattia_2005826_FinalProject/include /media/mattiascarpa/Archivio/Programming_Workspace/Computer_Vision_UniPD/00-ScarpaMattia_2005826_FinalProject/build /media/mattiascarpa/Archivio/Programming_Workspace/Computer_Vision_UniPD/00-ScarpaMattia_2005826_FinalProject/build/include /media/mattiascarpa/Archivio/Programming_Workspace/Computer_Vision_UniPD/00-ScarpaMattia_2005826_FinalProject/build/include/CMakeFiles/include.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : include/CMakeFiles/include.dir/depend
