#----------------------------------------------------------------------
# CMakeList.txt: CMake Project

cmake_minimum_required (VERSION 3.8)

project ("boatDetector")

# Look for OpenCV libraries

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OPENCV_LIB_DIR})

# Generate the executable
add_executable ("labeling_Augmentation" "src/main.cpp")


add_subdirectory(include)

#if(FOO_USE_EXTERNAL_JSON)
#  find_package(nlohmann_json 3.2.0 REQUIRED)
#else()
#  set(JSON_BuildTests OFF CACHE INTERNAL "")
#  add_subdirectory(include/nlohmann_json)
#endif()

target_link_libraries("labeling_Augmentation" ${OpenCV_LIBS} include) #nlohmann_json

