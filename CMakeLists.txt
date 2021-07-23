#----------------------------------------------------------------------
# CMakeList.txt: CMake Project

cmake_minimum_required (VERSION 3.8)

project ("boatDetector")

# Look for OpenCV libraries

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OPENCV_LIB_DIR})

# Generate the executable
add_executable ("datasetPreprocessing" "src/preprocess.cpp")
add_executable ("yoloPrediction" "src/yoloPredictor.cpp")


add_subdirectory(include)


target_link_libraries("datasetPreprocessing" ${OpenCV_LIBS} include)
target_link_libraries("yoloPrediction" ${OpenCV_LIBS} include)
