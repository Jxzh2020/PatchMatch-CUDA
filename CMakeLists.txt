cmake_minimum_required(VERSION 3.25)
project(PatchMatch CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)

find_package(OpenCV CONFIG REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
if(NOT OpenCV_FOUND)
    message(ERROR " OpenCV not found!")
endif(NOT OpenCV_FOUND)

find_package(CUDA)

file(GLOB_RECURSE C_SOURCES "src/*.c")
file(GLOB_RECURSE CXX_SOURCES "src/*.cpp")
file(GLOB_RECURSE CC_SOURCES "src/*.cc")
file(GLOB_RECURSE INCLUDE_FILES "include/*.h")
file(GLOB_RECURSE CUDA_SOURCES "src/*.cu")



include_directories(include)
include_directories(${CUDA_INCLUDE_DIRS})

set(PatchMatchSrc ${C_SOURCES} ${CXX_SOURCES} ${CC_SOURCES} ${INCLUDE_FILES} ${CUDA_SOURCES})

#CUDA_ADD_EXECUTABLE(PatchMatch ${CUDA_SOURCES})
ADD_EXECUTABLE(PatchMatch ${PatchMatchSrc})

#add_compile_options(-G)

set_target_properties(PatchMatch PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Add CUDA include directories and libraries
target_include_directories(PatchMatch PUBLIC ${CUDA_INCLUDE_DIRS})
target_link_libraries(PatchMatch PUBLIC ${CUDA_LIBRARIES})

# Add any other necessary compile options

# Link CUDA runtime library
target_link_libraries(PatchMatch PRIVATE cuda)

# Ensure that main.cpp can access functions in main.cu
target_include_directories(PatchMatch PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Link necessary libraries
target_link_libraries(PatchMatch PUBLIC ${OpenCV_LIBS})
