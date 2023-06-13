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

set(COSRC "src/main.cpp" "src/enum_gpu.cu")
file(GLOB_RECURSE CXX_SOURCES "src/patch_match/*.cpp")
file(GLOB_RECURSE INCLUDE_FILES "include/patch_match/*.h")
file(GLOB_RECURSE CUDA_SOURCES "src/patch_match/*.cu")




include_directories(${CUDA_INCLUDE_DIRS})

set(PatchMatchSrc ${COSRC} ${C_SOURCES} ${CXX_SOURCES} ${CC_SOURCES} ${INCLUDE_FILES} ${CUDA_SOURCES})

file(GLOB_RECURSE CXX_SOURCES "src/multi_frame/*.cpp")
file(GLOB_RECURSE INCLUDE_FILES "include/multi_frame/*.h")
file(GLOB_RECURSE CUDA_SOURCES "src/multi_frame/*.cu")

set(MultiFrameSrc ${COSRC} ${CXX_SOURCES} ${INCLUDE_FILES} ${CUDA_SOURCES})

#CUDA_ADD_EXECUTABLE(PatchMatch ${CUDA_SOURCES})
ADD_EXECUTABLE(PatchMatch ${PatchMatchSrc})
ADD_EXECUTABLE(MultiFrame ${MultiFrameSrc})

target_include_directories(PatchMatch PUBLIC "include/patch_match")
target_include_directories(MultiFrame PUBLIC "include/multi_frame")

#add_compile_options(-G)

set_target_properties(PatchMatch PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(MultiFrame PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Add CUDA include directories and libraries
target_include_directories(PatchMatch PUBLIC ${CUDA_INCLUDE_DIRS})
target_link_libraries(PatchMatch PUBLIC ${CUDA_LIBRARIES})
target_include_directories(MultiFrame PUBLIC ${CUDA_INCLUDE_DIRS})
target_link_libraries(MultiFrame PUBLIC ${CUDA_LIBRARIES})

# Add any other necessary compile options

# Link CUDA runtime library
target_link_libraries(PatchMatch PRIVATE cuda)
target_link_libraries(MultiFrame PRIVATE cuda)

# Ensure that main.cpp can access functions in main.cu
target_include_directories(PatchMatch PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(MultiFrame PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# Link necessary libraries
target_link_libraries(PatchMatch PUBLIC ${OpenCV_LIBS})
target_link_libraries(MultiFrame PUBLIC ${OpenCV_LIBS})
