cmake_minimum_required(VERSION 3.28)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
project(rigMath-UTIL)

# --------------------dependencies-------------------

include(CTest)
find_package(AdaptiveCpp CONFIG REQUIRED)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${ACPP_SOURCE_ROOT}/cmake)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(NOT ACPP_DEBUG_LEVEL)
  if(CMAKE_BUILD_TYPE MATCHES "Debug")
    set(ACPP_DEBUG_LEVEL 3 CACHE STRING
      "Choose the debug level, options are: 0 (no debug), 1 (print errors), 2 (also print warnings), 3 (also print general information)"
FORCE)
  else()
    set(ACPP_DEBUG_LEVEL 2 CACHE STRING
      "Choose the debug level, options are: 0 (no debug), 1 (print errors), 2 (also print warnings), 3 (also print general information)"
FORCE)
  endif()
endif()

#add_compile_definitions(HIPSYCL_DEBUG_LEVEL="${HIPSYCL_DEBUG_LEVEL}")
#Use add_definitions for now for older cmake versions
cmake_policy(SET CMP0005 NEW)
add_definitions(-DHIPSYCL_DEBUG_LEVEL=${ACPP_DEBUG_LEVEL})

if(WIN32)
  add_definitions(-D_USE_MATH_DEFINES)
endif()

include_directories(${ACPP_BINARY_ROOT} ${ACPP_SOURCE_ROOT})

#subdirs(bruteforce_nbody)

# --------------------libraries--------------------

# vect library
add_library(vect_lib INTERFACE)
target_include_directories(vect_lib
    INTERFACE
        include/vect
        include/shared_util
        sycl
        
)


#target_sources(vect_lib
#    PRIVATE
#        src/vect.cpp
#    PUBLIC
#        FILE_SET HEADERS
#        BASE_DIRS
#            include
#        FILES
#            include/vect.h
#            include/customexceptions.h
#)

add_executable(VECT--TEST)
target_sources(VECT--TEST
    PRIVATE
        tests/VECT--TEST.cpp
)

target_link_libraries(VECT--TEST
    PRIVATE
        vect_lib
)

add_test(
    NAME VECT--TEST
    COMMAND VECT--TEST
)

add_sycl_to_target(TARGET VECT--TEST)

#[[

# matrix library
add_library(matrix_lib)
target_sources(matrix_lib
    PRIVATE
        src/matrix.cpp
    PUBLIC
        FILE_SET HEADERS
        BASE_DIRS
            include
        FILES
            include/matrix.h
            include/customexceptions.h
)
#target_include_directories(matrix_lib PUBLIC include)

# model library
add_library(model_lib)
target_sources(model_lib
    PRIVATE
        src/model.cpp
        src/stats.cpp
    PUBLIC
        FILE_SET HEADERS
        BASE_DIRS include
        FILES include/model.h
        FILES include/stats.h
        FILES include/matrix.h
        FILES include/customexceptions.h
)

target_link_libraries(model_lib
    PUBLIC
        matrix_lib
)

# --------------------testing executables--------------------

# matrix test
add_executable(matrixTest)
target_sources(matrixTest
    PRIVATE
        tests/matrixTest.cpp
)

target_link_libraries(matrixTest
    PRIVATE
        matrix_lib
)

add_test(
    NAME matrixTest
    COMMAND matrixTest
)

# model test
add_executable(modelTest)
target_sources(modelTest
    PRIVATE
        tests/modelTest.cpp
)

target_link_libraries(modelTest
    PRIVATE
        model_lib
)

add_sycl_to_target( 
    TARGET 
        matrixTest    
)

]]



