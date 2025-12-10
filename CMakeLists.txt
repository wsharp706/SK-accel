cmake_minimum_required(VERSION 4.2)
project(LinAlg)
include(CTest)

# --------------------libraries--------------------

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

