cmake_minimum_required(VERSION 2.8.12)
project(tensorc-library)

if(NOT CMAKE_BUILD_TYPE)
	if(CMAKE_CONFIGURATION_TYPES)
	set(CMAKE_BUILD_TYPE ${CMAKE_CONFIGURATION_TYPES})
	else()
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
	endif()
endif()

SET(CMAKE_C_COMPILER "clang")

set(CMAKE_C_FLAGS "-std=c11 ${CMAKE_C_FLAGS} -O0")

#set project directories.
set(PROJECT_ROOT_PATH "${CMAKE_SOURCE_DIR}")
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/bin")
set(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}/lib")

set(INCLUDE_DIR ${PROJECT_ROOT_PATH}/include)
set(LINK_DIR /usr/local/lib ${CMAKE_BINARY_DIR}/lib)

include_directories(${INCLUDE_DIR}  )

link_directories(${LINK_DIR} )

add_subdirectory(lib)
