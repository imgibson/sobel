cmake_minimum_required(VERSION 3.8)

project(sobel)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS OFF)

# Add library
add_library(sobel_filter STATIC
   ${CMAKE_CURRENT_SOURCE_DIR}/include/sobel_filter.h
   ${CMAKE_CURRENT_SOURCE_DIR}/source/sobel_filter.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/source/sobel_filter_sse2.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/source/sobel_filter_avx2.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/source/sobel_filter_avx512.cpp
)

if(WIN32)
	# SSE2 is on by default for 64 bit platforms on windows (enabling it anyway issues a warning for VS2019)
	if("${CMAKE_SIZEOF_VOID_P}" STREQUAL "4")
		set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/source/sobel_filter_sse2.cpp PROPERTIES COMPILE_OPTIONS "/arch:SSE2")
	endif()
	set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/source/sobel_filter_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
	set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/source/sobel_filter_avx512.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
elseif(UNIX)
	set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/source/sobel_filter_sse2.cpp PROPERTIES COMPILE_OPTIONS "-msse2")
	set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/source/sobel_filter_avx2.cpp PROPERTIES COMPILE_OPTIONS "-msse2;-mavx;-mavx2;-mfma")
	set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/source/sobel_filter_avx512.cpp PROPERTIES COMPILE_OPTIONS "-msse2;-msse3;-mavx2;-mfma;-mavx512f")
endif()

# Export and use include directories
target_include_directories(sobel_filter PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
