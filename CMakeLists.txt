#
#       nnxx
#       CMakeLists.txt
#

cmake_minimum_required( VERSION 3.23 )

project( nnxx
         VERSION      0.0.1
         DESCRIPTION "nnxx"
         LANGUAGES    CXX
)

set( NNXX_COMPILE_OPTIONS -Wall -Wextra -pedantic
                          -ftemplate-backtrace-limit=0
                          -fconstexpr-backtrace-limit=0
                          -fno-exceptions -fno-rtti
                          -DUTI_RELEASE
)
if( CMAKE_BUILD_TYPE OR CMAKE_CONFIGURATION_TYPES )
        set( NNXX_DEBUG_OPT_LVL   -O0 )
        set( NNXX_RELEASE_OPT_LVL -O3 )

        set( NNXX_OPT_LVL "$<$<CONFIG:DEBUG>:${NNXX_DEBUG_OPT_LVL}>"
                          "$<$<CONFIG:DEBUG>:${NNXX_DEBUG_OPT_LVL}>"
        )
endif()

add_library( nnxx INTERFACE )
add_library( nnxx::nnxx ALIAS nnxx )

set_property( TARGET nnxx PROPERTY SOVERSION "${PROJECT_VERSION}" )

if( CMAKE_BUILD_TYPE OR CMAKE_CONFIGURATION_TYPES )
        set( NNXX_DEBUG_OPTIONS   -DNNXX_DEBUG  )
        set( NNXX_RELEASE_OPTIONS -DNNXX_RELESE )

        target_compile_options( nnxx INTERFACE "$<$<CONFIG:DEBUG>:${NNXX_DEBUG_OPTIONS}>" )
        target_compile_options( nnxx INTERFACE "$<$<CONFIG:RELEASE>:${NNXX_RELEASE_OPTIONS}>" )
endif()

target_compile_options( nnxx INTERFACE -std=c++26                )
target_compile_options( nnxx INTERFACE "${NNXX_COMPILE_OPTIONS}" )

target_sources( nnxx PUBLIC
        FILE_SET HEADERS
        BASE_DIRS ${PROJECT_SOURCE_DIR}/include
)

include( FetchContent )

FetchContent_Declare(
        uti
        GIT_REPOSITORY https://github.com/eddieavd/uti.git
        GIT_TAG        master
)
FetchContent_GetProperties( uti )

if( NOT uti_POPULATED )
        set( FETCHCONTENT_QUIET NO )
        FetchContent_MakeAvailable( uti )
endif()

FetchContent_Declare(
        haze
        GIT_REPOSITORY git@github.com:eddieavd/haze_v2.git
        GIT_TAG        refactor
)
FetchContent_GetProperties( haze )

if( NOT haze_POPULATED )
        set( FETCHCONTENT_QUIET NO )
        FetchContent_MakeAvailable( haze )
endif()

target_link_libraries( nnxx INTERFACE
        uti_core::uti_core
        haze::haze
)

if( NNXX_TEST )
        add_subdirectory( test )
endif()

if( NNXX_BENCH )
        add_subdirectory( bench )
endif()

add_subdirectory( examples )
