#
#       nnxx
#       examples/CMakeLists.txt
#

add_executable( nnxx_demo demo.cxx )

target_compile_options( nnxx_demo PRIVATE -std=c++26                )
target_compile_options( nnxx_demo PRIVATE "${NNXX_OPT_LVL}"         )
target_compile_options( nnxx_demo PRIVATE "${NNXX_COMPILE_OPTIONS}" )

set( RAYLIB_VERSION 5.5 )

FetchContent_Declare(
        raylib
        DOWNLOAD_EXTRACT_TIMESTAMP OFF
        URL https://github.com/raysan5/raylib/archive/refs/tags/${RAYLIB_VERSION}.tar.gz
)
FetchContent_GetProperties( raylib )

if( NOT raylib_POPULATED )
        set( FETCHCONTENT_QUIET NO )
        FetchContent_MakeAvailable( raylib )
endif()

target_link_libraries( nnxx_demo PRIVATE
        nnxx::nnxx
        uti_core::uti_core
        raylib
)

# target_link_libraries( nnxx_demo PRIVATE "-framework IOKit"  )
# target_link_libraries( nnxx_demo PRIVATE "-framework Cocoa"  )
# target_link_libraries( nnxx_demo PRIVATE "-framework OpenGL" )
