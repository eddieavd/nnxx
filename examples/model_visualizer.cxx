//
//
//      nnxx
//      examples/model_visualizer.cxx
//

#include <raylib.h>


int main ()
{
        static constexpr int    fps {  60 } ;
        static constexpr int  width { 800 } ;
        static constexpr int height { 600 } ;

        InitWindow( width, height, "nnxx viz" ) ;
        SetTargetFPS( fps ) ;

        while( !WindowShouldClose() )
        {
                BeginDrawing() ;
                ClearBackground( BLACK ) ;
                EndDrawing() ;
        }
        CloseWindow() ;


         return 0 ;
}
