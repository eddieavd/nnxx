//
//
//      nnxx
//      examples/model_visualizer.cxx
//

#include <cstdio>

#include <nnxx/layer/activation.hxx>
#include <nnxx/model/dense_network.hxx>
#include <nnxx/model/visualize.hxx>

#include <raylib.h>


template< typename Model >
constexpr void draw_model ( Model const & model, int width, int height ) noexcept ;

using model_type = nnxx::dense_neural_network< float, nnxx::relu_layer, 2, 4, 1 > ;

int main ()
{
        static constexpr int    fps {  60 } ;
        static constexpr int  width { 800 } ;
        static constexpr int height { 600 } ;

        model_type model ;

        nnxx::model_visualizer visualizer( model ) ;

        InitWindow( width, height, "nnxx viz" ) ;
        SetTargetFPS( fps ) ;

        while( !WindowShouldClose() )
        {
                BeginDrawing() ;
                ClearBackground( BLACK ) ;
                draw_model( model, width, height ) ;
                EndDrawing() ;
        }
        CloseWindow() ;


         return 0 ;
}

template< typename Model >
constexpr void draw_model ( Model const & model, int width, int height ) noexcept
{
        static constexpr int depth { Model::depth } ;

        uti::array< int, depth + 1 > layer_ns {} ;

        layer_ns[ 0 ] = model.input_n ;

        int idx { 1 } ;

        uti::for_each( model.layers(), [ & ]( auto const & layer ){ layer_ns[ idx ] = layer.size ; ++idx ; } ) ;

        float node_radius = 32 ;

//      Color color_zero {  18, 255,  18, 128 } ;
        Color color_one  {  18,  18, 255, 128 } ;
//      Color color_huge { 255,  18,  18, 128 } ;

        Vector2 draw_pos{ 128, static_cast< float >( height ) / 2.0f } ;

        for( int i = 0; i < depth + 1; ++i, draw_pos.x += node_radius * 4 )
        {
                if( layer_ns[ i ] % 2 == 0 )
                {
                        {
                                Vector2 circ_center{ draw_pos.x + node_radius / 2, draw_pos.y - node_radius * 2 } ;

                                for( int j = 0; j < layer_ns[ i ] / 2; ++j, circ_center.y -= node_radius * 4 )
                                {
                                        DrawCircle( circ_center.x, circ_center.y, node_radius, color_one ) ;
                                }
                        }
                        {
                                Vector2 circ_center{ draw_pos.x + node_radius / 2, draw_pos.y + node_radius * 2 } ;

                                for( int j = 0; j < layer_ns[ i ] / 2; ++j, circ_center.y += node_radius * 4 )
                                {
                                        DrawCircle( circ_center.x, circ_center.y, node_radius, color_one ) ;
                                }
                        }
                }
                else
                {
                        {
                                Vector2 circ_center{ draw_pos.x + node_radius / 2, draw_pos.y } ;

                                DrawCircle( circ_center.x, circ_center.y, node_radius, color_one ) ;
                        }
                        {

                        }
                        {

                        }
                }
        }
}
