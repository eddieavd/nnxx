//
//
//      nnxx
//      examples/demo.cxx
//

#include <cstdio>

#include <nnxx/layer/activation.hxx>
#include <nnxx/model/dense_network.hxx>


[[ maybe_unused ]] constexpr
nnxx::matrix< 4, 3, float > test_and { 0, 0, 0 ,
                                       0, 1, 0 ,
                                       1, 0, 0 ,
                                       1, 1, 1 } ;
[[ maybe_unused ]] constexpr
nnxx::matrix< 4, 3, float > test_xor { 0, 0, 0 ,
                                       0, 1, 1 ,
                                       1, 0, 1 ,
                                       1, 1, 0 } ;

int main ()
{
        using model_t = nnxx::dense_neural_network< float, nnxx::relu_layer, 2, 2, 1 > ;

        uti::array< model_t, 5 > models {} ;

        for( int i = 0; i < 10'000; ++i )
        {
                if( i % 100 == 0 )
                {
                        printf( "nnxx : model 1 cost : %f\n", models[ 0 ].cost( test_xor ) ) ;
                        printf( "nnxx : model 2 cost : %f\n", models[ 1 ].cost( test_xor ) ) ;
                        printf( "nnxx : model 3 cost : %f\n", models[ 2 ].cost( test_xor ) ) ;
                        printf( "nnxx : model 4 cost : %f\n", models[ 3 ].cost( test_xor ) ) ;
                        printf( "nnxx : model 5 cost : %f\n", models[ 4 ].cost( test_xor ) ) ;
                }
                models[ 0 ].train( test_xor, 1, 1e-3 ) ;
                models[ 1 ].train( test_xor, 1, 1e-3 ) ;
                models[ 2 ].train( test_xor, 1, 1e-3 ) ;
                models[ 3 ].train( test_xor, 1, 1e-3 ) ;
                models[ 4 ].train( test_xor, 1, 1e-3 ) ;
        }
        for( int m = 0; m < 5; ++m )
        {
                printf( "==========\n" ) ;
                printf( "model cost : %f\n", models[ m ].cost( test_xor ) ) ;
                for( float i = 0; i < 2; ++i )
                {
                        for( float j = 0; j < 2; ++j )
                        {
                                printf( "%.2f op %.2f eq %.4f\n", i, j, static_cast< float >( models[ m ].forward( { i, j } ) ) ) ;
                        }
                }
                printf( "==========\n" ) ;
        }



        return 0 ;
}
