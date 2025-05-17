//
//
//      nnxx
//      examples/demo.cxx
//

#include <cstdio>

#include <nnxx/layer/activation.hxx>
#include <nnxx/model/dense_network.hxx>


[[ maybe_unused ]] constexpr
nnxx::matrix< 4, 3, float > test_and {  0, 0, 0 ,
                                        0, 1, 0 ,
                                        1, 0, 0 ,
                                        1, 1, 1 } ;
[[ maybe_unused ]] constexpr
nnxx::matrix< 4, 3, float > test_xor {  0, 0, 0 ,
                                        0, 1, 1 ,
                                        1, 0, 1 ,
                                        1, 1, 0 } ;

int main ()
{
        using model_t = nnxx::dense_neural_network< float, nnxx::relu_layer, 2, 4, 1 > ;

        model_t model ;

        for( int i = 0; i < 1000; ++i )
        {
                printf( "nnxx : model cost : %.4f\n", model.cost( test_xor ) ) ;

                model.train( test_xor, 100, 1e-4 ) ;
        }

        for( float i = 0; i < 2; ++i )
        {
                for( float j = 0; j < 2; ++j )
                {
                        printf( "%.2f op %.2f eq %.4f\n", i, j, static_cast< float >( model.forward( { i, j } ) ) ) ;
                }
        }



        return 0 ;
}
