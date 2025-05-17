//
//
//      nnxx
//      examples/demo.cxx
//

#include <cstdio>

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
        static constexpr int training_iterations { 130 } ;

        using model_t = nnxx::dense_neural_network< float, nnxx::relu_activation_traits, 2, 4, 1 > ;

        static constexpr model_t model = [ & ]
        {
                model_t m ;

                m.train( test_and, training_iterations, 1e-3 ) ;

                return m ;
        }() ;

        printf( "nnxx : trained 'and' model at compile time over %d epochs\n", training_iterations ) ;
        printf( "nnxx : model cost : %.4f\n", model.cost( test_and ) ) ;

        for( float i = 0; i < 2; ++i )
        {
                for( float j = 0; j < 2; ++j )
                {
                        printf( "%.2f op %.2f eq %.4f\n", i, j, static_cast< float >( model.forward( { i, j } ) ) ) ;
                }
        }

        return 0 ;
}
