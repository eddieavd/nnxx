//
//
//      nnxx
//      layer/dense.hxx
//

#pragma once

#include <nnxx/common/types.hxx>

#include <nnxx/math/matrix.hxx>

#include <nnxx/layer/meta.hxx>
#include <nnxx/layer/base.hxx>
#include <nnxx/layer/activation.hxx>


namespace nnxx
{


////////////////////////////////////////////////////////////////////////////////

template< typename ActivationTraits, ssize_t PrevSize, ssize_t Size, uti::meta::floating_point Float >
class activated_dense_layer
        : public layer_base< activated_dense_layer< ActivationTraits, PrevSize, Size, Float >, PrevSize, Size, Float >
{
        using _self = activated_dense_layer                      ;
        using _base = layer_base< _self, PrevSize, Size, Float > ;

        friend _base ;
public:
        using  input_matrix_type = typename _base:: input_matrix_type ;
        using output_matrix_type = typename _base::output_matrix_type ;

        using activation_traits = ActivationTraits ;

        static constexpr auto    activation_fn    { activation_traits::activation_fn    } ;
        static constexpr auto    activation_dx_fn { activation_traits::activation_dx_fn } ;
        static constexpr auto w_initialization_fn { activation_traits::       w_init_fn } ;
        static constexpr auto b_initialization_fn { activation_traits::       b_init_fn } ;

        constexpr activated_dense_layer () noexcept { randomize() ; }
private:
        constexpr void randomize () noexcept
        {
                if( uti::is_constant_evaluated() )
                {
                        weights.fill( w_initialization_fn, 0, PrevSize, Size ) ;
                        biases .fill( b_initialization_fn, 0, PrevSize, Size ) ;
                }
                else
                {
                        static auto offset { 0 } ;
                        weights.fill( w_initialization_fn, offset++, PrevSize, Size ) ;
                        biases .fill( b_initialization_fn, offset++, PrevSize, Size ) ;
                }
        }

        constexpr output_matrix_type _forward ( input_matrix_type const & _input_ ) const noexcept
        {
                return ( weights * _input_ + biases ).apply( activation_fn ) ;
        }
        constexpr void _activate ( input_matrix_type const & _input_ ) noexcept
        {
                _base::last_input  = _input_ ;

                raw_out = weights * _base::last_input + biases ;

                _base::last_output = raw_out ;
                _base::last_output.apply( activation_fn ) ;
        }
        constexpr void _backprop ( output_matrix_type const & _gradient_, Float _rate_ ) noexcept
        {
                auto input( raw_out ) ;
                input.apply( activation_dx_fn ) ;

                auto         gradient = _gradient_ * input ;
                auto weights_gradient =  gradient  * _base::last_input.transposed() ;

                weights -= weights_gradient * _rate_ ;
                biases  -=         gradient * _rate_ ;

                _base::last_gradient = weights.transposed() * gradient ;
        }

        matrix< Size, PrevSize, Float > weights ;
        matrix< Size,        1, Float >  biases ;

        output_matrix_type raw_out ;
} ;

////////////////////////////////////////////////////////////////////////////////

template< ssize_t PrevSize, ssize_t Size, uti::meta::floating_point Float >
using dense_layer = activated_dense_layer< identity_activation_traits< Float >, PrevSize, Size, Float > ;

template< ssize_t PrevSize, ssize_t Size, uti::meta::floating_point Float >
using relu_dense_layer = activated_dense_layer< relu_activation_traits< Float >, PrevSize, Size, Float > ;

template< ssize_t PrevSize, ssize_t Size, uti::meta::floating_point Float >
using sigmoid_dense_layer = activated_dense_layer< sigmoid_activation_traits< Float >, PrevSize, Size, Float > ;

////////////////////////////////////////////////////////////////////////////////


} // namespace nnxx
