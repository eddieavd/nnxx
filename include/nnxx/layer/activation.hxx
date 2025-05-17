//
//
//      nnxx
//      layer/activation.hxx
//

#pragma once

#include <nnxx/common/types.hxx>

#include <nnxx/math/matrix.hxx>
#include <nnxx/math/activation.hxx>

#include <nnxx/layer/meta.hxx>
#include <nnxx/layer/base.hxx>


namespace nnxx
{


////////////////////////////////////////////////////////////////////////////////

template< auto ActFn, auto ActDxFn, ssize_t Size, uti::meta::floating_point Float >
struct activation_layer
        : public layer_base< activation_layer< ActFn, ActDxFn, Size, Float >, Size, Size, Float >
{
        using _self = activation_layer                       ;
        using _base = layer_base< _self, Size, Size, Float > ;

        using  input_matrix_type = typename _base:: input_matrix_type ;
        using output_matrix_type = typename _base::output_matrix_type ;

        constexpr activation_layer () noexcept = default ;

        constexpr output_matrix_type forward ( input_matrix_type const & _input_ ) const noexcept
        {
                auto result( _input_ ) ;
                result.apply( []( auto val ){ return ActFn( val ) ; } ) ;
                return result ;
        }
        constexpr void activate ( input_matrix_type const & _input_ ) noexcept
        {
                _base::last_input  = _input_ ;
                _base::last_output = _input_ ;

                _base::last_output.apply( []( auto val ){ return ActFn( val ) ; } ) ;
        }
        constexpr void backprop ( output_matrix_type const & _gradient_, Float ) noexcept
        {
//              auto input( _base::last_input ) ;
//              input.apply( []( auto val ){ return ActDxFn( val ) ; } ) ;
//              _base::last_gradient = _gradient_ * input ;

                _base::last_gradient = _gradient_ * _base::last_input.apply( []( auto val ){ return ActDxFn( val ) ; } ) ; // og impl
        }
} ;

////////////////////////////////////////////////////////////////////////////////

template< ssize_t Size, uti::meta::floating_point Float >
struct relu_layer : activation_layer< relu< Float >, relu_dx< Float >, Size, Float > {} ;

template< ssize_t Size, uti::meta::floating_point Float >
struct leaky_relu_layer : activation_layer< leaky_relu< Float >, leaky_relu_dx< Float >, Size, Float > {} ;

template< ssize_t Size, uti::meta::floating_point Float >
struct tanh_layer : activation_layer< tanh< Float >, tanh_dx< Float >, Size, Float > {} ;

template< ssize_t Size, uti::meta::floating_point Float >
struct sigmoid_layer : activation_layer< sigmoid< Float >, sigmoid_dx< Float >, Size, Float > {} ;

template< ssize_t Size, uti::meta::floating_point Float >
struct hard_sigmoid_layer : activation_layer< hard_sigmoid< Float >, hard_sigmoid_dx< Float >, Size, Float > {} ;

template< ssize_t Size, uti::meta::floating_point Float >
struct hard_silu_layer : activation_layer< hard_silu< Float >, hard_silu_dx< Float >, Size, Float > {} ;

////////////////////////////////////////////////////////////////////////////////


} // namespace nnxx
