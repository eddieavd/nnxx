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


namespace nnxx
{


////////////////////////////////////////////////////////////////////////////////

template< ssize_t PrevSize, ssize_t Size, uti::meta::floating_point Float >
struct dense_layer
        : public layer_base< dense_layer< PrevSize, Size, Float >, PrevSize, Size, Float >
{
        using _self = dense_layer                                ;
        using _base = layer_base< _self, PrevSize, Size, Float > ;

        using  input_matrix_type = typename _base:: input_matrix_type ;
        using output_matrix_type = typename _base::output_matrix_type ;

        constexpr dense_layer () noexcept { randomize() ; }

        constexpr output_matrix_type forward ( input_matrix_type const & _input_ ) const noexcept
        {
                return weights * _input_ + biases ;
        }
        constexpr void activate ( input_matrix_type const & _input_ ) noexcept
        {
                _base::last_input = _input_ ;
                _base::last_output = weights * _base::last_input + biases ;
        }
        constexpr void backprop ( output_matrix_type const & _gradient_, Float _rate_ ) noexcept
        {
                auto weights_gradient = _gradient_ * _base::last_input.transposed() ;

                weights -= weights_gradient  * _rate_ ;
                biases  -=        _gradient_ * _rate_ ;

                _base::last_gradient = weights.transposed() * _gradient_ ;
        }

        constexpr void randomize () noexcept
        {
                static auto offset { 0 } ;

                weights.fill( [ & ]{ return uti::meta::basic_rand_float( __FILE__, __LINE__ + ( offset++ ) ) * 10.0 ; } ) ;
                biases .fill( [ & ]{ return uti::meta::basic_rand_float( __FILE__, __LINE__ + ( offset++ ) ) * 10.0 ; } ) ;
        }

        matrix< Size, PrevSize, Float > weights ;
        matrix< Size,        1, Float >  biases ;
} ;

////////////////////////////////////////////////////////////////////////////////


} // namespace nnxx
