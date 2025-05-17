//
//
//      nnxx
//      layer/base.hxx
//

#pragma once

#include <nnxx/common/types.hxx>
#include <nnxx/math/matrix.hxx>
#include <nnxx/layer/meta.hxx>


namespace nnxx
{


////////////////////////////////////////////////////////////////////////////////

template< typename LayerImpl, ssize_t PrevSize, ssize_t Size, uti::meta::floating_point Float >
class layer_base
{
        using _impl = LayerImpl ;
public:
        using  input_matrix_type = matrix< PrevSize, 1, Float > ;
        using output_matrix_type = matrix<     Size, 1, Float > ;

        static constexpr ssize_t prev_size { PrevSize } ;
        static constexpr ssize_t      size {     Size } ;

        constexpr output_matrix_type forward ( input_matrix_type const & _input_ ) const noexcept
        {
                return static_cast< _impl const * >( this )->_forward( _input_ ) ;
        }
        constexpr void activate ( input_matrix_type const & _input_ ) noexcept
        {
                static_cast< _impl * >( this )->_activate( _input_ ) ;
        }
        constexpr void backprop ( output_matrix_type const & _gradient_, Float _rate_ ) noexcept
        {
                static_cast< _impl * >( this )->_backprop( _gradient_, _rate_ ) ;
        }

         input_matrix_type last_input    ;
         input_matrix_type last_gradient ;
        output_matrix_type last_output   ;
} ;

////////////////////////////////////////////////////////////////////////////////


} // namespace nnxx
