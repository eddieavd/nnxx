//
//
//      nnxx
//      math/loss.hxx
//

#pragma once

#include <nnxx/common/types.hxx>
#include <nnxx/math/matrix.hxx>


namespace nnxx
{


////////////////////////////////////////////////////////////////////////////////

template< uti::ssize_t Size, uti::meta::floating_point Float >
constexpr Float mse ( matrix< Size, 1, Float > const & _output_, matrix< Size, 1, Float > const & _expected_ ) noexcept
{
        auto diff = _expected_ - _output_ ;
        diff *= diff ;

        return diff.accumulate() / Float{ Size } ;
}

template< uti::ssize_t Size, uti::meta::floating_point Float >
constexpr matrix< Size, 1, Float > mse_dx ( matrix< Size, 1, Float > const & _output_, matrix< Size, 1, Float > const & _expected_ ) noexcept
{
        return ( _output_ - _expected_ ).apply( []( auto val ){ return val *    2 ; } )
                                        .apply( []( auto val ){ return val / Size ; } ) ;
}

////////////////////////////////////////////////////////////////////////////////


} // namespace nnxx
