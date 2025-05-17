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
constexpr Float mse ( matrix< Size, 1, Float > const & output, matrix< Size, 1, Float > const & expected ) noexcept
{
        auto diff = expected - output ;
        diff *= diff ;

        return diff.accumulate() / Float{ Size } ;
}

template< uti::ssize_t Size, uti::meta::floating_point Float >
constexpr matrix< Size, 1, Float > mse_dx ( matrix< Size, 1, Float > const & output, matrix< Size, 1, Float > const & expected ) noexcept
{
        return ( output - expected ).apply( []( auto val ){ return val *    2 ; } )
                                    .apply( []( auto val ){ return val / Size ; } ) ;
}

////////////////////////////////////////////////////////////////////////////////


} // namespace nnxx
