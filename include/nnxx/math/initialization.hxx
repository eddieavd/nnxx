//
//
//      nnxx
//      math/initialization.hxx
//

#pragma once

#include <nnxx/common/types.hxx>


namespace nnxx
{


////////////////////////////////////////////////////////////////////////////////

template< typename Float, bool IsWeights >
constexpr Float he_initialization ( ssize_t _off_, ssize_t _n_in_, ssize_t _n_out_ ) noexcept ;

template< typename Float, bool IsWeights >
constexpr Float xavier_initialization ( ssize_t _off_, ssize_t _n_in_, ssize_t _n_out_ ) noexcept ;

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template< typename Float, bool IsWeights >
constexpr Float he_initialization ( ssize_t _off_, ssize_t _n_in_, ssize_t ) noexcept
{
        if constexpr( IsWeights )
        {
//              return std::sqrt( uti::meta::basic_rand_float( __FILE__, __LINE__ + ( _off_ ) ) * ( 2.0 / _n_in_ ) ) ;
                return            uti::meta::basic_rand_float( __FILE__, __LINE__ + ( _off_ ) ) * ( 2.0 / _n_in_ )   ;
        }
        else
        {
                return 0 ;
        }
}

////////////////////////////////////////////////////////////////////////////////

template< typename Float, bool IsWeights >
constexpr Float xavier_initialization ( ssize_t _off_, ssize_t _n_in_, ssize_t _n_out_ ) noexcept
{
        if constexpr( IsWeights )
        {
//              return std::sqrt( uti::meta::basic_rand_float( __FILE__, __LINE__ + ( _off_ ) ) * ( 2.0 / ( _n_in_ + _n_out_ ) ) ) ;
                return            uti::meta::basic_rand_float( __FILE__, __LINE__ + ( _off_ ) ) * ( 2.0 / ( _n_in_ + _n_out_ ) )   ;
        }
        else
        {
                return 0 ;
        }
}

////////////////////////////////////////////////////////////////////////////////


} // namespace nnxx
