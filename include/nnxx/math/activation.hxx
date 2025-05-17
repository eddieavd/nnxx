//
//
//      nnxx
//      math/activation.hxx
//

#pragma once

#include <nnxx/common/types.hxx>


namespace nnxx
{


////////////////////////////////////////////////////////////////////////////////
/// identity

template< uti::meta::floating_point Float >
constexpr Float identity ( Float val ) noexcept
{
        return val ;
}

template< uti::meta::floating_point Float >
constexpr Float identity_dx ( Float ) noexcept
{
        return Float{ 1 } ;
}

////////////////////////////////////////////////////////////////////////////////
/// relu

template< uti::meta::floating_point Float >
constexpr Float relu ( Float val ) noexcept
{
        return val > 0 ? val : 0 ;
}

template< uti::meta::floating_point Float >
constexpr Float relu_dx ( Float val ) noexcept
{
        return val >= 0 ? 1 : 0 ;
}

////////////////////////////////////////////////////////////////////////////////
/// leaky relu

template< uti::meta::floating_point Float >
constexpr Float leaky_relu ( Float val ) noexcept
{
        return val > 0 ? val : val * Float{ 0.01 } ;
}

template< uti::meta::floating_point Float >
constexpr Float leaky_relu_dx ( Float val ) noexcept
{
        return val >= 0 ? 1 : Float{ 0.01 } ;
}

////////////////////////////////////////////////////////////////////////////////
/// tanh

template< uti::meta::floating_point Float >
constexpr Float tanh ( Float val ) noexcept
{
        Float  ex = __builtin_expf( val ) ;
        Float enx = __builtin_expf( -val ) ;

        return ( ex - enx ) / ( ex + enx ) ;
}

template< uti::meta::floating_point Float >
constexpr Float tanh_dx ( Float val ) noexcept
{
        return 1 - val * val ;
}

////////////////////////////////////////////////////////////////////////////////
/// sigmoid

template< uti::meta::floating_point Float >
constexpr Float sigmoid ( Float val ) noexcept
{
        if constexpr( uti::is_same_v< Float, float > )
        {
                return Float{ 1 } / ( Float{ 1 } + __builtin_expf( -val ) ) ;
        }
        else if constexpr( uti::is_same_v< Float, double > )
        {
                return Float{ 1 } / ( Float{ 1 } + __builtin_exp( -val ) ) ;
        }
        else if constexpr( uti::is_same_v< Float, long double > )
        {
                return Float{ 1 } / ( Float{ 1 } + __builtin_expl( -val ) ) ;
        }
        else
        {
                return Float{ 1 } / ( Float{ 1 } + __builtin_expl( -static_cast< long double >( val ) ) ) ;
        }
}

template< uti::meta::floating_point Float >
constexpr Float sigmoid_dx ( Float val ) noexcept
{
        return val * ( 1 - val ) ;
}

////////////////////////////////////////////////////////////////////////////////
/// hard sigmoid

template< uti::meta::floating_point Float >
constexpr Float hard_sigmoid ( Float val ) noexcept
{
        return val <= -3 ? 0
                         : val >= 3 ? 1
                                    : ( val / 6 ) + Float{ 0.5 } ;
}

template< uti::meta::floating_point Float >
constexpr Float hard_sigmoid_dx ( Float val ) noexcept
{
        return val <= -3 ? 0
                         : val >= 3 ? 1
                                    : ( Float{ 1 } / 6 ) ;
}

////////////////////////////////////////////////////////////////////////////////
/// hard silu

template< uti::meta::floating_point Float >
constexpr Float hard_silu ( Float val ) noexcept
{
        return val < -3 ? 0
                        : val > 3 ? val
                                  : val * ( val + 3 ) / 6 ;
}

template< uti::meta::floating_point Float >
constexpr Float hard_silu_dx ( Float val ) noexcept
{
        return val < -3 ? 0
                        : val > 3 ? 1
                                  : ( 2 * val + 3 ) / 6 ;
}


} // namespace nnxx
