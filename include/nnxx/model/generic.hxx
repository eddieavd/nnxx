//
//
//      nnxx
//      model/generic.hxx
//

#pragma once

#include <nnxx/common/types.hxx>

#include <nnxx/math/matrix.hxx>
#include <nnxx/math/activation.hxx>
#include <nnxx/math/loss.hxx>

#include <nnxx/layer/meta.hxx>
#include <nnxx/layer/base.hxx>


namespace nnxx
{


////////////////////////////////////////////////////////////////////////////////

template< uti::meta::floating_point Float, typename... Layers >
class generic_model
{
public:
        using  float_type = Float                   ;
        using layer_tuple = uti::tuple< Layers... > ;

        static constexpr ssize_t    depth { sizeof...( Layers ) } ;
        static constexpr ssize_t  input_n { uti::meta::list::front_t< layer_tuple >::prev_size } ;
        static constexpr ssize_t output_n { uti::meta::list:: back_t< layer_tuple >::     size } ;

        constexpr auto forward  ( matrix<  input_n, 1, float_type > const &    _input_                    ) const noexcept ;
        constexpr void activate ( matrix<  input_n, 1, float_type > const &    _input_                    )       noexcept ;
        constexpr void backprop ( matrix< output_n, 1, float_type > const & _expected_, float_type _rate_ )       noexcept ;

        template< ssize_t TestCount >
        constexpr float_type cost ( matrix< TestCount, input_n + output_n, float_type > const & _testset_ ) const noexcept ;

        template< ssize_t TestCount >
        constexpr void train ( matrix< TestCount, input_n + output_n, float_type > const & _testset_, ssize_t _iterations_, float_type _rate_ ) noexcept ;

        UTI_NODISCARD constexpr layer_tuple       & layers ()       noexcept { return layers_ ; }
        UTI_NODISCARD constexpr layer_tuple const & layers () const noexcept { return layers_ ; }
private:
        layer_tuple layers_ ;
} ;

////////////////////////////////////////////////////////////////////////////////

template< uti::meta::floating_point Float, typename... Layers >
struct generic_model< Float, uti::tuple< Layers... > > : generic_model< Float, Layers... > {} ;

////////////////////////////////////////////////////////////////////////////////

template< uti::meta::floating_point Float, typename... Layers >
using activation_tuple_type = uti::tuple< matrix< Layers::size, 1, Float >... > ;

////////////////////////////////////////////////////////////////////////////////

template< uti::meta::floating_point Float, typename... Layers >
constexpr auto
generic_model< Float, Layers... >::forward ( matrix< input_n, 1, float_type > const & _input_ ) const noexcept
{
        activation_tuple_type< Float, Layers... > activations ;

        [ & ]< ssize_t... Idxs >( uti::index_sequence< Idxs... > )
        {
                ( ... ,
                [ & ]
                {
                        if constexpr( Idxs == 0 )
                        {
                                uti::get< Idxs >( activations ) =
                                        uti::get< Idxs >( layers_ )
                                                .forward( _input_ ) ;
                        }
                        else
                        {
                                uti::get< Idxs >( activations ) =
                                        uti::get< Idxs >( layers_ )
                                                .forward( uti::get< Idxs - 1 >( activations ) ) ;
                        }
                }() ) ;
        }( uti::make_index_sequence< sizeof...( Layers ) >{} ) ;

        return uti::get< depth - 1 >( activations ) ;
}

////////////////////////////////////////////////////////////////////////////////

template< uti::meta::floating_point Float, typename... Layers >
constexpr void
generic_model< Float, Layers... >::activate ( matrix< input_n, 1, float_type > const & _input_ ) noexcept
{
        [ & ]< ssize_t... Idxs >( uti::index_sequence< Idxs... > )
        {
                ( ... ,
                [ & ]
                {
                        if constexpr( Idxs == 0 )
                        {
                                uti::get< Idxs >( layers_ )
                                        .activate( _input_ ) ;
                        }
                        else
                        {
                                uti::get< Idxs >( layers_ )
                                        .activate( uti::get< Idxs - 1 >( layers_ ).last_output ) ;
                        }
                }() ) ;
        }( uti::make_index_sequence< sizeof...( Layers ) >{} ) ;
}

////////////////////////////////////////////////////////////////////////////////

template< uti::meta::floating_point Float, typename... Layers >
constexpr void
generic_model< Float, Layers... >::backprop ( matrix< output_n, 1, float_type > const & _expected_, float_type _rate_ ) noexcept
{
        [ & ]< ssize_t... Idxs >( uti::index_sequence< Idxs... > )
        {
                ( ... ,
                [ & ]
                {
                        if constexpr( Idxs == 0 )
                        {
                                uti::get< depth - Idxs - 1 >( layers_ )
                                        .backprop( nnxx::mse_dx( uti::get< depth - Idxs - 1 >( layers_ ).last_output, _expected_ ), _rate_ ) ;
                        }
                        else
                        {
                                uti::get< depth - Idxs - 1 >( layers_ )
                                        .backprop( uti::get< depth - Idxs >( layers_ ).last_gradient, _rate_ ) ;
                        }
                }() ) ;
        }( uti::make_index_sequence< sizeof...( Layers ) >{} ) ;
}

////////////////////////////////////////////////////////////////////////////////

template< uti::meta::floating_point Float, typename... Layers >
template< ssize_t TestCount >
constexpr Float
generic_model< Float, Layers... >::cost ( matrix< TestCount, input_n + output_n, float_type > const & _testset_ ) const noexcept
{
        float_type costval {} ;

        auto testset_ins  = _testset_.template submatrix< 0,       0, TestCount - 1, input_n            - 1 >() ;
        auto testset_outs = _testset_.template submatrix< 0, input_n, TestCount - 1, input_n + output_n - 1 >() ;

        for( ssize_t i = 0; i < TestCount; ++i )
        {
                auto    input = testset_ins .row( i ) ;
                auto expected = testset_outs.row( i ) ;

                auto output = forward( input.transposed() ) ;

                costval += mse( output, expected.transposed() ) ;
        }
        return costval / TestCount ;
}

////////////////////////////////////////////////////////////////////////////////

template< uti::meta::floating_point Float, typename... Layers >
template< ssize_t TestCount >
constexpr void
generic_model< Float, Layers... >::train ( matrix< TestCount, input_n + output_n, float_type > const & _testset_, ssize_t _iterations_, float_type _rate_ ) noexcept
{
        auto testset_ins  = _testset_.template submatrix< 0,       0, TestCount - 1, input_n            - 1 >() ;
        auto testset_outs = _testset_.template submatrix< 0, input_n, TestCount - 1, input_n + output_n - 1 >() ;

        uti::array< matrix<  input_n, 1, float_type >, TestCount >  inputset ;
        uti::array< matrix< output_n, 1, float_type >, TestCount > expectset ;

        for( ssize_t i = 0; i < TestCount; ++i )
        {
                inputset .at( i ) = testset_ins .row( i ).transposed() ;
                expectset.at( i ) = testset_outs.row( i ).transposed() ;
        }
        while( _iterations_-- )
        {
                for( ssize_t i = 0; i < TestCount; ++i )
                {
                        auto const &    input = inputset.at( i ) ;
                        auto const & expected = expectset.at( i ) ;

                        activate(    input         ) ;
                        backprop( expected, _rate_ ) ;
                }
        }
}

////////////////////////////////////////////////////////////////////////////////


} // namespace nnxx
