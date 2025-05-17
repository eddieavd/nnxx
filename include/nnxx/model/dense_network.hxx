//
//
//      nnxx
//      model/dense_network.hxx
//

#pragma once

#include <nnxx/common/types.hxx>

#include <nnxx/math/matrix.hxx>
#include <nnxx/math/activation.hxx>
#include <nnxx/math/loss.hxx>

#include <nnxx/layer/meta.hxx>
#include <nnxx/layer/base.hxx>
#include <nnxx/layer/dense.hxx>

#include <nnxx/model/generic.hxx>


namespace nnxx
{


////////////////////////////////////////////////////////////////////////////////

namespace meta
{


template< uti::meta::floating_point Float, template< ssize_t, uti::meta::floating_point > typename ActLayer, ssize_t... LayerNs >
struct create_dense_network ;

template< uti::meta::floating_point Float, template< ssize_t, uti::meta::floating_point > typename ActLayer >
struct create_dense_network< Float, ActLayer > ;

template< uti::meta::floating_point Float, template< ssize_t, uti::meta::floating_point > typename ActLayer, ssize_t S >
struct create_dense_network< Float, ActLayer, S > ;

template< uti::meta::floating_point Float, template< ssize_t, uti::meta::floating_point > typename ActLayer, ssize_t S1, ssize_t S2 >
struct create_dense_network< Float, ActLayer, S1, S2 >
        : uti::type_identity< uti::tuple< dense_layer< S1, S2, Float >, ActLayer< S2, Float > > > {} ;

template< uti::meta::floating_point Float, template< ssize_t, uti::meta::floating_point > typename ActLayer, ssize_t S1, ssize_t S2, ssize_t... Ss >
struct create_dense_network< Float, ActLayer, S1, S2, Ss... >
{
        using type = typename uti::meta::join< uti::tuple, uti::meta::tuplify >::template
                                   fn< typename create_dense_network< Float, ActLayer, S1, S2    >::type ,
                                       typename create_dense_network< Float, ActLayer, S2, Ss... >::type > ;
} ;

template< uti::meta::floating_point Float, template< ssize_t, uti::meta::floating_point > typename ActLayer, ssize_t... LayerNs >
using dense_network_layers = typename create_dense_network< Float, ActLayer, LayerNs... >::type ;


} // namespace meta

////////////////////////////////////////////////////////////////////////////////

template< uti::meta::floating_point Float, template< ssize_t, uti::meta::floating_point > typename ActivationLayer, ssize_t... LayerNs >
using dense_neural_network
        = generic_model< Float, meta::dense_network_layers< Float, ActivationLayer, LayerNs... > > ;

////////////////////////////////////////////////////////////////////////////////


} // namespace nnxx
