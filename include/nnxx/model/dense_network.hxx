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

////////////////////////////////////////////////////////////////////////////////

template< uti::meta::floating_point Float, template< uti::meta::floating_point > typename ActivationTraits, ssize_t... LayerNs >
struct create_dense_network ;

template< uti::meta::floating_point Float, template< uti::meta::floating_point > typename ActivationTraits >
struct create_dense_network< Float, ActivationTraits > ;

template< uti::meta::floating_point Float, template< uti::meta::floating_point > typename ActivationTraits, ssize_t S >
struct create_dense_network< Float, ActivationTraits, S > ;

template< uti::meta::floating_point Float, template< uti::meta::floating_point > typename ActivationTraits, ssize_t S1, ssize_t S2 >
struct create_dense_network< Float, ActivationTraits, S1, S2 >
        : uti::type_identity< uti::tuple< activated_dense_layer< ActivationTraits< Float >, S1, S2, Float > > > {} ;

template< uti::meta::floating_point Float, template< uti::meta::floating_point > typename ActivationTraits, ssize_t S1, ssize_t S2, ssize_t... Ss >
struct create_dense_network< Float, ActivationTraits, S1, S2, Ss... >
{
        using type = typename uti::meta::join< uti::tuple, uti::meta::tuplify >::template
                                   fn< typename create_dense_network< Float, ActivationTraits, S1, S2    >::type ,
                                       typename create_dense_network< Float, ActivationTraits, S2, Ss... >::type > ;
} ;

template< uti::meta::floating_point Float, template< uti::meta::floating_point > typename ActivationTraits, ssize_t... LayerNs >
using dense_network_layers = typename create_dense_network< Float, ActivationTraits, LayerNs... >::type ;

////////////////////////////////////////////////////////////////////////////////

} // namespace meta

////////////////////////////////////////////////////////////////////////////////

template< uti::meta::floating_point Float, template< uti::meta::floating_point > typename ActivationTraits, ssize_t... LayerNs >
using dense_neural_network
        = generic_model< Float, meta::dense_network_layers< Float, ActivationTraits, LayerNs... > > ;

////////////////////////////////////////////////////////////////////////////////


} // namespace nnxx
