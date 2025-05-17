//
//
//      nnxx
//      layer/meta.hxx
//

#pragma once

#include <nnxx/common/types.hxx>


namespace nnxx::meta
{


////////////////////////////////////////////////////////////////////////////////

template< typename LayerType >
concept layer_like = requires( LayerType layer )
{
        typename LayerType:: input_matrix_type ;
        typename LayerType::output_matrix_type ;

        { layer.forward ( typename LayerType:: input_matrix_type{} ) } -> uti::meta::convertible_to< typename LayerType::output_matrix_type > ;
        { layer.activate( typename LayerType:: input_matrix_type{} ) } ;
        { layer.backprop( typename LayerType::output_matrix_type{} ) } ;
} ;

////////////////////////////////////////////////////////////////////////////////


} // namespace nnxx::meta
