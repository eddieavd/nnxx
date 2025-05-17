//
//
//      nnxx
//      math/matrix.hxx
//

#pragma once

#include <nnxx/common/types.hxx>


namespace nnxx
{


////////////////////////////////////////////////////////////////////////////////

template< ssize_t Rows, ssize_t Cols, uti::meta::arithmetic T = float >
        requires( Rows >= 1 && Cols >= 1 )
struct matrix
{
        using value_type =       T ;
        using ssize_type = ssize_t ;

        using         pointer = value_type       * ;
        using   const_pointer = value_type const * ;
        using       reference = value_type       & ;
        using const_reference = value_type const & ;

        static constexpr ssize_type rows { Rows } ;
        static constexpr ssize_type cols { Cols } ;

                      constexpr matrix & operator+= ( matrix const & _other_ )       noexcept ;
        UTI_NODISCARD constexpr matrix   operator+  ( matrix const & _other_ ) const noexcept ;

                      constexpr matrix & operator-= ( matrix const & _other_ )       noexcept ;
        UTI_NODISCARD constexpr matrix   operator-  ( matrix const & _other_ ) const noexcept ;

                      constexpr matrix & operator*= ( value_type _scalar_ )       noexcept ;
        UTI_NODISCARD constexpr matrix   operator*  ( value_type _scalar_ ) const noexcept ;

                      constexpr matrix & operator/= ( value_type _scalar_ )       noexcept ;
        UTI_NODISCARD constexpr matrix   operator/  ( value_type _scalar_ ) const noexcept ;

                      constexpr matrix & operator*= ( matrix const & _other_ )       noexcept ;
        UTI_NODISCARD constexpr matrix   operator*  ( matrix const & _other_ ) const noexcept ;

        template< ssize_type ColsOther >
        UTI_NODISCARD constexpr matrix< Rows, ColsOther, T > operator* ( matrix< Cols, ColsOther, T > const & _other_ ) const noexcept ;

        UTI_NODISCARD constexpr operator value_type () const noexcept requires( rows == 1 && cols == 1 ){ return data[ 0 ] ; }

        UTI_NODISCARD constexpr value_type accumulate () const noexcept ;

        constexpr matrix & fill ( value_type _val_ ) noexcept ;

        template< typename Filler, typename... Args >
                requires( uti::meta::convertible_to< decltype( uti::declval< Filler & >()( uti::declval< Args&& >()... ) ), value_type > )
        constexpr matrix & fill ( Filler&& _fill_, Args&&... _args_ ) noexcept ;

        template< typename Transform >
                requires( uti::meta::convertible_to< decltype( uti::declval< Transform & >()( value_type{} ) ), value_type > )
        constexpr matrix & apply ( Transform&& _transform_ ) noexcept ;

        UTI_NODISCARD constexpr matrix< Cols, Rows, T > transposed () const noexcept ;

        UTI_NODISCARD constexpr matrix< 1, Cols, T > row ( ssize_type _row_idx_ ) const noexcept ;
        UTI_NODISCARD constexpr matrix< Rows, 1, T > col ( ssize_type _col_idx_ ) const noexcept ;

        template< ssize_type X1, ssize_type Y1, ssize_type X2, ssize_type Y2 >
        UTI_NODISCARD constexpr matrix< X2 - X1 + 1, Y2 - Y1 + 1, T > submatrix () const noexcept ;

        template< typename Self >
        UTI_NODISCARD constexpr decltype( auto ) at ( this Self && self, ssize_type _row_, ssize_type _col_ ) noexcept
        {
                UTI_CEXPR_ASSERT( 0 <= _row_ && _row_ < rows &&
                                  0 <= _col_ && _col_ < cols  ,
                                 "nnxx::matrix::at: index out of bounds"
                ) ;
                return UTI_FWD( self ).data[ _row_ * Cols + _col_ ] ;
        }

        value_type data[ rows * cols ] { 0 } ;
} ;

////////////////////////////////////////////////////////////////////////////////

namespace meta
{


template< typename Matrix >
concept matrix_like = requires( Matrix & matrix, Matrix const & cmatrix )
{
        typename Matrix::value_type ;

        { Matrix::rows } -> uti::meta::convertible_to< ssize_t > ;
        { Matrix::cols } -> uti::meta::convertible_to< ssize_t > ;

        {  matrix.at( ssize_t{}, ssize_t{} ) } -> uti::meta::same_as< typename Matrix::value_type       & > ;
        { cmatrix.at( ssize_t{}, ssize_t{} ) } -> uti::meta::same_as< typename Matrix::value_type const & > ;
} ;


} // namespace meta

////////////////////////////////////////////////////////////////////////////////

template< ssize_t Rows, ssize_t Cols, uti::meta::arithmetic T >
        requires( Rows >= 1 && Cols >= 1 )
constexpr
matrix< Rows, Cols, T > &
matrix< Rows, Cols, T >::operator+= ( matrix const & _other_ ) noexcept
{
        for( ssize_type i = 0; i < rows * cols; ++i )
        {
                data[ i ] += _other_.data[ i ] ;
        }
        return *this ;
}

template< ssize_t Rows, ssize_t Cols, uti::meta::arithmetic T >
        requires( Rows >= 1 && Cols >= 1 )
UTI_NODISCARD constexpr
matrix< Rows, Cols, T >
matrix< Rows, Cols, T >::operator+ ( matrix const & _other_ ) const noexcept
{
        auto sum = *this ;
        sum += _other_ ;
        return sum ;
}

////////////////////////////////////////////////////////////////////////////////

template< ssize_t Rows, ssize_t Cols, uti::meta::arithmetic T >
        requires( Rows >= 1 && Cols >= 1 )
constexpr
matrix< Rows, Cols, T > &
matrix< Rows, Cols, T >::operator-= ( matrix const & _other_ ) noexcept
{
        for( ssize_type i = 0; i < rows * cols; ++i )
        {
                data[ i ] -= _other_.data[ i ] ;
        }
        return *this ;
}

template< ssize_t Rows, ssize_t Cols, uti::meta::arithmetic T >
        requires( Rows >= 1 && Cols >= 1 )
UTI_NODISCARD constexpr
matrix< Rows, Cols, T >
matrix< Rows, Cols, T >::operator- ( matrix const & _other_ ) const noexcept
{
        auto diff = *this ;
        diff -= _other_ ;
        return diff ;
}

////////////////////////////////////////////////////////////////////////////////

template< ssize_t Rows, ssize_t Cols, uti::meta::arithmetic T >
        requires( Rows >= 1 && Cols >= 1 )
constexpr
matrix< Rows, Cols, T > &
matrix< Rows, Cols, T >::operator*= ( matrix const & _other_ ) noexcept
{
        for( ssize_type i = 0; i < rows * cols; ++i )
        {
                data[ i ] *= _other_.data[ i ] ;
        }
        return *this ;
}

template< ssize_t Rows, ssize_t Cols, uti::meta::arithmetic T >
        requires( Rows >= 1 && Cols >= 1 )
UTI_NODISCARD constexpr
matrix< Rows, Cols, T >
matrix< Rows, Cols, T >::operator* ( matrix const & _other_ ) const noexcept
{
        auto product = *this ;
        product *= _other_ ;
        return product ;
}

////////////////////////////////////////////////////////////////////////////////

template< ssize_t Rows, ssize_t Cols, uti::meta::arithmetic T >
        requires( Rows >= 1 && Cols >= 1 )
constexpr
matrix< Rows, Cols, T > &
matrix< Rows, Cols, T >::operator*= ( value_type _scalar_ ) noexcept
{
        for( ssize_type i = 0; i < rows * cols; ++i )
        {
                data[ i ] *= _scalar_ ;
        }
        return *this ;
}

template< ssize_t Rows, ssize_t Cols, uti::meta::arithmetic T >
        requires( Rows >= 1 && Cols >= 1 )
UTI_NODISCARD constexpr
matrix< Rows, Cols, T >
matrix< Rows, Cols, T >::operator* ( value_type _scalar_ ) const noexcept
{
        auto scaled = *this ;
        scaled *= _scalar_ ;
        return scaled ;
}

////////////////////////////////////////////////////////////////////////////////

template< ssize_t Rows, ssize_t Cols, uti::meta::arithmetic T >
        requires( Rows >= 1 && Cols >= 1 )
constexpr
matrix< Rows, Cols, T > &
matrix< Rows, Cols, T >::operator/= ( value_type _scalar_ ) noexcept
{
        for( ssize_type i = 0; i < rows * cols; ++i )
        {
                data[ i ] /= _scalar_ ;
        }
        return *this ;
}

template< ssize_t Rows, ssize_t Cols, uti::meta::arithmetic T >
        requires( Rows >= 1 && Cols >= 1 )
UTI_NODISCARD constexpr
matrix< Rows, Cols, T >
matrix< Rows, Cols, T >::operator/ ( value_type _scalar_ ) const noexcept
{
        auto scaled = *this ;
        scaled /= _scalar_ ;
        return scaled ;
}

////////////////////////////////////////////////////////////////////////////////

template< ssize_t Rows, ssize_t Cols, uti::meta::arithmetic T >
        requires( Rows >= 1 && Cols >= 1 )
template< ssize_t ColsOther >
UTI_NODISCARD constexpr
matrix< Rows, ColsOther, T >
matrix< Rows, Cols, T >::operator* ( matrix< Cols, ColsOther, T > const & _other_ ) const noexcept
{
        matrix< Rows, ColsOther, T > product {} ;

        for( ssize_type i = 0; i < Rows; ++i )
        {
                for( ssize_type j = 0; j < ColsOther; ++j )
                {
                        for( ssize_type k = 0; k < Cols; ++k )
                        {
                                product.at( i, j ) += at( i, k ) * _other_.at( k, j ) ;
                        }
                }
        }
        return product ;
}

////////////////////////////////////////////////////////////////////////////////

template< ssize_t Rows, ssize_t Cols, uti::meta::arithmetic T >
        requires( Rows >= 1 && Cols >= 1 )
UTI_NODISCARD constexpr T
matrix< Rows, Cols, T >::accumulate () const noexcept
{
        value_type sum {} ;

        for( ssize_type i = 0; i < rows * cols; ++i )
        {
                sum += data[ i ] ;
        }
        return sum ;
}

////////////////////////////////////////////////////////////////////////////////

template< ssize_t Rows, ssize_t Cols, uti::meta::arithmetic T >
        requires( Rows >= 1 && Cols >= 1 )
constexpr
matrix< Rows, Cols, T > &
matrix< Rows, Cols, T >::fill ( value_type _val_ ) noexcept
{
        for( ssize_type i = 0; i < rows * cols; ++i )
        {
                data[ i ] = _val_ ;
        }
        return *this ;
}

////////////////////////////////////////////////////////////////////////////////

template< ssize_t Rows, ssize_t Cols, uti::meta::arithmetic T >
        requires( Rows >= 1 && Cols >= 1 )
template< typename Filler, typename... Args >
        requires( uti::meta::convertible_to< decltype( uti::declval< Filler & >()( uti::declval< Args&& >()... ) ), T > )
constexpr
matrix< Rows, Cols, T > &
matrix< Rows, Cols, T >::fill ( Filler&& _fill_, Args&&... _args_ ) noexcept
{
        for( ssize_type i = 0; i < rows * cols; ++i )
        {
                data[ i ] = _fill_( UTI_FWD( _args_ )... ) ;
        }
        return *this ;
}

////////////////////////////////////////////////////////////////////////////////

template< ssize_t Rows, ssize_t Cols, uti::meta::arithmetic T >
        requires( Rows >= 1 && Cols >= 1 )
template< typename Transform >
        requires( uti::meta::convertible_to< decltype( uti::declval< Transform & >()( T{} ) ), T > )
constexpr
matrix< Rows, Cols, T > &
matrix< Rows, Cols, T >::apply ( Transform&& _transform_ ) noexcept
{
        for( ssize_type i = 0; i < rows * cols; ++i )
        {
                data[ i ] = _transform_( data[ i ] ) ;
        }
        return *this ;
}

////////////////////////////////////////////////////////////////////////////////

template< ssize_t Rows, ssize_t Cols, uti::meta::arithmetic T >
        requires( Rows >= 1 && Cols >= 1 )
constexpr
matrix< Cols, Rows, T >
matrix< Rows, Cols, T >::transposed () const noexcept
{
        matrix< Cols, Rows, T > trans ;

        for( ssize_type i = 0; i < rows; ++i )
        {
                for( ssize_type j = 0; j < cols; ++j )
                {
                        trans.at( j, i ) = at( i, j ) ;
                }
        }
        return trans ;
}

////////////////////////////////////////////////////////////////////////////////

template< ssize_t Rows, ssize_t Cols, uti::meta::arithmetic T >
        requires( Rows >= 1 && Cols >= 1 )
constexpr
matrix<    1, Cols, T >
matrix< Rows, Cols, T >::row ( ssize_type _row_idx_ ) const noexcept
{
        UTI_CEXPR_ASSERT( _row_idx_ < Rows, "nnxx::matrix::row: index out of bounds" ) ;

        matrix< 1, cols, T > mat ;

        ::uti::copy( data + _row_idx_ * Cols, data + _row_idx_ * Cols + Cols, mat.data ) ;

        return mat ;
}

////////////////////////////////////////////////////////////////////////////////

template< ssize_t Rows, ssize_t Cols, uti::meta::arithmetic T >
        requires( Rows >= 1 && Cols >= 1 )
constexpr
matrix< Rows,    1, T >
matrix< Rows, Cols, T >::col ( ssize_type _col_idx_ ) const noexcept
{
        UTI_CEXPR_ASSERT( _col_idx_ < Rows, "nnxx::matrix::row: index out of bounds" ) ;

        matrix< rows, 1, T > mat ;

        for( ssize_type i = 0; i < rows; ++i )
        {
                mat.at( i, 0 ) = at( i, _col_idx_ ) ;
        }
        return mat ;
}

////////////////////////////////////////////////////////////////////////////////

template< ssize_t Rows, ssize_t Cols, uti::meta::arithmetic T >
        requires( Rows >= 1 && Cols >= 1 )
template< ssize_t X1, ssize_t Y1, ssize_t X2, ssize_t Y2 >
UTI_NODISCARD constexpr
matrix< X2 - X1 + 1, Y2 - Y1 + 1, T >
matrix< Rows, Cols, T >::submatrix () const noexcept
{
        matrix< X2 - X1 + 1, Y2 - Y1 + 1, T > submat ;

        for( ssize_type i = X1; i <= X2; ++i )
        {
                for( ssize_type j = Y1; j <= Y2; ++j )
                {
                        submat.at( i - X1, j - Y1 ) = at( i, j ) ;
                }
        }
        return submat ;
}

////////////////////////////////////////////////////////////////////////////////


} // namespace nnxx
