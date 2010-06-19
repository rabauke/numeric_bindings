//
// Copyright (c) 2002--2010
// Toon Knapen, Karl Meerbergen, Kresimir Fresl,
// Thomas Klimpel and Rutger ter Borg
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
// THIS FILE IS AUTOMATICALLY GENERATED
// PLEASE DO NOT EDIT!
//

#ifndef BOOST_NUMERIC_BINDINGS_LAPACK_COMPUTATIONAL_LATRS_HPP
#define BOOST_NUMERIC_BINDINGS_LAPACK_COMPUTATIONAL_LATRS_HPP

#include <boost/assert.hpp>
#include <boost/numeric/bindings/begin.hpp>
#include <boost/numeric/bindings/blas/detail/default_order.hpp>
#include <boost/numeric/bindings/diag_tag.hpp>
#include <boost/numeric/bindings/is_complex.hpp>
#include <boost/numeric/bindings/is_mutable.hpp>
#include <boost/numeric/bindings/is_real.hpp>
#include <boost/numeric/bindings/remove_imaginary.hpp>
#include <boost/numeric/bindings/size.hpp>
#include <boost/numeric/bindings/stride.hpp>
#include <boost/numeric/bindings/trans_tag.hpp>
#include <boost/numeric/bindings/uplo_tag.hpp>
#include <boost/numeric/bindings/value_type.hpp>
#include <boost/static_assert.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/type_traits/remove_const.hpp>
#include <boost/utility/enable_if.hpp>

//
// The LAPACK-backend for latrs is the netlib-compatible backend.
//
#include <boost/numeric/bindings/lapack/detail/lapack.h>
#include <boost/numeric/bindings/lapack/detail/lapack_option.hpp>

namespace boost {
namespace numeric {
namespace bindings {
namespace lapack {

//
// The detail namespace contains value-type-overloaded functions that
// dispatch to the appropriate back-end LAPACK-routine.
//
namespace detail {

//
// Overloaded function for dispatching to
// * netlib-compatible LAPACK backend (the default), and
// * float value-type.
//
template< typename Trans, typename Diag >
inline std::ptrdiff_t latrs( const char uplo, const Trans trans,
        const Diag diag, const char normin, const fortran_int_t n,
        const float* a, const fortran_int_t lda, float* x, float& scale,
        float* cnorm ) {
    fortran_int_t info(0);
    LAPACK_SLATRS( &uplo, &lapack_option< Trans >::value, &lapack_option<
            Diag >::value, &normin, &n, a, &lda, x, &scale, cnorm, &info );
    return info;
}

//
// Overloaded function for dispatching to
// * netlib-compatible LAPACK backend (the default), and
// * double value-type.
//
template< typename Trans, typename Diag >
inline std::ptrdiff_t latrs( const char uplo, const Trans trans,
        const Diag diag, const char normin, const fortran_int_t n,
        const double* a, const fortran_int_t lda, double* x, double& scale,
        double* cnorm ) {
    fortran_int_t info(0);
    LAPACK_DLATRS( &uplo, &lapack_option< Trans >::value, &lapack_option<
            Diag >::value, &normin, &n, a, &lda, x, &scale, cnorm, &info );
    return info;
}

//
// Overloaded function for dispatching to
// * netlib-compatible LAPACK backend (the default), and
// * complex<float> value-type.
//
template< typename Trans, typename Diag >
inline std::ptrdiff_t latrs( const char uplo, const Trans trans,
        const Diag diag, const char normin, const fortran_int_t n,
        const std::complex<float>* a, const fortran_int_t lda,
        std::complex<float>* x, float& scale, float* cnorm ) {
    fortran_int_t info(0);
    LAPACK_CLATRS( &uplo, &lapack_option< Trans >::value, &lapack_option<
            Diag >::value, &normin, &n, a, &lda, x, &scale, cnorm, &info );
    return info;
}

//
// Overloaded function for dispatching to
// * netlib-compatible LAPACK backend (the default), and
// * complex<double> value-type.
//
template< typename Trans, typename Diag >
inline std::ptrdiff_t latrs( const char uplo, const Trans trans,
        const Diag diag, const char normin, const fortran_int_t n,
        const std::complex<double>* a, const fortran_int_t lda,
        std::complex<double>* x, double& scale, double* cnorm ) {
    fortran_int_t info(0);
    LAPACK_ZLATRS( &uplo, &lapack_option< Trans >::value, &lapack_option<
            Diag >::value, &normin, &n, a, &lda, x, &scale, cnorm, &info );
    return info;
}

} // namespace detail

//
// Value-type based template class. Use this class if you need a type
// for dispatching to latrs.
//
template< typename Value, typename Enable = void >
struct latrs_impl {};

//
// This implementation is enabled if Value is a real type.
//
template< typename Value >
struct latrs_impl< Value, typename boost::enable_if< is_real< Value > >::type > {

    typedef Value value_type;
    typedef typename remove_imaginary< Value >::type real_type;

    //
    // Static member function, that
    // * Deduces the required arguments for dispatching to LAPACK, and
    // * Asserts that most arguments make sense.
    //
    template< typename MatrixA, typename VectorX, typename VectorCNORM >
    static std::ptrdiff_t invoke( const char uplo, const char normin,
            const MatrixA& a, VectorX& x, real_type& scale,
            VectorCNORM& cnorm ) {
        namespace bindings = ::boost::numeric::bindings;
        typedef typename detail::default_order< MatrixA >::type order;
        typedef typename result_of::trans_tag< MatrixA, order >::type trans;
        typedef typename result_of::diag_tag< MatrixA >::type diag;
        BOOST_STATIC_ASSERT( (boost::is_same< typename remove_const<
                typename bindings::value_type< MatrixA >::type >::type,
                typename remove_const< typename bindings::value_type<
                VectorX >::type >::type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename remove_const<
                typename bindings::value_type< MatrixA >::type >::type,
                typename remove_const< typename bindings::value_type<
                VectorCNORM >::type >::type >::value) );
        BOOST_STATIC_ASSERT( (bindings::is_mutable< VectorX >::value) );
        BOOST_STATIC_ASSERT( (bindings::is_mutable< VectorCNORM >::value) );
        BOOST_ASSERT( bindings::size(x) >= bindings::size_column_op(a,
                trans()) );
        BOOST_ASSERT( bindings::size_column_op(a, trans()) >= 0 );
        BOOST_ASSERT( bindings::size_minor(a) == 1 ||
                bindings::stride_minor(a) == 1 );
        BOOST_ASSERT( bindings::stride_major(a) >= ?MAX );
        BOOST_ASSERT( normin == 'Y' || normin == 'N' );
        return detail::latrs( uplo, trans(), diag(), normin,
                bindings::size_column_op(a, trans()),
                bindings::begin_value(a), bindings::stride_major(a),
                bindings::begin_value(x), scale,
                bindings::begin_value(cnorm) );
    }

};

//
// This implementation is enabled if Value is a complex type.
//
template< typename Value >
struct latrs_impl< Value, typename boost::enable_if< is_complex< Value > >::type > {

    typedef Value value_type;
    typedef typename remove_imaginary< Value >::type real_type;

    //
    // Static member function, that
    // * Deduces the required arguments for dispatching to LAPACK, and
    // * Asserts that most arguments make sense.
    //
    template< typename MatrixA, typename VectorX, typename VectorCNORM >
    static std::ptrdiff_t invoke( const char uplo, const char normin,
            const MatrixA& a, VectorX& x, real_type& scale,
            VectorCNORM& cnorm ) {
        namespace bindings = ::boost::numeric::bindings;
        typedef typename detail::default_order< MatrixA >::type order;
        typedef typename result_of::trans_tag< MatrixA, order >::type trans;
        typedef typename result_of::diag_tag< MatrixA >::type diag;
        BOOST_STATIC_ASSERT( (boost::is_same< typename remove_const<
                typename bindings::value_type< MatrixA >::type >::type,
                typename remove_const< typename bindings::value_type<
                VectorX >::type >::type >::value) );
        BOOST_STATIC_ASSERT( (bindings::is_mutable< VectorX >::value) );
        BOOST_STATIC_ASSERT( (bindings::is_mutable< VectorCNORM >::value) );
        BOOST_ASSERT( bindings::size(x) >= bindings::size_column_op(a,
                trans()) );
        BOOST_ASSERT( bindings::size_column_op(a, trans()) >= 0 );
        BOOST_ASSERT( bindings::size_minor(a) == 1 ||
                bindings::stride_minor(a) == 1 );
        BOOST_ASSERT( bindings::stride_major(a) >= ?MAX );
        BOOST_ASSERT( normin == 'Y' || normin == 'N' );
        return detail::latrs( uplo, trans(), diag(), normin,
                bindings::size_column_op(a, trans()),
                bindings::begin_value(a), bindings::stride_major(a),
                bindings::begin_value(x), scale,
                bindings::begin_value(cnorm) );
    }

};


//
// Functions for direct use. These functions are overloaded for temporaries,
// so that wrapped types can still be passed and used for write-access. In
// addition, if applicable, they are overloaded for user-defined workspaces.
// Calls to these functions are passed to the latrs_impl classes. In the 
// documentation, most overloads are collapsed to avoid a large number of
// prototypes which are very similar.
//

//
// Overloaded function for latrs. Its overload differs for
//
template< typename MatrixA, typename VectorX, typename VectorCNORM >
inline std::ptrdiff_t latrs( const char uplo, const char normin,
        const MatrixA& a, VectorX& x, typename remove_imaginary<
        typename bindings::value_type< MatrixA >::type >::type& scale,
        VectorCNORM& cnorm ) {
    return latrs_impl< typename bindings::value_type<
            MatrixA >::type >::invoke( uplo, normin, a, x, scale, cnorm );
}

} // namespace lapack
} // namespace bindings
} // namespace numeric
} // namespace boost

#endif
