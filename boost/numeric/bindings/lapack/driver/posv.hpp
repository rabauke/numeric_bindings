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

#ifndef BOOST_NUMERIC_BINDINGS_LAPACK_DRIVER_POSV_HPP
#define BOOST_NUMERIC_BINDINGS_LAPACK_DRIVER_POSV_HPP

#include <boost/assert.hpp>
#include <boost/numeric/bindings/begin.hpp>
#include <boost/numeric/bindings/data_side.hpp>
#include <boost/numeric/bindings/is_mutable.hpp>
#include <boost/numeric/bindings/remove_imaginary.hpp>
#include <boost/numeric/bindings/size.hpp>
#include <boost/numeric/bindings/stride.hpp>
#include <boost/numeric/bindings/value.hpp>
#include <boost/static_assert.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/type_traits/remove_const.hpp>

//
// The LAPACK-backend for posv is selected by defining a pre-processor
// variable, which can be one of
// * for ATLAS's CLAPACK, define BOOST_NUMERIC_BINDINGS_LAPACK_CLAPACK
// * netlib-compatible LAPACK is the default
//
#if defined BOOST_NUMERIC_BINDINGS_LAPACK_CLAPACK
#include <boost/numeric/bindings/lapack/detail/clapack.h>
#include <boost/numeric/bindings/lapack/detail/clapack_option.hpp>
#else
#include <boost/numeric/bindings/lapack/detail/lapack.h>
#include <boost/numeric/bindings/lapack/detail/lapack_option.hpp>
#endif

namespace boost {
namespace numeric {
namespace bindings {
namespace lapack {

//
// The detail namespace contains value-type-overloaded functions that
// dispatch to the appropriate back-end LAPACK-routine.
//
namespace detail {

#if defined BOOST_NUMERIC_BINDINGS_LAPACK_CLAPACK
//
// Overloaded function for dispatching to
// * ATLAS's CLAPACK backend, and
// * float value-type.
//
template< typename Order, typename UpLo >
inline std::ptrdiff_t posv( Order, UpLo, const int n, const int nrhs, float* a,
        const int lda, float* b, const int ldb ) {
    return clapack_sposv( clapack_option< Order >::value, clapack_option<
            UpLo >::value, n, nrhs, a, lda, b, ldb );
}

//
// Overloaded function for dispatching to
// * ATLAS's CLAPACK backend, and
// * double value-type.
//
template< typename Order, typename UpLo >
inline std::ptrdiff_t posv( Order, UpLo, const int n, const int nrhs,
        double* a, const int lda, double* b, const int ldb ) {
    return clapack_dposv( clapack_option< Order >::value, clapack_option<
            UpLo >::value, n, nrhs, a, lda, b, ldb );
}

//
// Overloaded function for dispatching to
// * ATLAS's CLAPACK backend, and
// * complex<float> value-type.
//
template< typename Order, typename UpLo >
inline std::ptrdiff_t posv( Order, UpLo, const int n, const int nrhs,
        std::complex<float>* a, const int lda, std::complex<float>* b,
        const int ldb ) {
    return clapack_cposv( clapack_option< Order >::value, clapack_option<
            UpLo >::value, n, nrhs, a, lda, b, ldb );
}

//
// Overloaded function for dispatching to
// * ATLAS's CLAPACK backend, and
// * complex<double> value-type.
//
template< typename Order, typename UpLo >
inline std::ptrdiff_t posv( Order, UpLo, const int n, const int nrhs,
        std::complex<double>* a, const int lda, std::complex<double>* b,
        const int ldb ) {
    return clapack_zposv( clapack_option< Order >::value, clapack_option<
            UpLo >::value, n, nrhs, a, lda, b, ldb );
}

#else
//
// Overloaded function for dispatching to
// * netlib-compatible LAPACK backend (the default), and
// * float value-type.
//
template< typename Order, typename UpLo >
inline std::ptrdiff_t posv( Order, UpLo, const fortran_int_t n,
        const fortran_int_t nrhs, float* a, const fortran_int_t lda, float* b,
        const fortran_int_t ldb ) {
    BOOST_STATIC_ASSERT( (is_same<Order, tag::column_major>::value) );
    fortran_int_t info(0);
    LAPACK_SPOSV( &lapack_option< UpLo >::value, &n, &nrhs, a, &lda, b, &ldb,
            &info );
    return info;
}

//
// Overloaded function for dispatching to
// * netlib-compatible LAPACK backend (the default), and
// * double value-type.
//
template< typename Order, typename UpLo >
inline std::ptrdiff_t posv( Order, UpLo, const fortran_int_t n,
        const fortran_int_t nrhs, double* a, const fortran_int_t lda,
        double* b, const fortran_int_t ldb ) {
    BOOST_STATIC_ASSERT( (is_same<Order, tag::column_major>::value) );
    fortran_int_t info(0);
    LAPACK_DPOSV( &lapack_option< UpLo >::value, &n, &nrhs, a, &lda, b, &ldb,
            &info );
    return info;
}

//
// Overloaded function for dispatching to
// * netlib-compatible LAPACK backend (the default), and
// * complex<float> value-type.
//
template< typename Order, typename UpLo >
inline std::ptrdiff_t posv( Order, UpLo, const fortran_int_t n,
        const fortran_int_t nrhs, std::complex<float>* a,
        const fortran_int_t lda, std::complex<float>* b,
        const fortran_int_t ldb ) {
    BOOST_STATIC_ASSERT( (is_same<Order, tag::column_major>::value) );
    fortran_int_t info(0);
    LAPACK_CPOSV( &lapack_option< UpLo >::value, &n, &nrhs, a, &lda, b, &ldb,
            &info );
    return info;
}

//
// Overloaded function for dispatching to
// * netlib-compatible LAPACK backend (the default), and
// * complex<double> value-type.
//
template< typename Order, typename UpLo >
inline std::ptrdiff_t posv( Order, UpLo, const fortran_int_t n,
        const fortran_int_t nrhs, std::complex<double>* a,
        const fortran_int_t lda, std::complex<double>* b,
        const fortran_int_t ldb ) {
    BOOST_STATIC_ASSERT( (is_same<Order, tag::column_major>::value) );
    fortran_int_t info(0);
    LAPACK_ZPOSV( &lapack_option< UpLo >::value, &n, &nrhs, a, &lda, b, &ldb,
            &info );
    return info;
}

#endif
} // namespace detail

//
// Value-type based template class. Use this class if you need a type
// for dispatching to posv.
//
template< typename Value >
struct posv_impl {

    typedef Value value_type;
    typedef typename remove_imaginary< Value >::type real_type;
    typedef tag::column_major order;

    //
    // Static member function, that
    // * Deduces the required arguments for dispatching to LAPACK, and
    // * Asserts that most arguments make sense.
    //
    template< typename MatrixA, typename MatrixB >
    static std::ptrdiff_t invoke( MatrixA& a, MatrixB& b ) {
        namespace bindings = ::boost::numeric::bindings;
        typedef typename result_of::data_side< MatrixA >::type uplo;
        BOOST_STATIC_ASSERT( (boost::is_same< typename remove_const<
                typename value< MatrixA >::type >::type,
                typename remove_const< typename value<
                MatrixB >::type >::type >::value) );
        BOOST_STATIC_ASSERT( (bindings::is_mutable< MatrixA >::value) );
        BOOST_STATIC_ASSERT( (bindings::is_mutable< MatrixB >::value) );
        BOOST_ASSERT( bindings::size_column(a) >= 0 );
        BOOST_ASSERT( bindings::size_column(b) >= 0 );
        BOOST_ASSERT( bindings::size_minor(a) == 1 ||
                bindings::stride_minor(a) == 1 );
        BOOST_ASSERT( bindings::size_minor(b) == 1 ||
                bindings::stride_minor(b) == 1 );
        BOOST_ASSERT( bindings::stride_major(a) >= std::max< std::ptrdiff_t >(1,
                bindings::size_column(a)) );
        BOOST_ASSERT( bindings::stride_major(b) >= std::max< std::ptrdiff_t >(1,
                bindings::size_column(a)) );
        return detail::posv( order(), uplo(), bindings::size_column(a),
                bindings::size_column(b), bindings::begin_value(a),
                bindings::stride_major(a), bindings::begin_value(b),
                bindings::stride_major(b) );
    }

};


//
// Functions for direct use. These functions are overloaded for temporaries,
// so that wrapped types can still be passed and used for write-access. In
// addition, if applicable, they are overloaded for user-defined workspaces.
// Calls to these functions are passed to the posv_impl classes. In the 
// documentation, most overloads are collapsed to avoid a large number of
// prototypes which are very similar.
//

//
// Overloaded function for posv. Its overload differs for
// * MatrixA&
// * MatrixB&
//
template< typename MatrixA, typename MatrixB >
inline std::ptrdiff_t posv( MatrixA& a, MatrixB& b ) {
    return posv_impl< typename value< MatrixA >::type >::invoke( a, b );
}

//
// Overloaded function for posv. Its overload differs for
// * const MatrixA&
// * MatrixB&
//
template< typename MatrixA, typename MatrixB >
inline std::ptrdiff_t posv( const MatrixA& a, MatrixB& b ) {
    return posv_impl< typename value< MatrixA >::type >::invoke( a, b );
}

//
// Overloaded function for posv. Its overload differs for
// * MatrixA&
// * const MatrixB&
//
template< typename MatrixA, typename MatrixB >
inline std::ptrdiff_t posv( MatrixA& a, const MatrixB& b ) {
    return posv_impl< typename value< MatrixA >::type >::invoke( a, b );
}

//
// Overloaded function for posv. Its overload differs for
// * const MatrixA&
// * const MatrixB&
//
template< typename MatrixA, typename MatrixB >
inline std::ptrdiff_t posv( const MatrixA& a, const MatrixB& b ) {
    return posv_impl< typename value< MatrixA >::type >::invoke( a, b );
}

} // namespace lapack
} // namespace bindings
} // namespace numeric
} // namespace boost

#endif
