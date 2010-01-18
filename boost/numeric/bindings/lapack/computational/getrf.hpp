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

#ifndef BOOST_NUMERIC_BINDINGS_LAPACK_COMPUTATIONAL_GETRF_HPP
#define BOOST_NUMERIC_BINDINGS_LAPACK_COMPUTATIONAL_GETRF_HPP

#include <boost/assert.hpp>
#include <boost/numeric/bindings/begin.hpp>
#include <boost/numeric/bindings/is_mutable.hpp>
#include <boost/numeric/bindings/remove_imaginary.hpp>
#include <boost/numeric/bindings/size.hpp>
#include <boost/numeric/bindings/stride.hpp>
#include <boost/numeric/bindings/value_type.hpp>
#include <boost/static_assert.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/type_traits/remove_const.hpp>

//
// The LAPACK-backend for getrf is selected by defining a pre-processor
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
template< typename Order >
inline std::ptrdiff_t getrf( Order, const int m, const int n, float* a,
        const int lda, int* ipiv ) {
    return clapack_sgetrf( clapack_option< Order >::value, m, n, a, lda,
            ipiv );
}

//
// Overloaded function for dispatching to
// * ATLAS's CLAPACK backend, and
// * double value-type.
//
template< typename Order >
inline std::ptrdiff_t getrf( Order, const int m, const int n, double* a,
        const int lda, int* ipiv ) {
    return clapack_dgetrf( clapack_option< Order >::value, m, n, a, lda,
            ipiv );
}

//
// Overloaded function for dispatching to
// * ATLAS's CLAPACK backend, and
// * complex<float> value-type.
//
template< typename Order >
inline std::ptrdiff_t getrf( Order, const int m, const int n,
        std::complex<float>* a, const int lda, int* ipiv ) {
    return clapack_cgetrf( clapack_option< Order >::value, m, n, a, lda,
            ipiv );
}

//
// Overloaded function for dispatching to
// * ATLAS's CLAPACK backend, and
// * complex<double> value-type.
//
template< typename Order >
inline std::ptrdiff_t getrf( Order, const int m, const int n,
        std::complex<double>* a, const int lda, int* ipiv ) {
    return clapack_zgetrf( clapack_option< Order >::value, m, n, a, lda,
            ipiv );
}

#else
//
// Overloaded function for dispatching to
// * netlib-compatible LAPACK backend (the default), and
// * float value-type.
//
template< typename Order >
inline std::ptrdiff_t getrf( Order, const fortran_int_t m,
        const fortran_int_t n, float* a, const fortran_int_t lda,
        fortran_int_t* ipiv ) {
    BOOST_STATIC_ASSERT( (is_same<Order, tag::column_major>::value) );
    fortran_int_t info(0);
    LAPACK_SGETRF( &m, &n, a, &lda, ipiv, &info );
    return info;
}

//
// Overloaded function for dispatching to
// * netlib-compatible LAPACK backend (the default), and
// * double value-type.
//
template< typename Order >
inline std::ptrdiff_t getrf( Order, const fortran_int_t m,
        const fortran_int_t n, double* a, const fortran_int_t lda,
        fortran_int_t* ipiv ) {
    BOOST_STATIC_ASSERT( (is_same<Order, tag::column_major>::value) );
    fortran_int_t info(0);
    LAPACK_DGETRF( &m, &n, a, &lda, ipiv, &info );
    return info;
}

//
// Overloaded function for dispatching to
// * netlib-compatible LAPACK backend (the default), and
// * complex<float> value-type.
//
template< typename Order >
inline std::ptrdiff_t getrf( Order, const fortran_int_t m,
        const fortran_int_t n, std::complex<float>* a,
        const fortran_int_t lda, fortran_int_t* ipiv ) {
    BOOST_STATIC_ASSERT( (is_same<Order, tag::column_major>::value) );
    fortran_int_t info(0);
    LAPACK_CGETRF( &m, &n, a, &lda, ipiv, &info );
    return info;
}

//
// Overloaded function for dispatching to
// * netlib-compatible LAPACK backend (the default), and
// * complex<double> value-type.
//
template< typename Order >
inline std::ptrdiff_t getrf( Order, const fortran_int_t m,
        const fortran_int_t n, std::complex<double>* a,
        const fortran_int_t lda, fortran_int_t* ipiv ) {
    BOOST_STATIC_ASSERT( (is_same<Order, tag::column_major>::value) );
    fortran_int_t info(0);
    LAPACK_ZGETRF( &m, &n, a, &lda, ipiv, &info );
    return info;
}

#endif
} // namespace detail

//
// Value-type based template class. Use this class if you need a type
// for dispatching to getrf.
//
template< typename Value >
struct getrf_impl {

    typedef Value value_type;
    typedef typename remove_imaginary< Value >::type real_type;
    typedef tag::column_major order;

    //
    // Static member function, that
    // * Deduces the required arguments for dispatching to LAPACK, and
    // * Asserts that most arguments make sense.
    //
    template< typename MatrixA, typename VectorIPIV >
    static std::ptrdiff_t invoke( MatrixA& a, VectorIPIV& ipiv ) {
        namespace bindings = ::boost::numeric::bindings;
        BOOST_STATIC_ASSERT( (bindings::is_mutable< MatrixA >::value) );
        BOOST_STATIC_ASSERT( (bindings::is_mutable< VectorIPIV >::value) );
        BOOST_ASSERT( bindings::size(ipiv) >= std::min<
                std::ptrdiff_t >(bindings::size_row(a),
                bindings::size_column(a)) );
        BOOST_ASSERT( bindings::size_column(a) >= 0 );
        BOOST_ASSERT( bindings::size_minor(a) == 1 ||
                bindings::stride_minor(a) == 1 );
        BOOST_ASSERT( bindings::size_row(a) >= 0 );
        BOOST_ASSERT( bindings::stride_major(a) >= std::max< std::ptrdiff_t >(1,
                bindings::size_row(a)) );
        return detail::getrf( order(), bindings::size_row(a),
                bindings::size_column(a), bindings::begin_value(a),
                bindings::stride_major(a), bindings::begin_value(ipiv) );
    }

};


//
// Functions for direct use. These functions are overloaded for temporaries,
// so that wrapped types can still be passed and used for write-access. In
// addition, if applicable, they are overloaded for user-defined workspaces.
// Calls to these functions are passed to the getrf_impl classes. In the 
// documentation, most overloads are collapsed to avoid a large number of
// prototypes which are very similar.
//

//
// Overloaded function for getrf. Its overload differs for
// * MatrixA&
// * VectorIPIV&
//
template< typename MatrixA, typename VectorIPIV >
inline std::ptrdiff_t getrf( MatrixA& a, VectorIPIV& ipiv ) {
    return getrf_impl< typename bindings::value_type<
            MatrixA >::type >::invoke( a, ipiv );
}

//
// Overloaded function for getrf. Its overload differs for
// * const MatrixA&
// * VectorIPIV&
//
template< typename MatrixA, typename VectorIPIV >
inline std::ptrdiff_t getrf( const MatrixA& a, VectorIPIV& ipiv ) {
    return getrf_impl< typename bindings::value_type<
            MatrixA >::type >::invoke( a, ipiv );
}

//
// Overloaded function for getrf. Its overload differs for
// * MatrixA&
// * const VectorIPIV&
//
template< typename MatrixA, typename VectorIPIV >
inline std::ptrdiff_t getrf( MatrixA& a, const VectorIPIV& ipiv ) {
    return getrf_impl< typename bindings::value_type<
            MatrixA >::type >::invoke( a, ipiv );
}

//
// Overloaded function for getrf. Its overload differs for
// * const MatrixA&
// * const VectorIPIV&
//
template< typename MatrixA, typename VectorIPIV >
inline std::ptrdiff_t getrf( const MatrixA& a, const VectorIPIV& ipiv ) {
    return getrf_impl< typename bindings::value_type<
            MatrixA >::type >::invoke( a, ipiv );
}

} // namespace lapack
} // namespace bindings
} // namespace numeric
} // namespace boost

#endif
