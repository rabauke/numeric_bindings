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

#ifndef BOOST_NUMERIC_BINDINGS_BLAS_LEVEL2_HBMV_HPP
#define BOOST_NUMERIC_BINDINGS_BLAS_LEVEL2_HBMV_HPP

#include <boost/assert.hpp>
#include <boost/numeric/bindings/bandwidth.hpp>
#include <boost/numeric/bindings/begin.hpp>
#include <boost/numeric/bindings/data_order.hpp>
#include <boost/numeric/bindings/is_mutable.hpp>
#include <boost/numeric/bindings/remove_imaginary.hpp>
#include <boost/numeric/bindings/size.hpp>
#include <boost/numeric/bindings/stride.hpp>
#include <boost/numeric/bindings/uplo_tag.hpp>
#include <boost/numeric/bindings/value.hpp>
#include <boost/static_assert.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/type_traits/remove_const.hpp>

//
// The BLAS-backend is selected by defining a pre-processor variable,
//  which can be one of
// * for CBLAS, define BOOST_NUMERIC_BINDINGS_BLAS_CBLAS
// * for CUBLAS, define BOOST_NUMERIC_BINDINGS_BLAS_CUBLAS
// * netlib-compatible BLAS is the default
//
#if defined BOOST_NUMERIC_BINDINGS_BLAS_CBLAS
#include <boost/numeric/bindings/blas/detail/cblas.h>
#include <boost/numeric/bindings/blas/detail/cblas_option.hpp>
#elif defined BOOST_NUMERIC_BINDINGS_BLAS_CUBLAS
#include <boost/numeric/bindings/blas/detail/cublas.h>
#include <boost/numeric/bindings/blas/detail/blas_option.hpp>
#else
#include <boost/numeric/bindings/blas/detail/blas.h>
#include <boost/numeric/bindings/blas/detail/blas_option.hpp>
#endif

namespace boost {
namespace numeric {
namespace bindings {
namespace blas {

//
// The detail namespace contains value-type-overloaded functions that
// dispatch to the appropriate back-end BLAS-routine.
//
namespace detail {

#if defined BOOST_NUMERIC_BINDINGS_BLAS_CBLAS
//
// Overloaded function for dispatching to
// * CBLAS backend, and
// * complex<float> value-type.
//
template< typename Order, typename UpLo >
inline void hbmv( Order, UpLo, const int n, const int k,
        const std::complex<float> alpha, const std::complex<float>* a,
        const int lda, const std::complex<float>* x, const int incx,
        const std::complex<float> beta, std::complex<float>* y,
        const int incy ) {
    cblas_chbmv( cblas_option< Order >::value, cblas_option< UpLo >::value, n,
            k, &alpha, a, lda, x, incx, &beta, y, incy );
}

//
// Overloaded function for dispatching to
// * CBLAS backend, and
// * complex<double> value-type.
//
template< typename Order, typename UpLo >
inline void hbmv( Order, UpLo, const int n, const int k,
        const std::complex<double> alpha, const std::complex<double>* a,
        const int lda, const std::complex<double>* x, const int incx,
        const std::complex<double> beta, std::complex<double>* y,
        const int incy ) {
    cblas_zhbmv( cblas_option< Order >::value, cblas_option< UpLo >::value, n,
            k, &alpha, a, lda, x, incx, &beta, y, incy );
}

#elif defined BOOST_NUMERIC_BINDINGS_BLAS_CUBLAS
//
// Overloaded function for dispatching to
// * CUBLAS backend, and
// * complex<float> value-type.
//
template< typename Order, typename UpLo >
inline void hbmv( Order, UpLo, const int n, const int k,
        const std::complex<float> alpha, const std::complex<float>* a,
        const int lda, const std::complex<float>* x, const int incx,
        const std::complex<float> beta, std::complex<float>* y,
        const int incy ) {
    BOOST_STATIC_ASSERT( (is_same<Order, tag::column_major>::value) );
    cublasChbmv( blas_option< UpLo >::value, n, k, alpha, a, lda, x, incx,
            beta, y, incy );
}

//
// Overloaded function for dispatching to
// * CUBLAS backend, and
// * complex<double> value-type.
//
template< typename Order, typename UpLo >
inline void hbmv( Order, UpLo, const int n, const int k,
        const std::complex<double> alpha, const std::complex<double>* a,
        const int lda, const std::complex<double>* x, const int incx,
        const std::complex<double> beta, std::complex<double>* y,
        const int incy ) {
    BOOST_STATIC_ASSERT( (is_same<Order, tag::column_major>::value) );
    // NOT FOUND();
}

#else
//
// Overloaded function for dispatching to
// * netlib-compatible BLAS backend (the default), and
// * complex<float> value-type.
//
template< typename Order, typename UpLo >
inline void hbmv( Order, UpLo, const fortran_int_t n, const fortran_int_t k,
        const std::complex<float> alpha, const std::complex<float>* a,
        const fortran_int_t lda, const std::complex<float>* x,
        const fortran_int_t incx, const std::complex<float> beta,
        std::complex<float>* y, const fortran_int_t incy ) {
    BOOST_STATIC_ASSERT( (is_same<Order, tag::column_major>::value) );
    BLAS_CHBMV( &blas_option< UpLo >::value, &n, &k, &alpha, a, &lda, x,
            &incx, &beta, y, &incy );
}

//
// Overloaded function for dispatching to
// * netlib-compatible BLAS backend (the default), and
// * complex<double> value-type.
//
template< typename Order, typename UpLo >
inline void hbmv( Order, UpLo, const fortran_int_t n, const fortran_int_t k,
        const std::complex<double> alpha, const std::complex<double>* a,
        const fortran_int_t lda, const std::complex<double>* x,
        const fortran_int_t incx, const std::complex<double> beta,
        std::complex<double>* y, const fortran_int_t incy ) {
    BOOST_STATIC_ASSERT( (is_same<Order, tag::column_major>::value) );
    BLAS_ZHBMV( &blas_option< UpLo >::value, &n, &k, &alpha, a, &lda, x,
            &incx, &beta, y, &incy );
}

#endif

} // namespace detail

//
// Value-type based template class. Use this class if you need a type
// for dispatching to hbmv.
//
template< typename Value >
struct hbmv_impl {

    typedef Value value_type;
    typedef typename remove_imaginary< Value >::type real_type;
    typedef void return_type;

    //
    // Static member function that
    // * Deduces the required arguments for dispatching to BLAS, and
    // * Asserts that most arguments make sense.
    //
    template< typename MatrixA, typename VectorX, typename VectorY >
    static return_type invoke( const value_type alpha, const MatrixA& a,
            const VectorX& x, const value_type beta, VectorY& y ) {
        namespace bindings = ::boost::numeric::bindings;
        typedef typename result_of::data_order< MatrixA >::type order;
        typedef typename result_of::uplo_tag< MatrixA >::type uplo;
        BOOST_STATIC_ASSERT( (is_same< typename remove_const< typename value<
                MatrixA >::type >::type, typename remove_const<
                typename value< VectorX >::type >::type >::value) );
        BOOST_STATIC_ASSERT( (is_same< typename remove_const< typename value<
                MatrixA >::type >::type, typename remove_const<
                typename value< VectorY >::type >::type >::value) );
        BOOST_STATIC_ASSERT( (bindings::is_mutable< VectorY >::value) );
        BOOST_ASSERT( bindings::size_minor(a) == 1 ||
                bindings::stride_minor(a) == 1 );
        detail::hbmv( order(), uplo(), bindings::size_column(a),
                bindings::bandwidth_upper(a), alpha, bindings::begin_value(a),
                bindings::stride_major(a), bindings::begin_value(x),
                bindings::stride(x), beta, bindings::begin_value(y),
                bindings::stride(y) );
    }
};

//
// Functions for direct use. These functions are overloaded for temporaries,
// so that wrapped types can still be passed and used for write-access. Calls
// to these functions are passed to the hbmv_impl classes. In the 
// documentation, the const-overloads are collapsed to avoid a large number of
// prototypes which are very similar.
//

//
// Overloaded function for hbmv. Its overload differs for
// * VectorY&
//
template< typename MatrixA, typename VectorX, typename VectorY >
inline typename hbmv_impl< typename value< MatrixA >::type >::return_type
hbmv( const typename value< MatrixA >::type alpha, const MatrixA& a,
        const VectorX& x, const typename value< MatrixA >::type beta,
        VectorY& y ) {
    hbmv_impl< typename value< MatrixA >::type >::invoke( alpha, a, x,
            beta, y );
}

//
// Overloaded function for hbmv. Its overload differs for
// * const VectorY&
//
template< typename MatrixA, typename VectorX, typename VectorY >
inline typename hbmv_impl< typename value< MatrixA >::type >::return_type
hbmv( const typename value< MatrixA >::type alpha, const MatrixA& a,
        const VectorX& x, const typename value< MatrixA >::type beta,
        const VectorY& y ) {
    hbmv_impl< typename value< MatrixA >::type >::invoke( alpha, a, x,
            beta, y );
}

} // namespace blas
} // namespace bindings
} // namespace numeric
} // namespace boost

#endif
