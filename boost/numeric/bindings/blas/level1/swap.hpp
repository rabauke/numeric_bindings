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

#ifndef BOOST_NUMERIC_BINDINGS_BLAS_LEVEL1_SWAP_HPP
#define BOOST_NUMERIC_BINDINGS_BLAS_LEVEL1_SWAP_HPP

#include <boost/assert.hpp>
#include <boost/numeric/bindings/begin.hpp>
#include <boost/numeric/bindings/is_mutable.hpp>
#include <boost/numeric/bindings/remove_imaginary.hpp>
#include <boost/numeric/bindings/size.hpp>
#include <boost/numeric/bindings/stride.hpp>
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
// * float value-type.
//
inline void swap( const int n, float* x, const int incx, float* y,
        const int incy ) {
    cblas_sswap( n, x, incx, y, incy );
}

//
// Overloaded function for dispatching to
// * CBLAS backend, and
// * double value-type.
//
inline void swap( const int n, double* x, const int incx, double* y,
        const int incy ) {
    cblas_dswap( n, x, incx, y, incy );
}

//
// Overloaded function for dispatching to
// * CBLAS backend, and
// * complex<float> value-type.
//
inline void swap( const int n, std::complex<float>* x, const int incx,
        std::complex<float>* y, const int incy ) {
    cblas_cswap( n, x, incx, y, incy );
}

//
// Overloaded function for dispatching to
// * CBLAS backend, and
// * complex<double> value-type.
//
inline void swap( const int n, std::complex<double>* x, const int incx,
        std::complex<double>* y, const int incy ) {
    cblas_zswap( n, x, incx, y, incy );
}

#elif defined BOOST_NUMERIC_BINDINGS_BLAS_CUBLAS
//
// Overloaded function for dispatching to
// * CUBLAS backend, and
// * float value-type.
//
inline void swap( const int n, float* x, const int incx, float* y,
        const int incy ) {
    cublasSswap( n, x, incx, y, incy );
}

//
// Overloaded function for dispatching to
// * CUBLAS backend, and
// * double value-type.
//
inline void swap( const int n, double* x, const int incx, double* y,
        const int incy ) {
    cublasDswap( n, x, incx, y, incy );
}

//
// Overloaded function for dispatching to
// * CUBLAS backend, and
// * complex<float> value-type.
//
inline void swap( const int n, std::complex<float>* x, const int incx,
        std::complex<float>* y, const int incy ) {
    cublasCswap( n, x, incx, y, incy );
}

//
// Overloaded function for dispatching to
// * CUBLAS backend, and
// * complex<double> value-type.
//
inline void swap( const int n, std::complex<double>* x, const int incx,
        std::complex<double>* y, const int incy ) {
    // NOT FOUND();
}

#else
//
// Overloaded function for dispatching to
// * netlib-compatible BLAS backend (the default), and
// * float value-type.
//
inline void swap( const fortran_int_t n, float* x, const fortran_int_t incx,
        float* y, const fortran_int_t incy ) {
    BLAS_SSWAP( &n, x, &incx, y, &incy );
}

//
// Overloaded function for dispatching to
// * netlib-compatible BLAS backend (the default), and
// * double value-type.
//
inline void swap( const fortran_int_t n, double* x, const fortran_int_t incx,
        double* y, const fortran_int_t incy ) {
    BLAS_DSWAP( &n, x, &incx, y, &incy );
}

//
// Overloaded function for dispatching to
// * netlib-compatible BLAS backend (the default), and
// * complex<float> value-type.
//
inline void swap( const fortran_int_t n, std::complex<float>* x,
        const fortran_int_t incx, std::complex<float>* y,
        const fortran_int_t incy ) {
    BLAS_CSWAP( &n, x, &incx, y, &incy );
}

//
// Overloaded function for dispatching to
// * netlib-compatible BLAS backend (the default), and
// * complex<double> value-type.
//
inline void swap( const fortran_int_t n, std::complex<double>* x,
        const fortran_int_t incx, std::complex<double>* y,
        const fortran_int_t incy ) {
    BLAS_ZSWAP( &n, x, &incx, y, &incy );
}

#endif

} // namespace detail

//
// Value-type based template class. Use this class if you need a type
// for dispatching to swap.
//
template< typename Value >
struct swap_impl {

    typedef Value value_type;
    typedef typename remove_imaginary< Value >::type real_type;
    typedef void return_type;

    //
    // Static member function that
    // * Deduces the required arguments for dispatching to BLAS, and
    // * Asserts that most arguments make sense.
    //
    template< typename VectorX, typename VectorY >
    static return_type invoke( VectorX& x, VectorY& y ) {
        BOOST_STATIC_ASSERT( (is_same< typename remove_const< typename value<
                VectorX >::type >::type, typename remove_const<
                typename value< VectorY >::type >::type >::value) );
        BOOST_STATIC_ASSERT( (is_mutable< VectorX >::value) );
        BOOST_STATIC_ASSERT( (is_mutable< VectorY >::value) );
        detail::swap( size(x), begin_value(x), stride(x),
                begin_value(y), stride(y) );
    }
};

//
// Functions for direct use. These functions are overloaded for temporaries,
// so that wrapped types can still be passed and used for write-access. Calls
// to these functions are passed to the swap_impl classes. In the 
// documentation, the const-overloads are collapsed to avoid a large number of
// prototypes which are very similar.
//

//
// Overloaded function for swap. Its overload differs for
// * VectorX&
// * VectorY&
//
template< typename VectorX, typename VectorY >
inline typename swap_impl< typename value< VectorX >::type >::return_type
swap( VectorX& x, VectorY& y ) {
    swap_impl< typename value< VectorX >::type >::invoke( x, y );
}

//
// Overloaded function for swap. Its overload differs for
// * const VectorX&
// * VectorY&
//
template< typename VectorX, typename VectorY >
inline typename swap_impl< typename value< VectorX >::type >::return_type
swap( const VectorX& x, VectorY& y ) {
    swap_impl< typename value< VectorX >::type >::invoke( x, y );
}

//
// Overloaded function for swap. Its overload differs for
// * VectorX&
// * const VectorY&
//
template< typename VectorX, typename VectorY >
inline typename swap_impl< typename value< VectorX >::type >::return_type
swap( VectorX& x, const VectorY& y ) {
    swap_impl< typename value< VectorX >::type >::invoke( x, y );
}

//
// Overloaded function for swap. Its overload differs for
// * const VectorX&
// * const VectorY&
//
template< typename VectorX, typename VectorY >
inline typename swap_impl< typename value< VectorX >::type >::return_type
swap( const VectorX& x, const VectorY& y ) {
    swap_impl< typename value< VectorX >::type >::invoke( x, y );
}

} // namespace blas
} // namespace bindings
} // namespace numeric
} // namespace boost

#endif
