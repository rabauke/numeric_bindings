//
// Copyright (c) 2003--2009
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

#ifndef BOOST_NUMERIC_BINDINGS_BLAS_LEVEL1_ROTM_HPP
#define BOOST_NUMERIC_BINDINGS_BLAS_LEVEL1_ROTM_HPP

#include <boost/assert.hpp>
#include <boost/numeric/bindings/is_column_major.hpp>
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
// * CBLAS backend
// * float value-type
//
template< typename Order >
inline void rotm( Order, const std::ptrdiff_t n, float* x,
        const std::ptrdiff_t incx, float* y, const std::ptrdiff_t incy,
        float* param ) {
    cblas_srotm( cblas_option< Order >::value, n, x, incx, y, incy, param );
}

//
// Overloaded function for dispatching to
// * CBLAS backend
// * double value-type
//
template< typename Order >
inline void rotm( Order, const std::ptrdiff_t n, double* x,
        const std::ptrdiff_t incx, double* y, const std::ptrdiff_t incy,
        double* param ) {
    cblas_drotm( cblas_option< Order >::value, n, x, incx, y, incy, param );
}

#elif defined BOOST_NUMERIC_BINDINGS_BLAS_CUBLAS
//
// Overloaded function for dispatching to
// * CUBLAS backend
// * float value-type
//
template< typename Order >
inline void rotm( Order, const std::ptrdiff_t n, float* x,
        const std::ptrdiff_t incx, float* y, const std::ptrdiff_t incy,
        float* param ) {
    BOOST_STATIC_ASSERT( (is_column_major<Order>::value) );
    cublasSrotm( n, x, incx, y, incy, param );
}

//
// Overloaded function for dispatching to
// * CUBLAS backend
// * double value-type
//
template< typename Order >
inline void rotm( Order, const std::ptrdiff_t n, double* x,
        const std::ptrdiff_t incx, double* y, const std::ptrdiff_t incy,
        double* param ) {
    BOOST_STATIC_ASSERT( (is_column_major<Order>::value) );
    cublasDrotm( n, x, incx, y, incy, param );
}

#else
//
// Overloaded function for dispatching to
// * netlib-compatible BLAS backend (the default)
// * float value-type
//
template< typename Order >
inline void rotm( Order, const std::ptrdiff_t n, float* x,
        const std::ptrdiff_t incx, float* y, const std::ptrdiff_t incy,
        float* param ) {
    BOOST_STATIC_ASSERT( (is_column_major<Order>::value) );
    BLAS_SROTM( &n, x, &incx, y, &incy, param );
}

//
// Overloaded function for dispatching to
// * netlib-compatible BLAS backend (the default)
// * double value-type
//
template< typename Order >
inline void rotm( Order, const std::ptrdiff_t n, double* x,
        const std::ptrdiff_t incx, double* y, const std::ptrdiff_t incy,
        double* param ) {
    BOOST_STATIC_ASSERT( (is_column_major<Order>::value) );
    BLAS_DROTM( &n, x, &incx, y, &incy, param );
}

#endif

} // namespace detail

//
// Value-type based template class. Use this class if you need a type
// for dispatching to rotm.
//
template< typename Value >
struct rotm_impl {

    typedef Value value_type;
    typedef typename remove_imaginary< Value >::type real_type;
    typedef void return_type;

    //
    // Static member function that
    // * Deduces the required arguments for dispatching to BLAS, and
    // * Asserts that most arguments make sense.
    //
    template< typename VectorX, typename VectorY, typename VectorPARAM >
    static return_type invoke( const std::ptrdiff_t n, VectorX& x,
            const std::ptrdiff_t incx, VectorY& y, const std::ptrdiff_t incy,
            VectorPARAM& param ) {
        BOOST_STATIC_ASSERT( (is_same< typename remove_const< typename value<
                VectorX >::type >::type, typename remove_const<
                typename value< VectorY >::type >::type >::value) );
        BOOST_STATIC_ASSERT( (is_same< typename remove_const< typename value<
                VectorX >::type >::type, typename remove_const<
                typename value< VectorPARAM >::type >::type >::value) );
        BOOST_STATIC_ASSERT( (is_mutable< VectorX >::value ) );
        BOOST_STATIC_ASSERT( (is_mutable< VectorY >::value ) );
        BOOST_STATIC_ASSERT( (is_mutable< VectorPARAM >::value ) );
        
        detail::rotm( n, begin_value(x), incx, begin_value(y), incy,
                begin_value(param) );
    }
};

//
// Functions for direct use. These functions are overloaded for temporaries,
// so that wrapped types can still be passed and used for write-access. Calls
// to these functions are passed to the rotm_impl classes. In the 
// documentation, the const-overloads are collapsed to avoid a large number of
// prototypes which are very similar.
//

//
// Overloaded function for rotm. Its overload differs for
// * VectorX&
    // * VectorY&
    // * VectorPARAM&
//
template< typename VectorX, typename VectorY, typename VectorPARAM >
inline typename rotm_impl< typename value< VectorX >::type >::return_type
rotm( const std::ptrdiff_t n, VectorX& x, const std::ptrdiff_t incx,
        VectorY& y, const std::ptrdiff_t incy, VectorPARAM& param ) {
    rotm_impl< typename value< VectorX >::type >::invoke( n, x, incx, y,
            incy, param );
}

//
// Overloaded function for rotm. Its overload differs for
// * const VectorX&
    // * VectorY&
    // * VectorPARAM&
//
template< typename VectorX, typename VectorY, typename VectorPARAM >
inline typename rotm_impl< typename value< VectorX >::type >::return_type
rotm( const std::ptrdiff_t n, const VectorX& x,
        const std::ptrdiff_t incx, VectorY& y, const std::ptrdiff_t incy,
        VectorPARAM& param ) {
    rotm_impl< typename value< VectorX >::type >::invoke( n, x, incx, y,
            incy, param );
}

//
// Overloaded function for rotm. Its overload differs for
// * VectorX&
    // * const VectorY&
    // * VectorPARAM&
//
template< typename VectorX, typename VectorY, typename VectorPARAM >
inline typename rotm_impl< typename value< VectorX >::type >::return_type
rotm( const std::ptrdiff_t n, VectorX& x, const std::ptrdiff_t incx,
        const VectorY& y, const std::ptrdiff_t incy, VectorPARAM& param ) {
    rotm_impl< typename value< VectorX >::type >::invoke( n, x, incx, y,
            incy, param );
}

//
// Overloaded function for rotm. Its overload differs for
// * const VectorX&
    // * const VectorY&
    // * VectorPARAM&
//
template< typename VectorX, typename VectorY, typename VectorPARAM >
inline typename rotm_impl< typename value< VectorX >::type >::return_type
rotm( const std::ptrdiff_t n, const VectorX& x,
        const std::ptrdiff_t incx, const VectorY& y,
        const std::ptrdiff_t incy, VectorPARAM& param ) {
    rotm_impl< typename value< VectorX >::type >::invoke( n, x, incx, y,
            incy, param );
}

//
// Overloaded function for rotm. Its overload differs for
// * VectorX&
    // * VectorY&
    // * const VectorPARAM&
//
template< typename VectorX, typename VectorY, typename VectorPARAM >
inline typename rotm_impl< typename value< VectorX >::type >::return_type
rotm( const std::ptrdiff_t n, VectorX& x, const std::ptrdiff_t incx,
        VectorY& y, const std::ptrdiff_t incy, const VectorPARAM& param ) {
    rotm_impl< typename value< VectorX >::type >::invoke( n, x, incx, y,
            incy, param );
}

//
// Overloaded function for rotm. Its overload differs for
// * const VectorX&
    // * VectorY&
    // * const VectorPARAM&
//
template< typename VectorX, typename VectorY, typename VectorPARAM >
inline typename rotm_impl< typename value< VectorX >::type >::return_type
rotm( const std::ptrdiff_t n, const VectorX& x,
        const std::ptrdiff_t incx, VectorY& y, const std::ptrdiff_t incy,
        const VectorPARAM& param ) {
    rotm_impl< typename value< VectorX >::type >::invoke( n, x, incx, y,
            incy, param );
}

//
// Overloaded function for rotm. Its overload differs for
// * VectorX&
    // * const VectorY&
    // * const VectorPARAM&
//
template< typename VectorX, typename VectorY, typename VectorPARAM >
inline typename rotm_impl< typename value< VectorX >::type >::return_type
rotm( const std::ptrdiff_t n, VectorX& x, const std::ptrdiff_t incx,
        const VectorY& y, const std::ptrdiff_t incy,
        const VectorPARAM& param ) {
    rotm_impl< typename value< VectorX >::type >::invoke( n, x, incx, y,
            incy, param );
}

//
// Overloaded function for rotm. Its overload differs for
// * const VectorX&
    // * const VectorY&
    // * const VectorPARAM&
//
template< typename VectorX, typename VectorY, typename VectorPARAM >
inline typename rotm_impl< typename value< VectorX >::type >::return_type
rotm( const std::ptrdiff_t n, const VectorX& x,
        const std::ptrdiff_t incx, const VectorY& y,
        const std::ptrdiff_t incy, const VectorPARAM& param ) {
    rotm_impl< typename value< VectorX >::type >::invoke( n, x, incx, y,
            incy, param );
}

} // namespace blas
} // namespace bindings
} // namespace numeric
} // namespace boost

#endif
