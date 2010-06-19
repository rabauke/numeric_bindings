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

#ifndef BOOST_NUMERIC_BINDINGS_BLAS_LEVEL1_DOT_HPP
#define BOOST_NUMERIC_BINDINGS_BLAS_LEVEL1_DOT_HPP

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
inline float dot( const int n, const float* x, const int incx, const float* y,
        const int incy ) {
    return cblas_sdot( n, x, incx, y, incy );
}

//
// Overloaded function for dispatching to
// * CBLAS backend, and
// * double value-type.
//
inline double dot( const int n, const double* x, const int incx,
        const double* y, const int incy ) {
    return cblas_ddot( n, x, incx, y, incy );
}

//
// Overloaded function for dispatching to
// * CBLAS backend, and
// * complex<float> value-type.
//
inline std::complex<float> dot( const int n, const std::complex<float>* x,
        const int incx, const std::complex<float>* y, const int incy ) {
    std::complex<float> result;
    cblas_cdotu_sub( n, x, incx, y, incy, &result );
    return result;
}

//
// Overloaded function for dispatching to
// * CBLAS backend, and
// * complex<double> value-type.
//
inline std::complex<double> dot( const int n, const std::complex<double>* x,
        const int incx, const std::complex<double>* y, const int incy ) {
    std::complex<double> result;
    cblas_zdotu_sub( n, x, incx, y, incy, &result );
    return result;
}

#elif defined BOOST_NUMERIC_BINDINGS_BLAS_CUBLAS
//
// Overloaded function for dispatching to
// * CUBLAS backend, and
// * float value-type.
//
inline float dot( const int n, const float* x, const int incx, const float* y,
        const int incy ) {
    return cublasSdot( n, x, incx, y, incy );
}

//
// Overloaded function for dispatching to
// * CUBLAS backend, and
// * double value-type.
//
inline double dot( const int n, const double* x, const int incx,
        const double* y, const int incy ) {
    return cublasDdot( n, x, incx, y, incy );
}

//
// Overloaded function for dispatching to
// * CUBLAS backend, and
// * complex<float> value-type.
//
inline std::complex<float> dot( const int n, const std::complex<float>* x,
        const int incx, const std::complex<float>* y, const int incy ) {
    return cublasCdotu( n, x, incx, y, incy );
}

//
// Overloaded function for dispatching to
// * CUBLAS backend, and
// * complex<double> value-type.
//
inline std::complex<double> dot( const int n, const std::complex<double>* x,
        const int incx, const std::complex<double>* y, const int incy ) {
    return cublasZdotu( n, x, incx, y, incy );
}

#else
//
// Overloaded function for dispatching to
// * netlib-compatible BLAS backend (the default), and
// * float value-type.
//
inline float dot( const fortran_int_t n, const float* x,
        const fortran_int_t incx, const float* y, const fortran_int_t incy ) {
    return BLAS_SDOT( &n, x, &incx, y, &incy );
}

//
// Overloaded function for dispatching to
// * netlib-compatible BLAS backend (the default), and
// * double value-type.
//
inline double dot( const fortran_int_t n, const double* x,
        const fortran_int_t incx, const double* y, const fortran_int_t incy ) {
    return BLAS_DDOT( &n, x, &incx, y, &incy );
}

//
// Overloaded function for dispatching to
// * netlib-compatible BLAS backend (the default), and
// * complex<float> value-type.
//
inline std::complex<float> dot( const fortran_int_t n,
        const std::complex<float>* x, const fortran_int_t incx,
        const std::complex<float>* y, const fortran_int_t incy ) {
    return BLAS_CDOTU( &n, x, &incx, y, &incy );
}

//
// Overloaded function for dispatching to
// * netlib-compatible BLAS backend (the default), and
// * complex<double> value-type.
//
inline std::complex<double> dot( const fortran_int_t n,
        const std::complex<double>* x, const fortran_int_t incx,
        const std::complex<double>* y, const fortran_int_t incy ) {
    return BLAS_ZDOTU( &n, x, &incx, y, &incy );
}

#endif

} // namespace detail

//
// Value-type based template class. Use this class if you need a type
// for dispatching to dot.
//
template< typename Value >
struct dot_impl {

    typedef Value value_type;
    typedef typename remove_imaginary< Value >::type real_type;
    typedef value_type result_type;

    //
    // Static member function that
    // * Deduces the required arguments for dispatching to BLAS, and
    // * Asserts that most arguments make sense.
    //
    template< typename VectorViewX, typename VectorViewY >
    static result_type invoke( const VectorViewX& x, const VectorViewY& y ) {
        namespace bindings = ::boost::numeric::bindings;
        BOOST_STATIC_ASSERT( (is_same< typename remove_const<
                typename bindings::value_type< VectorViewX >::type >::type,
                typename remove_const< typename bindings::value_type<
                VectorViewY >::type >::type >::value) );
        return detail::dot( bindings::size(x),
                bindings::begin_value(x), bindings::stride(x),
                bindings::begin_value(y), bindings::stride(y) );
    }
};

//
// Functions for direct use. These functions are overloaded for temporaries,
// so that wrapped types can still be passed and used for write-access. Calls
// to these functions are passed to the dot_impl classes. In the 
// documentation, the const-overloads are collapsed to avoid a large number of
// prototypes which are very similar.
//

//
// Overloaded function for dot. Its overload differs for
//
template< typename VectorViewX, typename VectorViewY >
inline typename dot_impl< typename bindings::value_type<
        VectorViewX >::type >::result_type
dot( const VectorViewX& x, const VectorViewY& y ) {
    return dot_impl< typename bindings::value_type<
            VectorViewX >::type >::invoke( x, y );
}

} // namespace blas
} // namespace bindings
} // namespace numeric
} // namespace boost

#endif
