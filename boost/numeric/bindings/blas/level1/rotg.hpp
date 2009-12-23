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

#ifndef BOOST_NUMERIC_BINDINGS_BLAS_LEVEL1_ROTG_HPP
#define BOOST_NUMERIC_BINDINGS_BLAS_LEVEL1_ROTG_HPP

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
inline void rotg( Order, float& a, float& b, float& c, float& s ) {
    cblas_srotg( cblas_option< Order >::value, &a, &b, &c, &s );
}

//
// Overloaded function for dispatching to
// * CBLAS backend
// * double value-type
//
template< typename Order >
inline void rotg( Order, double& a, double& b, double& c, double& s ) {
    cblas_drotg( cblas_option< Order >::value, &a, &b, &c, &s );
}

#elif defined BOOST_NUMERIC_BINDINGS_BLAS_CUBLAS
//
// Overloaded function for dispatching to
// * CUBLAS backend
// * float value-type
//
template< typename Order >
inline void rotg( Order, float& a, float& b, float& c, float& s ) {
    BOOST_STATIC_ASSERT( (is_column_major<Order>::value) );
    cublasSrotg( &a, &b, &c, &s );
}

//
// Overloaded function for dispatching to
// * CUBLAS backend
// * double value-type
//
template< typename Order >
inline void rotg( Order, double& a, double& b, double& c, double& s ) {
    BOOST_STATIC_ASSERT( (is_column_major<Order>::value) );
    cublasDrotg( &a, &b, &c, &s );
}

#else
//
// Overloaded function for dispatching to
// * netlib-compatible BLAS backend (the default)
// * float value-type
//
template< typename Order >
inline void rotg( Order, float& a, float& b, float& c, float& s ) {
    BOOST_STATIC_ASSERT( (is_column_major<Order>::value) );
    BLAS_SROTG( &a, &b, &c, &s );
}

//
// Overloaded function for dispatching to
// * netlib-compatible BLAS backend (the default)
// * double value-type
//
template< typename Order >
inline void rotg( Order, double& a, double& b, double& c, double& s ) {
    BOOST_STATIC_ASSERT( (is_column_major<Order>::value) );
    BLAS_DROTG( &a, &b, &c, &s );
}

#endif

} // namespace detail

//
// Value-type based template class. Use this class if you need a type
// for dispatching to rotg.
//
template< typename Value >
struct rotg_impl {

    typedef Value value_type;
    typedef typename remove_imaginary< Value >::type real_type;
    typedef void return_type;

    //
    // Static member function that
    // * Deduces the required arguments for dispatching to BLAS, and
    // * Asserts that most arguments make sense.
    //
    template<  >
    static return_type invoke( real_type& a, real_type& b, real_type& c,
            real_type& s ) {
        
        detail::rotg( a, b, c, s );
    }
};

//
// Functions for direct use. These functions are overloaded for temporaries,
// so that wrapped types can still be passed and used for write-access. Calls
// to these functions are passed to the rotg_impl classes. In the 
// documentation, the const-overloads are collapsed to avoid a large number of
// prototypes which are very similar.
//

//
// Overloaded function for rotg. Its overload differs for
// * typename remove_imaginary< typename value< TODO >::type >::type&
    // * typename remove_imaginary< typename value< TODO >::type >::type&
    // * typename remove_imaginary< typename value< TODO >::type >::type&
    // * typename remove_imaginary< typename value< TODO >::type >::type&
//
template<  >
inline typename rotg_impl< typename value< TODO >::type >::return_type
rotg( typename remove_imaginary< typename value<
        TODO >::type >::type& a, typename remove_imaginary< typename value<
        TODO >::type >::type& b, typename remove_imaginary< typename value<
        TODO >::type >::type& c, typename remove_imaginary< typename value<
        TODO >::type >::type& s ) {
    rotg_impl< typename value< TODO >::type >::invoke( a, b, c, s );
}

//
// Overloaded function for rotg. Its overload differs for
// * const typename remove_imaginary< typename value< TODO >::type >::type&
    // * typename remove_imaginary< typename value< TODO >::type >::type&
    // * typename remove_imaginary< typename value< TODO >::type >::type&
    // * typename remove_imaginary< typename value< TODO >::type >::type&
//
template<  >
inline typename rotg_impl< typename value< TODO >::type >::return_type
rotg( const typename remove_imaginary< typename value<
        TODO >::type >::type& a, typename remove_imaginary< typename value<
        TODO >::type >::type& b, typename remove_imaginary< typename value<
        TODO >::type >::type& c, typename remove_imaginary< typename value<
        TODO >::type >::type& s ) {
    rotg_impl< typename value< TODO >::type >::invoke( a, b, c, s );
}

//
// Overloaded function for rotg. Its overload differs for
// * typename remove_imaginary< typename value< TODO >::type >::type&
    // * const typename remove_imaginary< typename value< TODO >::type >::type&
    // * typename remove_imaginary< typename value< TODO >::type >::type&
    // * typename remove_imaginary< typename value< TODO >::type >::type&
//
template<  >
inline typename rotg_impl< typename value< TODO >::type >::return_type
rotg( typename remove_imaginary< typename value<
        TODO >::type >::type& a, const typename remove_imaginary<
        typename value< TODO >::type >::type& b, typename remove_imaginary<
        typename value< TODO >::type >::type& c, typename remove_imaginary<
        typename value< TODO >::type >::type& s ) {
    rotg_impl< typename value< TODO >::type >::invoke( a, b, c, s );
}

//
// Overloaded function for rotg. Its overload differs for
// * const typename remove_imaginary< typename value< TODO >::type >::type&
    // * const typename remove_imaginary< typename value< TODO >::type >::type&
    // * typename remove_imaginary< typename value< TODO >::type >::type&
    // * typename remove_imaginary< typename value< TODO >::type >::type&
//
template<  >
inline typename rotg_impl< typename value< TODO >::type >::return_type
rotg( const typename remove_imaginary< typename value<
        TODO >::type >::type& a, const typename remove_imaginary<
        typename value< TODO >::type >::type& b, typename remove_imaginary<
        typename value< TODO >::type >::type& c, typename remove_imaginary<
        typename value< TODO >::type >::type& s ) {
    rotg_impl< typename value< TODO >::type >::invoke( a, b, c, s );
}

//
// Overloaded function for rotg. Its overload differs for
// * typename remove_imaginary< typename value< TODO >::type >::type&
    // * typename remove_imaginary< typename value< TODO >::type >::type&
    // * const typename remove_imaginary< typename value< TODO >::type >::type&
    // * typename remove_imaginary< typename value< TODO >::type >::type&
//
template<  >
inline typename rotg_impl< typename value< TODO >::type >::return_type
rotg( typename remove_imaginary< typename value<
        TODO >::type >::type& a, typename remove_imaginary< typename value<
        TODO >::type >::type& b, const typename remove_imaginary<
        typename value< TODO >::type >::type& c, typename remove_imaginary<
        typename value< TODO >::type >::type& s ) {
    rotg_impl< typename value< TODO >::type >::invoke( a, b, c, s );
}

//
// Overloaded function for rotg. Its overload differs for
// * const typename remove_imaginary< typename value< TODO >::type >::type&
    // * typename remove_imaginary< typename value< TODO >::type >::type&
    // * const typename remove_imaginary< typename value< TODO >::type >::type&
    // * typename remove_imaginary< typename value< TODO >::type >::type&
//
template<  >
inline typename rotg_impl< typename value< TODO >::type >::return_type
rotg( const typename remove_imaginary< typename value<
        TODO >::type >::type& a, typename remove_imaginary< typename value<
        TODO >::type >::type& b, const typename remove_imaginary<
        typename value< TODO >::type >::type& c, typename remove_imaginary<
        typename value< TODO >::type >::type& s ) {
    rotg_impl< typename value< TODO >::type >::invoke( a, b, c, s );
}

//
// Overloaded function for rotg. Its overload differs for
// * typename remove_imaginary< typename value< TODO >::type >::type&
    // * const typename remove_imaginary< typename value< TODO >::type >::type&
    // * const typename remove_imaginary< typename value< TODO >::type >::type&
    // * typename remove_imaginary< typename value< TODO >::type >::type&
//
template<  >
inline typename rotg_impl< typename value< TODO >::type >::return_type
rotg( typename remove_imaginary< typename value<
        TODO >::type >::type& a, const typename remove_imaginary<
        typename value< TODO >::type >::type& b,
        const typename remove_imaginary< typename value<
        TODO >::type >::type& c, typename remove_imaginary< typename value<
        TODO >::type >::type& s ) {
    rotg_impl< typename value< TODO >::type >::invoke( a, b, c, s );
}

//
// Overloaded function for rotg. Its overload differs for
// * const typename remove_imaginary< typename value< TODO >::type >::type&
    // * const typename remove_imaginary< typename value< TODO >::type >::type&
    // * const typename remove_imaginary< typename value< TODO >::type >::type&
    // * typename remove_imaginary< typename value< TODO >::type >::type&
//
template<  >
inline typename rotg_impl< typename value< TODO >::type >::return_type
rotg( const typename remove_imaginary< typename value<
        TODO >::type >::type& a, const typename remove_imaginary<
        typename value< TODO >::type >::type& b,
        const typename remove_imaginary< typename value<
        TODO >::type >::type& c, typename remove_imaginary< typename value<
        TODO >::type >::type& s ) {
    rotg_impl< typename value< TODO >::type >::invoke( a, b, c, s );
}

//
// Overloaded function for rotg. Its overload differs for
// * typename remove_imaginary< typename value< TODO >::type >::type&
    // * typename remove_imaginary< typename value< TODO >::type >::type&
    // * typename remove_imaginary< typename value< TODO >::type >::type&
    // * const typename remove_imaginary< typename value< TODO >::type >::type&
//
template<  >
inline typename rotg_impl< typename value< TODO >::type >::return_type
rotg( typename remove_imaginary< typename value<
        TODO >::type >::type& a, typename remove_imaginary< typename value<
        TODO >::type >::type& b, typename remove_imaginary< typename value<
        TODO >::type >::type& c, const typename remove_imaginary<
        typename value< TODO >::type >::type& s ) {
    rotg_impl< typename value< TODO >::type >::invoke( a, b, c, s );
}

//
// Overloaded function for rotg. Its overload differs for
// * const typename remove_imaginary< typename value< TODO >::type >::type&
    // * typename remove_imaginary< typename value< TODO >::type >::type&
    // * typename remove_imaginary< typename value< TODO >::type >::type&
    // * const typename remove_imaginary< typename value< TODO >::type >::type&
//
template<  >
inline typename rotg_impl< typename value< TODO >::type >::return_type
rotg( const typename remove_imaginary< typename value<
        TODO >::type >::type& a, typename remove_imaginary< typename value<
        TODO >::type >::type& b, typename remove_imaginary< typename value<
        TODO >::type >::type& c, const typename remove_imaginary<
        typename value< TODO >::type >::type& s ) {
    rotg_impl< typename value< TODO >::type >::invoke( a, b, c, s );
}

//
// Overloaded function for rotg. Its overload differs for
// * typename remove_imaginary< typename value< TODO >::type >::type&
    // * const typename remove_imaginary< typename value< TODO >::type >::type&
    // * typename remove_imaginary< typename value< TODO >::type >::type&
    // * const typename remove_imaginary< typename value< TODO >::type >::type&
//
template<  >
inline typename rotg_impl< typename value< TODO >::type >::return_type
rotg( typename remove_imaginary< typename value<
        TODO >::type >::type& a, const typename remove_imaginary<
        typename value< TODO >::type >::type& b, typename remove_imaginary<
        typename value< TODO >::type >::type& c,
        const typename remove_imaginary< typename value<
        TODO >::type >::type& s ) {
    rotg_impl< typename value< TODO >::type >::invoke( a, b, c, s );
}

//
// Overloaded function for rotg. Its overload differs for
// * const typename remove_imaginary< typename value< TODO >::type >::type&
    // * const typename remove_imaginary< typename value< TODO >::type >::type&
    // * typename remove_imaginary< typename value< TODO >::type >::type&
    // * const typename remove_imaginary< typename value< TODO >::type >::type&
//
template<  >
inline typename rotg_impl< typename value< TODO >::type >::return_type
rotg( const typename remove_imaginary< typename value<
        TODO >::type >::type& a, const typename remove_imaginary<
        typename value< TODO >::type >::type& b, typename remove_imaginary<
        typename value< TODO >::type >::type& c,
        const typename remove_imaginary< typename value<
        TODO >::type >::type& s ) {
    rotg_impl< typename value< TODO >::type >::invoke( a, b, c, s );
}

//
// Overloaded function for rotg. Its overload differs for
// * typename remove_imaginary< typename value< TODO >::type >::type&
    // * typename remove_imaginary< typename value< TODO >::type >::type&
    // * const typename remove_imaginary< typename value< TODO >::type >::type&
    // * const typename remove_imaginary< typename value< TODO >::type >::type&
//
template<  >
inline typename rotg_impl< typename value< TODO >::type >::return_type
rotg( typename remove_imaginary< typename value<
        TODO >::type >::type& a, typename remove_imaginary< typename value<
        TODO >::type >::type& b, const typename remove_imaginary<
        typename value< TODO >::type >::type& c,
        const typename remove_imaginary< typename value<
        TODO >::type >::type& s ) {
    rotg_impl< typename value< TODO >::type >::invoke( a, b, c, s );
}

//
// Overloaded function for rotg. Its overload differs for
// * const typename remove_imaginary< typename value< TODO >::type >::type&
    // * typename remove_imaginary< typename value< TODO >::type >::type&
    // * const typename remove_imaginary< typename value< TODO >::type >::type&
    // * const typename remove_imaginary< typename value< TODO >::type >::type&
//
template<  >
inline typename rotg_impl< typename value< TODO >::type >::return_type
rotg( const typename remove_imaginary< typename value<
        TODO >::type >::type& a, typename remove_imaginary< typename value<
        TODO >::type >::type& b, const typename remove_imaginary<
        typename value< TODO >::type >::type& c,
        const typename remove_imaginary< typename value<
        TODO >::type >::type& s ) {
    rotg_impl< typename value< TODO >::type >::invoke( a, b, c, s );
}

//
// Overloaded function for rotg. Its overload differs for
// * typename remove_imaginary< typename value< TODO >::type >::type&
    // * const typename remove_imaginary< typename value< TODO >::type >::type&
    // * const typename remove_imaginary< typename value< TODO >::type >::type&
    // * const typename remove_imaginary< typename value< TODO >::type >::type&
//
template<  >
inline typename rotg_impl< typename value< TODO >::type >::return_type
rotg( typename remove_imaginary< typename value<
        TODO >::type >::type& a, const typename remove_imaginary<
        typename value< TODO >::type >::type& b,
        const typename remove_imaginary< typename value<
        TODO >::type >::type& c, const typename remove_imaginary<
        typename value< TODO >::type >::type& s ) {
    rotg_impl< typename value< TODO >::type >::invoke( a, b, c, s );
}

//
// Overloaded function for rotg. Its overload differs for
// * const typename remove_imaginary< typename value< TODO >::type >::type&
    // * const typename remove_imaginary< typename value< TODO >::type >::type&
    // * const typename remove_imaginary< typename value< TODO >::type >::type&
    // * const typename remove_imaginary< typename value< TODO >::type >::type&
//
template<  >
inline typename rotg_impl< typename value< TODO >::type >::return_type
rotg( const typename remove_imaginary< typename value<
        TODO >::type >::type& a, const typename remove_imaginary<
        typename value< TODO >::type >::type& b,
        const typename remove_imaginary< typename value<
        TODO >::type >::type& c, const typename remove_imaginary<
        typename value< TODO >::type >::type& s ) {
    rotg_impl< typename value< TODO >::type >::invoke( a, b, c, s );
}

} // namespace blas
} // namespace bindings
} // namespace numeric
} // namespace boost

#endif
