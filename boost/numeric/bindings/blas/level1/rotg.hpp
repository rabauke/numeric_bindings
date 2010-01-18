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

#ifndef BOOST_NUMERIC_BINDINGS_BLAS_LEVEL1_ROTG_HPP
#define BOOST_NUMERIC_BINDINGS_BLAS_LEVEL1_ROTG_HPP

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
inline void rotg( float& a, float& b, float& c, float& s ) {
    cblas_srotg( &a, &b, &c, &s );
}

//
// Overloaded function for dispatching to
// * CBLAS backend, and
// * double value-type.
//
inline void rotg( double& a, double& b, double& c, double& s ) {
    cblas_drotg( &a, &b, &c, &s );
}

//
// Overloaded function for dispatching to
// * CBLAS backend, and
// * complex<float> value-type.
//
inline void rotg( std::complex<float>& a, std::complex<float>& b, float& c,
        std::complex<float>& s ) {
    // NOT FOUND();
}

//
// Overloaded function for dispatching to
// * CBLAS backend, and
// * complex<double> value-type.
//
inline void rotg( std::complex<double>& a, std::complex<double>& b, double& c,
        std::complex<double>& s ) {
    // NOT FOUND();
}

#elif defined BOOST_NUMERIC_BINDINGS_BLAS_CUBLAS
//
// Overloaded function for dispatching to
// * CUBLAS backend, and
// * float value-type.
//
inline void rotg( float& a, float& b, float& c, float& s ) {
    cublasSrotg( &a, &b, &c, &s );
}

//
// Overloaded function for dispatching to
// * CUBLAS backend, and
// * double value-type.
//
inline void rotg( double& a, double& b, double& c, double& s ) {
    cublasDrotg( &a, &b, &c, &s );
}

//
// Overloaded function for dispatching to
// * CUBLAS backend, and
// * complex<float> value-type.
//
inline void rotg( std::complex<float>& a, std::complex<float>& b, float& c,
        std::complex<float>& s ) {
    // NOT FOUND();
}

//
// Overloaded function for dispatching to
// * CUBLAS backend, and
// * complex<double> value-type.
//
inline void rotg( std::complex<double>& a, std::complex<double>& b, double& c,
        std::complex<double>& s ) {
    // NOT FOUND();
}

#else
//
// Overloaded function for dispatching to
// * netlib-compatible BLAS backend (the default), and
// * float value-type.
//
inline void rotg( float& a, float& b, float& c, float& s ) {
    BLAS_SROTG( &a, &b, &c, &s );
}

//
// Overloaded function for dispatching to
// * netlib-compatible BLAS backend (the default), and
// * double value-type.
//
inline void rotg( double& a, double& b, double& c, double& s ) {
    BLAS_DROTG( &a, &b, &c, &s );
}

//
// Overloaded function for dispatching to
// * netlib-compatible BLAS backend (the default), and
// * complex<float> value-type.
//
inline void rotg( std::complex<float>& a, std::complex<float>& b, float& c,
        std::complex<float>& s ) {
    BLAS_CROTG( &a, &b, &c, &s );
}

//
// Overloaded function for dispatching to
// * netlib-compatible BLAS backend (the default), and
// * complex<double> value-type.
//
inline void rotg( std::complex<double>& a, std::complex<double>& b, double& c,
        std::complex<double>& s ) {
    BLAS_ZROTG( &a, &b, &c, &s );
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
    static return_type invoke( value_type& a, value_type& b, real_type& c,
            value_type& s ) {
        namespace bindings = ::boost::numeric::bindings;
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
//
template< typename Value >
inline typename rotg_impl< Value >::return_type
rotg( Value& a, Value& b, Value& c, Value& s ) {
    rotg_impl< Value >::invoke( a, b, c, s );
}

} // namespace blas
} // namespace bindings
} // namespace numeric
} // namespace boost

#endif
