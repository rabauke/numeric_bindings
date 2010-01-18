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

#ifndef BOOST_NUMERIC_BINDINGS_LAPACK_COMPUTATIONAL_PTTRS_HPP
#define BOOST_NUMERIC_BINDINGS_LAPACK_COMPUTATIONAL_PTTRS_HPP

#include <boost/assert.hpp>
#include <boost/numeric/bindings/begin.hpp>
#include <boost/numeric/bindings/is_complex.hpp>
#include <boost/numeric/bindings/is_mutable.hpp>
#include <boost/numeric/bindings/is_real.hpp>
#include <boost/numeric/bindings/remove_imaginary.hpp>
#include <boost/numeric/bindings/size.hpp>
#include <boost/numeric/bindings/stride.hpp>
#include <boost/numeric/bindings/uplo_tag.hpp>
#include <boost/numeric/bindings/value_type.hpp>
#include <boost/static_assert.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/type_traits/remove_const.hpp>
#include <boost/utility/enable_if.hpp>

//
// The LAPACK-backend for pttrs is the netlib-compatible backend.
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
inline std::ptrdiff_t pttrs( const fortran_int_t n, const fortran_int_t nrhs,
        const float* d, const float* e, float* b, const fortran_int_t ldb ) {
    fortran_int_t info(0);
    LAPACK_SPTTRS( &n, &nrhs, d, e, b, &ldb, &info );
    return info;
}

//
// Overloaded function for dispatching to
// * netlib-compatible LAPACK backend (the default), and
// * double value-type.
//
inline std::ptrdiff_t pttrs( const fortran_int_t n, const fortran_int_t nrhs,
        const double* d, const double* e, double* b,
        const fortran_int_t ldb ) {
    fortran_int_t info(0);
    LAPACK_DPTTRS( &n, &nrhs, d, e, b, &ldb, &info );
    return info;
}

//
// Overloaded function for dispatching to
// * netlib-compatible LAPACK backend (the default), and
// * complex<float> value-type.
//
inline std::ptrdiff_t pttrs( const char uplo, const fortran_int_t n,
        const fortran_int_t nrhs, const float* d,
        const std::complex<float>* e, std::complex<float>* b,
        const fortran_int_t ldb ) {
    fortran_int_t info(0);
    LAPACK_CPTTRS( &uplo, &n, &nrhs, d, e, b, &ldb, &info );
    return info;
}

//
// Overloaded function for dispatching to
// * netlib-compatible LAPACK backend (the default), and
// * complex<double> value-type.
//
inline std::ptrdiff_t pttrs( const char uplo, const fortran_int_t n,
        const fortran_int_t nrhs, const double* d,
        const std::complex<double>* e, std::complex<double>* b,
        const fortran_int_t ldb ) {
    fortran_int_t info(0);
    LAPACK_ZPTTRS( &uplo, &n, &nrhs, d, e, b, &ldb, &info );
    return info;
}

} // namespace detail

//
// Value-type based template class. Use this class if you need a type
// for dispatching to pttrs.
//
template< typename Value, typename Enable = void >
struct pttrs_impl {};

//
// This implementation is enabled if Value is a real type.
//
template< typename Value >
struct pttrs_impl< Value, typename boost::enable_if< is_real< Value > >::type > {

    typedef Value value_type;
    typedef typename remove_imaginary< Value >::type real_type;
    typedef tag::column_major order;

    //
    // Static member function, that
    // * Deduces the required arguments for dispatching to LAPACK, and
    // * Asserts that most arguments make sense.
    //
    template< typename VectorD, typename VectorE, typename MatrixB >
    static std::ptrdiff_t invoke( const VectorD& d, const VectorE& e,
            MatrixB& b ) {
        namespace bindings = ::boost::numeric::bindings;
        BOOST_STATIC_ASSERT( (boost::is_same< typename remove_const<
                typename bindings::value_type< VectorD >::type >::type,
                typename remove_const< typename bindings::value_type<
                VectorE >::type >::type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename remove_const<
                typename bindings::value_type< VectorD >::type >::type,
                typename remove_const< typename bindings::value_type<
                MatrixB >::type >::type >::value) );
        BOOST_STATIC_ASSERT( (bindings::is_mutable< MatrixB >::value) );
        BOOST_ASSERT( bindings::size(d) >= bindings::size(d) );
        BOOST_ASSERT( bindings::size(d) >= 0 );
        BOOST_ASSERT( bindings::size(e) >= bindings::size(d)-1 );
        BOOST_ASSERT( bindings::size_column(b) >= 0 );
        BOOST_ASSERT( bindings::size_minor(b) == 1 ||
                bindings::stride_minor(b) == 1 );
        BOOST_ASSERT( bindings::stride_major(b) >= std::max< std::ptrdiff_t >(1,
                bindings::size(d)) );
        return detail::pttrs( bindings::size(d), bindings::size_column(b),
                bindings::begin_value(d), bindings::begin_value(e),
                bindings::begin_value(b), bindings::stride_major(b) );
    }

};

//
// This implementation is enabled if Value is a complex type.
//
template< typename Value >
struct pttrs_impl< Value, typename boost::enable_if< is_complex< Value > >::type > {

    typedef Value value_type;
    typedef typename remove_imaginary< Value >::type real_type;
    typedef tag::column_major order;

    //
    // Static member function, that
    // * Deduces the required arguments for dispatching to LAPACK, and
    // * Asserts that most arguments make sense.
    //
    template< typename VectorD, typename VectorE, typename MatrixB >
    static std::ptrdiff_t invoke( const char uplo, const VectorD& d,
            const VectorE& e, MatrixB& b ) {
        namespace bindings = ::boost::numeric::bindings;
        BOOST_STATIC_ASSERT( (boost::is_same< typename remove_const<
                typename bindings::value_type< VectorE >::type >::type,
                typename remove_const< typename bindings::value_type<
                MatrixB >::type >::type >::value) );
        BOOST_STATIC_ASSERT( (bindings::is_mutable< MatrixB >::value) );
        BOOST_ASSERT( bindings::size(d) >= bindings::size(d) );
        BOOST_ASSERT( bindings::size(d) >= 0 );
        BOOST_ASSERT( bindings::size_column(b) >= 0 );
        BOOST_ASSERT( bindings::size_minor(b) == 1 ||
                bindings::stride_minor(b) == 1 );
        BOOST_ASSERT( bindings::stride_major(b) >= std::max< std::ptrdiff_t >(1,
                bindings::size(d)) );
        return detail::pttrs( uplo, bindings::size(d),
                bindings::size_column(b), bindings::begin_value(d),
                bindings::begin_value(e), bindings::begin_value(b),
                bindings::stride_major(b) );
    }

};


//
// Functions for direct use. These functions are overloaded for temporaries,
// so that wrapped types can still be passed and used for write-access. In
// addition, if applicable, they are overloaded for user-defined workspaces.
// Calls to these functions are passed to the pttrs_impl classes. In the 
// documentation, most overloads are collapsed to avoid a large number of
// prototypes which are very similar.
//

//
// Overloaded function for pttrs. Its overload differs for
// * MatrixB&
//
template< typename VectorD, typename VectorE, typename MatrixB >
inline std::ptrdiff_t pttrs( const VectorD& d, const VectorE& e,
        MatrixB& b ) {
    return pttrs_impl< typename bindings::value_type<
            VectorE >::type >::invoke( d, e, b );
}

//
// Overloaded function for pttrs. Its overload differs for
// * const MatrixB&
//
template< typename VectorD, typename VectorE, typename MatrixB >
inline std::ptrdiff_t pttrs( const VectorD& d, const VectorE& e,
        const MatrixB& b ) {
    return pttrs_impl< typename bindings::value_type<
            VectorE >::type >::invoke( d, e, b );
}
//
// Overloaded function for pttrs. Its overload differs for
// * MatrixB&
//
template< typename VectorD, typename VectorE, typename MatrixB >
inline std::ptrdiff_t pttrs( const char uplo, const VectorD& d,
        const VectorE& e, MatrixB& b ) {
    return pttrs_impl< typename bindings::value_type<
            VectorE >::type >::invoke( uplo, d, e, b );
}

//
// Overloaded function for pttrs. Its overload differs for
// * const MatrixB&
//
template< typename VectorD, typename VectorE, typename MatrixB >
inline std::ptrdiff_t pttrs( const char uplo, const VectorD& d,
        const VectorE& e, const MatrixB& b ) {
    return pttrs_impl< typename bindings::value_type<
            VectorE >::type >::invoke( uplo, d, e, b );
}

} // namespace lapack
} // namespace bindings
} // namespace numeric
} // namespace boost

#endif
