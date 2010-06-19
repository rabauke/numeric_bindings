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

#ifndef BOOST_NUMERIC_BINDINGS_LAPACK_DRIVER_GBSV_HPP
#define BOOST_NUMERIC_BINDINGS_LAPACK_DRIVER_GBSV_HPP

#include <boost/assert.hpp>
#include <boost/numeric/bindings/bandwidth.hpp>
#include <boost/numeric/bindings/begin.hpp>
#include <boost/numeric/bindings/is_column_major.hpp>
#include <boost/numeric/bindings/is_mutable.hpp>
#include <boost/numeric/bindings/remove_imaginary.hpp>
#include <boost/numeric/bindings/size.hpp>
#include <boost/numeric/bindings/stride.hpp>
#include <boost/numeric/bindings/value_type.hpp>
#include <boost/static_assert.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/type_traits/remove_const.hpp>

//
// The LAPACK-backend for gbsv is the netlib-compatible backend.
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
inline std::ptrdiff_t gbsv( const fortran_int_t n, const fortran_int_t kl,
        const fortran_int_t ku, const fortran_int_t nrhs, float* ab,
        const fortran_int_t ldab, fortran_int_t* ipiv, float* b,
        const fortran_int_t ldb ) {
    fortran_int_t info(0);
    LAPACK_SGBSV( &n, &kl, &ku, &nrhs, ab, &ldab, ipiv, b, &ldb, &info );
    return info;
}

//
// Overloaded function for dispatching to
// * netlib-compatible LAPACK backend (the default), and
// * double value-type.
//
inline std::ptrdiff_t gbsv( const fortran_int_t n, const fortran_int_t kl,
        const fortran_int_t ku, const fortran_int_t nrhs, double* ab,
        const fortran_int_t ldab, fortran_int_t* ipiv, double* b,
        const fortran_int_t ldb ) {
    fortran_int_t info(0);
    LAPACK_DGBSV( &n, &kl, &ku, &nrhs, ab, &ldab, ipiv, b, &ldb, &info );
    return info;
}

//
// Overloaded function for dispatching to
// * netlib-compatible LAPACK backend (the default), and
// * complex<float> value-type.
//
inline std::ptrdiff_t gbsv( const fortran_int_t n, const fortran_int_t kl,
        const fortran_int_t ku, const fortran_int_t nrhs,
        std::complex<float>* ab, const fortran_int_t ldab,
        fortran_int_t* ipiv, std::complex<float>* b,
        const fortran_int_t ldb ) {
    fortran_int_t info(0);
    LAPACK_CGBSV( &n, &kl, &ku, &nrhs, ab, &ldab, ipiv, b, &ldb, &info );
    return info;
}

//
// Overloaded function for dispatching to
// * netlib-compatible LAPACK backend (the default), and
// * complex<double> value-type.
//
inline std::ptrdiff_t gbsv( const fortran_int_t n, const fortran_int_t kl,
        const fortran_int_t ku, const fortran_int_t nrhs,
        std::complex<double>* ab, const fortran_int_t ldab,
        fortran_int_t* ipiv, std::complex<double>* b,
        const fortran_int_t ldb ) {
    fortran_int_t info(0);
    LAPACK_ZGBSV( &n, &kl, &ku, &nrhs, ab, &ldab, ipiv, b, &ldb, &info );
    return info;
}

} // namespace detail

//
// Value-type based template class. Use this class if you need a type
// for dispatching to gbsv.
//
template< typename Value >
struct gbsv_impl {

    typedef Value value_type;
    typedef typename remove_imaginary< Value >::type real_type;

    //
    // Static member function, that
    // * Deduces the required arguments for dispatching to LAPACK, and
    // * Asserts that most arguments make sense.
    //
    template< typename MatrixAB, typename VectorIPIV, typename MatrixB >
    static std::ptrdiff_t invoke( MatrixAB& ab, VectorIPIV& ipiv, MatrixB& b ) {
        namespace bindings = ::boost::numeric::bindings;
        BOOST_STATIC_ASSERT( (bindings::is_column_major< MatrixAB >::value) );
        BOOST_STATIC_ASSERT( (bindings::is_column_major< MatrixB >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename remove_const<
                typename bindings::value_type< MatrixAB >::type >::type,
                typename remove_const< typename bindings::value_type<
                MatrixB >::type >::type >::value) );
        BOOST_STATIC_ASSERT( (bindings::is_mutable< MatrixAB >::value) );
        BOOST_STATIC_ASSERT( (bindings::is_mutable< VectorIPIV >::value) );
        BOOST_STATIC_ASSERT( (bindings::is_mutable< MatrixB >::value) );
        BOOST_ASSERT( bindings::bandwidth_lower(ab) >= 0 );
        BOOST_ASSERT( bindings::bandwidth_upper(ab) >= 0 );
        BOOST_ASSERT( bindings::size(ipiv) >= bindings::size_column(ab) );
        BOOST_ASSERT( bindings::size_column(ab) >= 0 );
        BOOST_ASSERT( bindings::size_column(b) >= 0 );
        BOOST_ASSERT( bindings::size_minor(ab) == 1 ||
                bindings::stride_minor(ab) == 1 );
        BOOST_ASSERT( bindings::size_minor(b) == 1 ||
                bindings::stride_minor(b) == 1 );
        BOOST_ASSERT( bindings::stride_major(ab) >= 2 );
        BOOST_ASSERT( bindings::stride_major(b) >= std::max< std::ptrdiff_t >(1,
                bindings::size_column(ab)) );
        return detail::gbsv( bindings::size_column(ab),
                bindings::bandwidth_lower(ab), bindings::bandwidth_upper(ab),
                bindings::size_column(b), bindings::begin_value(ab),
                bindings::stride_major(ab), bindings::begin_value(ipiv),
                bindings::begin_value(b), bindings::stride_major(b) );
    }

};


//
// Functions for direct use. These functions are overloaded for temporaries,
// so that wrapped types can still be passed and used for write-access. In
// addition, if applicable, they are overloaded for user-defined workspaces.
// Calls to these functions are passed to the gbsv_impl classes. In the 
// documentation, most overloads are collapsed to avoid a large number of
// prototypes which are very similar.
//

//
// Overloaded function for gbsv. Its overload differs for
// * MatrixAB&
// * MatrixB&
//
template< typename MatrixAB, typename VectorIPIV, typename MatrixB >
inline std::ptrdiff_t gbsv( MatrixAB& ab, VectorIPIV& ipiv, MatrixB& b ) {
    return gbsv_impl< typename bindings::value_type<
            MatrixAB >::type >::invoke( ab, ipiv, b );
}

//
// Overloaded function for gbsv. Its overload differs for
// * const MatrixAB&
// * MatrixB&
//
template< typename MatrixAB, typename VectorIPIV, typename MatrixB >
inline std::ptrdiff_t gbsv( const MatrixAB& ab, VectorIPIV& ipiv,
        MatrixB& b ) {
    return gbsv_impl< typename bindings::value_type<
            MatrixAB >::type >::invoke( ab, ipiv, b );
}

//
// Overloaded function for gbsv. Its overload differs for
// * MatrixAB&
// * const MatrixB&
//
template< typename MatrixAB, typename VectorIPIV, typename MatrixB >
inline std::ptrdiff_t gbsv( MatrixAB& ab, VectorIPIV& ipiv,
        const MatrixB& b ) {
    return gbsv_impl< typename bindings::value_type<
            MatrixAB >::type >::invoke( ab, ipiv, b );
}

//
// Overloaded function for gbsv. Its overload differs for
// * const MatrixAB&
// * const MatrixB&
//
template< typename MatrixAB, typename VectorIPIV, typename MatrixB >
inline std::ptrdiff_t gbsv( const MatrixAB& ab, VectorIPIV& ipiv,
        const MatrixB& b ) {
    return gbsv_impl< typename bindings::value_type<
            MatrixAB >::type >::invoke( ab, ipiv, b );
}

} // namespace lapack
} // namespace bindings
} // namespace numeric
} // namespace boost

#endif
