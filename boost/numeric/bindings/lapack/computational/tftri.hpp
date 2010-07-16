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

#ifndef BOOST_NUMERIC_BINDINGS_LAPACK_COMPUTATIONAL_TFTRI_HPP
#define BOOST_NUMERIC_BINDINGS_LAPACK_COMPUTATIONAL_TFTRI_HPP

#include <boost/assert.hpp>
#include <boost/numeric/bindings/begin.hpp>
#include <boost/numeric/bindings/blas/detail/default_order.hpp>
#include <boost/numeric/bindings/diag_tag.hpp>
#include <boost/numeric/bindings/is_mutable.hpp>
#include <boost/numeric/bindings/remove_imaginary.hpp>
#include <boost/numeric/bindings/size.hpp>
#include <boost/numeric/bindings/stride.hpp>
#include <boost/numeric/bindings/trans_tag.hpp>
#include <boost/numeric/bindings/uplo_tag.hpp>
#include <boost/numeric/bindings/value_type.hpp>
#include <boost/static_assert.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/type_traits/remove_const.hpp>

//
// The LAPACK-backend for tftri is the netlib-compatible backend.
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
template< typename TransR, typename UpLo, typename Diag >
inline std::ptrdiff_t tftri( const TransR transr, const UpLo uplo,
        const Diag diag, const fortran_int_t n, float* a ) {
    fortran_int_t info(0);
    LAPACK_STFTRI( &lapack_option< TransR >::value, &lapack_option<
            UpLo >::value, &lapack_option< Diag >::value, &n, a, &info );
    return info;
}

//
// Overloaded function for dispatching to
// * netlib-compatible LAPACK backend (the default), and
// * double value-type.
//
template< typename TransR, typename UpLo, typename Diag >
inline std::ptrdiff_t tftri( const TransR transr, const UpLo uplo,
        const Diag diag, const fortran_int_t n, double* a ) {
    fortran_int_t info(0);
    LAPACK_DTFTRI( &lapack_option< TransR >::value, &lapack_option<
            UpLo >::value, &lapack_option< Diag >::value, &n, a, &info );
    return info;
}

//
// Overloaded function for dispatching to
// * netlib-compatible LAPACK backend (the default), and
// * complex<float> value-type.
//
template< typename TransR, typename UpLo, typename Diag >
inline std::ptrdiff_t tftri( const TransR transr, const UpLo uplo,
        const Diag diag, const fortran_int_t n, std::complex<float>* a ) {
    fortran_int_t info(0);
    LAPACK_CTFTRI( &lapack_option< TransR >::value, &lapack_option<
            UpLo >::value, &lapack_option< Diag >::value, &n, a, &info );
    return info;
}

//
// Overloaded function for dispatching to
// * netlib-compatible LAPACK backend (the default), and
// * complex<double> value-type.
//
template< typename TransR, typename UpLo, typename Diag >
inline std::ptrdiff_t tftri( const TransR transr, const UpLo uplo,
        const Diag diag, const fortran_int_t n, std::complex<double>* a ) {
    fortran_int_t info(0);
    LAPACK_ZTFTRI( &lapack_option< TransR >::value, &lapack_option<
            UpLo >::value, &lapack_option< Diag >::value, &n, a, &info );
    return info;
}

} // namespace detail

//
// Value-type based template class. Use this class if you need a type
// for dispatching to tftri.
//
template< typename Value >
struct tftri_impl {

    typedef Value value_type;
    typedef typename remove_imaginary< Value >::type real_type;

    //
    // Static member function, that
    // * Deduces the required arguments for dispatching to LAPACK, and
    // * Asserts that most arguments make sense.
    //
    template< typename MatrixA >
    static std::ptrdiff_t invoke( MatrixA& a ) {
        namespace bindings = ::boost::numeric::bindings;
        typedef typename blas::detail::default_order< MatrixA >::type order;
        typedef typename result_of::trans_tag< MatrixA, order >::type transr;
        typedef typename result_of::uplo_tag< MatrixA, transr >::type uplo;
        typedef typename result_of::diag_tag< MatrixA >::type diag;
        BOOST_STATIC_ASSERT( (bindings::is_mutable< MatrixA >::value) );
        BOOST_ASSERT( bindings::size_column_op(a, transr()) >= 0 );
        return detail::tftri( transr(), uplo(), diag(),
                bindings::size_column_op(a, transr()),
                bindings::begin_value(a) );
    }

};


//
// Functions for direct use. These functions are overloaded for temporaries,
// so that wrapped types can still be passed and used for write-access. In
// addition, if applicable, they are overloaded for user-defined workspaces.
// Calls to these functions are passed to the tftri_impl classes. In the 
// documentation, most overloads are collapsed to avoid a large number of
// prototypes which are very similar.
//

//
// Overloaded function for tftri. Its overload differs for
// * MatrixA&
//
template< typename MatrixA >
inline std::ptrdiff_t tftri( MatrixA& a ) {
    return tftri_impl< typename bindings::value_type<
            MatrixA >::type >::invoke( a );
}

//
// Overloaded function for tftri. Its overload differs for
// * const MatrixA&
//
template< typename MatrixA >
inline std::ptrdiff_t tftri( const MatrixA& a ) {
    return tftri_impl< typename bindings::value_type<
            MatrixA >::type >::invoke( a );
}

} // namespace lapack
} // namespace bindings
} // namespace numeric
} // namespace boost

#endif
