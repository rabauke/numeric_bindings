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

#ifndef BOOST_NUMERIC_BINDINGS_LAPACK_COMPUTATIONAL_TPTRI_HPP
#define BOOST_NUMERIC_BINDINGS_LAPACK_COMPUTATIONAL_TPTRI_HPP

#include <boost/assert.hpp>
#include <boost/numeric/bindings/begin.hpp>
#include <boost/numeric/bindings/data_side.hpp>
#include <boost/numeric/bindings/diag_tag.hpp>
#include <boost/numeric/bindings/is_mutable.hpp>
#include <boost/numeric/bindings/remove_imaginary.hpp>
#include <boost/numeric/bindings/size.hpp>
#include <boost/numeric/bindings/stride.hpp>
#include <boost/numeric/bindings/value.hpp>
#include <boost/static_assert.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/type_traits/remove_const.hpp>

//
// The LAPACK-backend for tptri is the netlib-compatible backend.
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
template< typename UpLo, typename Diag >
inline std::ptrdiff_t tptri( UpLo, Diag, fortran_int_t n, float* ap ) {
    fortran_int_t info(0);
    LAPACK_STPTRI( &lapack_option< UpLo >::value, &lapack_option<
            Diag >::value, &n, ap, &info );
    return info;
}

//
// Overloaded function for dispatching to
// * netlib-compatible LAPACK backend (the default), and
// * double value-type.
//
template< typename UpLo, typename Diag >
inline std::ptrdiff_t tptri( UpLo, Diag, fortran_int_t n, double* ap ) {
    fortran_int_t info(0);
    LAPACK_DTPTRI( &lapack_option< UpLo >::value, &lapack_option<
            Diag >::value, &n, ap, &info );
    return info;
}

//
// Overloaded function for dispatching to
// * netlib-compatible LAPACK backend (the default), and
// * complex<float> value-type.
//
template< typename UpLo, typename Diag >
inline std::ptrdiff_t tptri( UpLo, Diag, fortran_int_t n,
        std::complex<float>* ap ) {
    fortran_int_t info(0);
    LAPACK_CTPTRI( &lapack_option< UpLo >::value, &lapack_option<
            Diag >::value, &n, ap, &info );
    return info;
}

//
// Overloaded function for dispatching to
// * netlib-compatible LAPACK backend (the default), and
// * complex<double> value-type.
//
template< typename UpLo, typename Diag >
inline std::ptrdiff_t tptri( UpLo, Diag, fortran_int_t n,
        std::complex<double>* ap ) {
    fortran_int_t info(0);
    LAPACK_ZTPTRI( &lapack_option< UpLo >::value, &lapack_option<
            Diag >::value, &n, ap, &info );
    return info;
}

} // namespace detail

//
// Value-type based template class. Use this class if you need a type
// for dispatching to tptri.
//
template< typename Value >
struct tptri_impl {

    typedef Value value_type;
    typedef typename remove_imaginary< Value >::type real_type;
    typedef tag::column_major order;

    //
    // Static member function, that
    // * Deduces the required arguments for dispatching to LAPACK, and
    // * Asserts that most arguments make sense.
    //
    template< typename MatrixAP >
    static std::ptrdiff_t invoke( MatrixAP& ap ) {
        typedef typename result_of::data_side< MatrixAP >::type uplo;
        typedef typename result_of::diag_tag< MatrixAP >::type diag;
        BOOST_STATIC_ASSERT( (is_mutable< MatrixAP >::value) );
        BOOST_ASSERT( size_column(ap) >= 0 );
        return detail::tptri( uplo(), diag(), size_column(ap),
                begin_value(ap) );
    }

};


//
// Functions for direct use. These functions are overloaded for temporaries,
// so that wrapped types can still be passed and used for write-access. In
// addition, if applicable, they are overloaded for user-defined workspaces.
// Calls to these functions are passed to the tptri_impl classes. In the 
// documentation, most overloads are collapsed to avoid a large number of
// prototypes which are very similar.
//

//
// Overloaded function for tptri. Its overload differs for
// * MatrixAP&
//
template< typename MatrixAP >
inline std::ptrdiff_t tptri( MatrixAP& ap ) {
    return tptri_impl< typename value< MatrixAP >::type >::invoke( ap );
}

//
// Overloaded function for tptri. Its overload differs for
// * const MatrixAP&
//
template< typename MatrixAP >
inline std::ptrdiff_t tptri( const MatrixAP& ap ) {
    return tptri_impl< typename value< MatrixAP >::type >::invoke( ap );
}

} // namespace lapack
} // namespace bindings
} // namespace numeric
} // namespace boost

#endif
