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

#ifndef BOOST_NUMERIC_BINDINGS_LAPACK_COMPUTATIONAL_HETRI_HPP
#define BOOST_NUMERIC_BINDINGS_LAPACK_COMPUTATIONAL_HETRI_HPP

#include <boost/assert.hpp>
#include <boost/numeric/bindings/begin.hpp>
#include <boost/numeric/bindings/data_side.hpp>
#include <boost/numeric/bindings/detail/array.hpp>
#include <boost/numeric/bindings/is_mutable.hpp>
#include <boost/numeric/bindings/lapack/workspace.hpp>
#include <boost/numeric/bindings/remove_imaginary.hpp>
#include <boost/numeric/bindings/size.hpp>
#include <boost/numeric/bindings/stride.hpp>
#include <boost/numeric/bindings/value.hpp>
#include <boost/static_assert.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/type_traits/remove_const.hpp>

//
// The LAPACK-backend for hetri is the netlib-compatible backend.
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
// * complex<float> value-type.
//
inline std::ptrdiff_t hetri( char uplo, fortran_int_t n,
        std::complex<float>* a, fortran_int_t lda, const fortran_int_t* ipiv,
        std::complex<float>* work ) {
    fortran_int_t info(0);
    LAPACK_CHETRI( &uplo, &n, a, &lda, ipiv, work, &info );
    return info;
}

//
// Overloaded function for dispatching to
// * netlib-compatible LAPACK backend (the default), and
// * complex<double> value-type.
//
inline std::ptrdiff_t hetri( char uplo, fortran_int_t n,
        std::complex<double>* a, fortran_int_t lda, const fortran_int_t* ipiv,
        std::complex<double>* work ) {
    fortran_int_t info(0);
    LAPACK_ZHETRI( &uplo, &n, a, &lda, ipiv, work, &info );
    return info;
}

} // namespace detail

//
// Value-type based template class. Use this class if you need a type
// for dispatching to hetri.
//
template< typename Value >
struct hetri_impl {

    typedef Value value_type;
    typedef typename remove_imaginary< Value >::type real_type;
    typedef tag::column_major order;

    //
    // Static member function for user-defined workspaces, that
    // * Deduces the required arguments for dispatching to LAPACK, and
    // * Asserts that most arguments make sense.
    //
    template< typename MatrixA, typename VectorIPIV, typename WORK >
    static std::ptrdiff_t invoke( const char uplo, MatrixA& a,
            const VectorIPIV& ipiv, detail::workspace1< WORK > work ) {
        BOOST_STATIC_ASSERT( (is_mutable< MatrixA >::value) );
        BOOST_ASSERT( size(ipiv) >= size_column(a) );
        BOOST_ASSERT( size(work.select(value_type())) >= min_size_work(
                size_column(a) ));
        BOOST_ASSERT( size_column(a) >= 0 );
        BOOST_ASSERT( size_minor(a) == 1 || stride_minor(a) == 1 );
        BOOST_ASSERT( stride_major(a) >= std::max< std::ptrdiff_t >(1,
                size_column(a)) );
        return detail::hetri( uplo, size_column(a), begin_value(a),
                stride_major(a), begin_value(ipiv),
                begin_value(work.select(value_type())) );
    }

    //
    // Static member function that
    // * Figures out the minimal workspace requirements, and passes
    //   the results to the user-defined workspace overload of the 
    //   invoke static member function
    // * Enables the unblocked algorithm (BLAS level 2)
    //
    template< typename MatrixA, typename VectorIPIV >
    static std::ptrdiff_t invoke( const char uplo, MatrixA& a,
            const VectorIPIV& ipiv, minimal_workspace work ) {
        bindings::detail::array< value_type > tmp_work( min_size_work(
                size_column(a) ) );
        return invoke( uplo, a, ipiv, workspace( tmp_work ) );
    }

    //
    // Static member function that
    // * Figures out the optimal workspace requirements, and passes
    //   the results to the user-defined workspace overload of the 
    //   invoke static member
    // * Enables the blocked algorithm (BLAS level 3)
    //
    template< typename MatrixA, typename VectorIPIV >
    static std::ptrdiff_t invoke( const char uplo, MatrixA& a,
            const VectorIPIV& ipiv, optimal_workspace work ) {
        return invoke( uplo, a, ipiv, minimal_workspace() );
    }

    //
    // Static member function that returns the minimum size of
    // workspace-array work.
    //
    static std::ptrdiff_t min_size_work( const std::ptrdiff_t n ) {
        return n;
    }
};


//
// Functions for direct use. These functions are overloaded for temporaries,
// so that wrapped types can still be passed and used for write-access. In
// addition, if applicable, they are overloaded for user-defined workspaces.
// Calls to these functions are passed to the hetri_impl classes. In the 
// documentation, most overloads are collapsed to avoid a large number of
// prototypes which are very similar.
//

//
// Overloaded function for hetri. Its overload differs for
// * MatrixA&
// * User-defined workspace
//
template< typename MatrixA, typename VectorIPIV, typename Workspace >
inline std::ptrdiff_t hetri( const char uplo, MatrixA& a,
        const VectorIPIV& ipiv, Workspace work ) {
    return hetri_impl< typename value< MatrixA >::type >::invoke( uplo,
            a, ipiv, work );
}

//
// Overloaded function for hetri. Its overload differs for
// * MatrixA&
// * Default workspace-type (optimal)
//
template< typename MatrixA, typename VectorIPIV >
inline std::ptrdiff_t hetri( const char uplo, MatrixA& a,
        const VectorIPIV& ipiv ) {
    return hetri_impl< typename value< MatrixA >::type >::invoke( uplo,
            a, ipiv, optimal_workspace() );
}

//
// Overloaded function for hetri. Its overload differs for
// * const MatrixA&
// * User-defined workspace
//
template< typename MatrixA, typename VectorIPIV, typename Workspace >
inline std::ptrdiff_t hetri( const char uplo, const MatrixA& a,
        const VectorIPIV& ipiv, Workspace work ) {
    return hetri_impl< typename value< MatrixA >::type >::invoke( uplo,
            a, ipiv, work );
}

//
// Overloaded function for hetri. Its overload differs for
// * const MatrixA&
// * Default workspace-type (optimal)
//
template< typename MatrixA, typename VectorIPIV >
inline std::ptrdiff_t hetri( const char uplo, const MatrixA& a,
        const VectorIPIV& ipiv ) {
    return hetri_impl< typename value< MatrixA >::type >::invoke( uplo,
            a, ipiv, optimal_workspace() );
}

} // namespace lapack
} // namespace bindings
} // namespace numeric
} // namespace boost

#endif
