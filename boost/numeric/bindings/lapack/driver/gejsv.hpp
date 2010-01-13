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

#ifndef BOOST_NUMERIC_BINDINGS_LAPACK_DRIVER_GEJSV_HPP
#define BOOST_NUMERIC_BINDINGS_LAPACK_DRIVER_GEJSV_HPP

#include <boost/assert.hpp>
#include <boost/numeric/bindings/begin.hpp>
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
// The LAPACK-backend for gejsv is the netlib-compatible backend.
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
inline std::ptrdiff_t gejsv( const char joba, const char jobu, const char jobv,
        const char jobr, const char jobt, const char jobp,
        const fortran_int_t m, const fortran_int_t n, float* a,
        const fortran_int_t lda, float* sva, float* u,
        const fortran_int_t ldu, float* v, const fortran_int_t ldv,
        float* work, const fortran_int_t lwork, fortran_int_t* iwork ) {
    fortran_int_t info(0);
    LAPACK_SGEJSV( &joba, &jobu, &jobv, &jobr, &jobt, &jobp, &m, &n, a, &lda,
            sva, u, &ldu, v, &ldv, work, &lwork, iwork, &info );
    return info;
}

//
// Overloaded function for dispatching to
// * netlib-compatible LAPACK backend (the default), and
// * double value-type.
//
inline std::ptrdiff_t gejsv( const char joba, const char jobu, const char jobv,
        const char jobr, const char jobt, const char jobp,
        const fortran_int_t m, const fortran_int_t n, double* a,
        const fortran_int_t lda, double* sva, double* u,
        const fortran_int_t ldu, double* v, const fortran_int_t ldv,
        double* work, const fortran_int_t lwork, fortran_int_t* iwork ) {
    fortran_int_t info(0);
    LAPACK_DGEJSV( &joba, &jobu, &jobv, &jobr, &jobt, &jobp, &m, &n, a, &lda,
            sva, u, &ldu, v, &ldv, work, &lwork, iwork, &info );
    return info;
}

} // namespace detail

//
// Value-type based template class. Use this class if you need a type
// for dispatching to gejsv.
//
template< typename Value >
struct gejsv_impl {

    typedef Value value_type;
    typedef typename remove_imaginary< Value >::type real_type;
    typedef tag::column_major order;

    //
    // Static member function for user-defined workspaces, that
    // * Deduces the required arguments for dispatching to LAPACK, and
    // * Asserts that most arguments make sense.
    //
    template< typename MatrixA, typename SVA, typename U, typename V,
            typename WORK, typename IWORK >
    static std::ptrdiff_t invoke( const char joba, const char jobu,
            const char jobv, const char jobr, const char jobt,
            const char jobp, MatrixA& a, const fortran_int_t lwork,
            detail::workspace5< SVA, U, V, WORK, IWORK > work ) {
        BOOST_STATIC_ASSERT( (is_mutable< MatrixA >::value) );
        BOOST_ASSERT( size(work.select(fortran_int_t())) >=
                min_size_iwork( $CALL_MIN_SIZE ));
        BOOST_ASSERT( size(work.select(real_type())) >=
                min_size_sva( $CALL_MIN_SIZE ));
        BOOST_ASSERT( size(work.select(real_type())) >=
                min_size_u( $CALL_MIN_SIZE ));
        BOOST_ASSERT( size(work.select(real_type())) >=
                min_size_v( $CALL_MIN_SIZE ));
        BOOST_ASSERT( size(work.select(real_type())) >= min_size_work(
                $CALL_MIN_SIZE ));
        BOOST_ASSERT( size_minor(a) == 1 || stride_minor(a) == 1 );
        BOOST_ASSERT( size_row(a) >= 0 );
        BOOST_ASSERT( stride_major(a) >= std::max< std::ptrdiff_t >(1,
                size_row(a)) );
        return detail::gejsv( joba, jobu, jobv, jobr, jobt, jobp, size_row(a),
                size_column(a), begin_value(a), stride_major(a),
                begin_value(work.select(real_type())), begin_value(u),
                stride_major(u), begin_value(v), stride_major(v),
                begin_value(work.select(real_type())), lwork,
                begin_value(work.select(fortran_int_t())) );
    }

    //
    // Static member function that
    // * Figures out the minimal workspace requirements, and passes
    //   the results to the user-defined workspace overload of the 
    //   invoke static member function
    // * Enables the unblocked algorithm (BLAS level 2)
    //
    template< typename MatrixA >
    static std::ptrdiff_t invoke( const char joba, const char jobu,
            const char jobv, const char jobr, const char jobt,
            const char jobp, MatrixA& a, const fortran_int_t lwork,
            minimal_workspace work ) {
        bindings::detail::array<
                real_type > tmp_sva( min_size_sva( $CALL_MIN_SIZE ) );
        bindings::detail::array<
                real_type > tmp_u( min_size_u( $CALL_MIN_SIZE ) );
        bindings::detail::array<
                real_type > tmp_v( min_size_v( $CALL_MIN_SIZE ) );
        bindings::detail::array< real_type > tmp_work( min_size_work(
                $CALL_MIN_SIZE ) );
        bindings::detail::array< fortran_int_t > tmp_iwork(
                min_size_iwork( $CALL_MIN_SIZE ) );
        return invoke( joba, jobu, jobv, jobr, jobt, jobp, a, lwork,
                workspace( tmp_sva, tmp_u, tmp_v, tmp_work, tmp_iwork ) );
    }

    //
    // Static member function that
    // * Figures out the optimal workspace requirements, and passes
    //   the results to the user-defined workspace overload of the 
    //   invoke static member
    // * Enables the blocked algorithm (BLAS level 3)
    //
    template< typename MatrixA >
    static std::ptrdiff_t invoke( const char joba, const char jobu,
            const char jobv, const char jobr, const char jobt,
            const char jobp, MatrixA& a, const fortran_int_t lwork,
            optimal_workspace work ) {
        return invoke( joba, jobu, jobv, jobr, jobt, jobp, a, lwork,
                minimal_workspace() );
    }

    //
    // Static member function that returns the minimum size of
    // workspace-array sva.
    //
    static std::ptrdiff_t min_size_sva( $ARGUMENTS ) {
        $MIN_SIZE
    }

    //
    // Static member function that returns the minimum size of
    // workspace-array u.
    //
    static std::ptrdiff_t min_size_u( $ARGUMENTS ) {
        $MIN_SIZE
    }

    //
    // Static member function that returns the minimum size of
    // workspace-array v.
    //
    static std::ptrdiff_t min_size_v( $ARGUMENTS ) {
        $MIN_SIZE
    }

    //
    // Static member function that returns the minimum size of
    // workspace-array work.
    //
    static std::ptrdiff_t min_size_work( $ARGUMENTS ) {
        $MIN_SIZE
    }

    //
    // Static member function that returns the minimum size of
    // workspace-array iwork.
    //
    static std::ptrdiff_t min_size_iwork( $ARGUMENTS ) {
        $MIN_SIZE
    }
};


//
// Functions for direct use. These functions are overloaded for temporaries,
// so that wrapped types can still be passed and used for write-access. In
// addition, if applicable, they are overloaded for user-defined workspaces.
// Calls to these functions are passed to the gejsv_impl classes. In the 
// documentation, most overloads are collapsed to avoid a large number of
// prototypes which are very similar.
//

//
// Overloaded function for gejsv. Its overload differs for
// * MatrixA&
// * User-defined workspace
//
template< typename MatrixA, typename Workspace >
inline std::ptrdiff_t gejsv( const char joba, const char jobu,
        const char jobv, const char jobr, const char jobt, const char jobp,
        MatrixA& a, const fortran_int_t lwork, Workspace work ) {
    return gejsv_impl< typename value< MatrixA >::type >::invoke( joba,
            jobu, jobv, jobr, jobt, jobp, a, lwork, work );
}

//
// Overloaded function for gejsv. Its overload differs for
// * MatrixA&
// * Default workspace-type (optimal)
//
template< typename MatrixA >
inline std::ptrdiff_t gejsv( const char joba, const char jobu,
        const char jobv, const char jobr, const char jobt, const char jobp,
        MatrixA& a, const fortran_int_t lwork ) {
    return gejsv_impl< typename value< MatrixA >::type >::invoke( joba,
            jobu, jobv, jobr, jobt, jobp, a, lwork, optimal_workspace() );
}

//
// Overloaded function for gejsv. Its overload differs for
// * const MatrixA&
// * User-defined workspace
//
template< typename MatrixA, typename Workspace >
inline std::ptrdiff_t gejsv( const char joba, const char jobu,
        const char jobv, const char jobr, const char jobt, const char jobp,
        const MatrixA& a, const fortran_int_t lwork, Workspace work ) {
    return gejsv_impl< typename value< MatrixA >::type >::invoke( joba,
            jobu, jobv, jobr, jobt, jobp, a, lwork, work );
}

//
// Overloaded function for gejsv. Its overload differs for
// * const MatrixA&
// * Default workspace-type (optimal)
//
template< typename MatrixA >
inline std::ptrdiff_t gejsv( const char joba, const char jobu,
        const char jobv, const char jobr, const char jobt, const char jobp,
        const MatrixA& a, const fortran_int_t lwork ) {
    return gejsv_impl< typename value< MatrixA >::type >::invoke( joba,
            jobu, jobv, jobr, jobt, jobp, a, lwork, optimal_workspace() );
}

} // namespace lapack
} // namespace bindings
} // namespace numeric
} // namespace boost

#endif
