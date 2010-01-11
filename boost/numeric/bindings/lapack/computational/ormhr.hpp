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

#ifndef BOOST_NUMERIC_BINDINGS_LAPACK_COMPUTATIONAL_ORMHR_HPP
#define BOOST_NUMERIC_BINDINGS_LAPACK_COMPUTATIONAL_ORMHR_HPP

#include <boost/assert.hpp>
#include <boost/numeric/bindings/begin.hpp>
#include <boost/numeric/bindings/detail/array.hpp>
#include <boost/numeric/bindings/is_mutable.hpp>
#include <boost/numeric/bindings/lapack/workspace.hpp>
#include <boost/numeric/bindings/remove_imaginary.hpp>
#include <boost/numeric/bindings/size.hpp>
#include <boost/numeric/bindings/stride.hpp>
#include <boost/numeric/bindings/traits/detail/utils.hpp>
#include <boost/numeric/bindings/trans_tag.hpp>
#include <boost/numeric/bindings/value.hpp>
#include <boost/static_assert.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/type_traits/remove_const.hpp>

//
// The LAPACK-backend for ormhr is the netlib-compatible backend.
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
template< typename Trans >
inline std::ptrdiff_t ormhr( char side, Trans, fortran_int_t m,
        fortran_int_t n, fortran_int_t ilo, fortran_int_t ihi, const float* a,
        fortran_int_t lda, const float* tau, float* c, fortran_int_t ldc,
        float* work, fortran_int_t lwork ) {
    fortran_int_t info(0);
    LAPACK_SORMHR( &side, &lapack_option< Trans >::value, &m, &n, &ilo, &ihi,
            a, &lda, tau, c, &ldc, work, &lwork, &info );
    return info;
}

//
// Overloaded function for dispatching to
// * netlib-compatible LAPACK backend (the default), and
// * double value-type.
//
template< typename Trans >
inline std::ptrdiff_t ormhr( char side, Trans, fortran_int_t m,
        fortran_int_t n, fortran_int_t ilo, fortran_int_t ihi,
        const double* a, fortran_int_t lda, const double* tau, double* c,
        fortran_int_t ldc, double* work, fortran_int_t lwork ) {
    fortran_int_t info(0);
    LAPACK_DORMHR( &side, &lapack_option< Trans >::value, &m, &n, &ilo, &ihi,
            a, &lda, tau, c, &ldc, work, &lwork, &info );
    return info;
}

} // namespace detail

//
// Value-type based template class. Use this class if you need a type
// for dispatching to ormhr.
//
template< typename Value >
struct ormhr_impl {

    typedef Value value_type;
    typedef typename remove_imaginary< Value >::type real_type;
    typedef tag::column_major order;

    //
    // Static member function for user-defined workspaces, that
    // * Deduces the required arguments for dispatching to LAPACK, and
    // * Asserts that most arguments make sense.
    //
    template< typename MatrixA, typename VectorTAU, typename MatrixC,
            typename WORK >
    static std::ptrdiff_t invoke( const char side, const fortran_int_t ilo,
            const fortran_int_t ihi, const MatrixA& a,
            const VectorTAU& tau, MatrixC& c, detail::workspace1<
            WORK > work ) {
        typedef typename result_of::trans_tag< MatrixA, order >::type trans;
        BOOST_STATIC_ASSERT( (boost::is_same< typename remove_const<
                typename value< MatrixA >::type >::type,
                typename remove_const< typename value<
                VectorTAU >::type >::type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename remove_const<
                typename value< MatrixA >::type >::type,
                typename remove_const< typename value<
                MatrixC >::type >::type >::value) );
        BOOST_STATIC_ASSERT( (is_mutable< MatrixC >::value) );
        BOOST_ASSERT( side == 'L' || side == 'R' );
        BOOST_ASSERT( size(work.select(real_type())) >= min_size_work( side,
                size_row(c), size_column(c) ));
        BOOST_ASSERT( size_column(c) >= 0 );
        BOOST_ASSERT( size_minor(a) == 1 || stride_minor(a) == 1 );
        BOOST_ASSERT( size_minor(c) == 1 || stride_minor(c) == 1 );
        BOOST_ASSERT( size_row(c) >= 0 );
        BOOST_ASSERT( stride_major(c) >= std::max< std::ptrdiff_t >(1,
                size_row(c)) );
        return detail::ormhr( side, trans(), size_row(c), size_column(c), ilo,
                ihi, begin_value(a), stride_major(a), begin_value(tau),
                begin_value(c), stride_major(c),
                begin_value(work.select(real_type())),
                size(work.select(real_type())) );
    }

    //
    // Static member function that
    // * Figures out the minimal workspace requirements, and passes
    //   the results to the user-defined workspace overload of the 
    //   invoke static member function
    // * Enables the unblocked algorithm (BLAS level 2)
    //
    template< typename MatrixA, typename VectorTAU, typename MatrixC >
    static std::ptrdiff_t invoke( const char side, const fortran_int_t ilo,
            const fortran_int_t ihi, const MatrixA& a,
            const VectorTAU& tau, MatrixC& c, minimal_workspace work ) {
        typedef typename result_of::trans_tag< MatrixA, order >::type trans;
        bindings::detail::array< real_type > tmp_work( min_size_work( side,
                size_row(c), size_column(c) ) );
        return invoke( side, ilo, ihi, a, tau, c, workspace( tmp_work ) );
    }

    //
    // Static member function that
    // * Figures out the optimal workspace requirements, and passes
    //   the results to the user-defined workspace overload of the 
    //   invoke static member
    // * Enables the blocked algorithm (BLAS level 3)
    //
    template< typename MatrixA, typename VectorTAU, typename MatrixC >
    static std::ptrdiff_t invoke( const char side, const fortran_int_t ilo,
            const fortran_int_t ihi, const MatrixA& a,
            const VectorTAU& tau, MatrixC& c, optimal_workspace work ) {
        typedef typename result_of::trans_tag< MatrixA, order >::type trans;
        real_type opt_size_work;
        detail::ormhr( side, trans(), size_row(c), size_column(c), ilo,
                ihi, begin_value(a), stride_major(a), begin_value(tau),
                begin_value(c), stride_major(c), &opt_size_work, -1 );
        bindings::detail::array< real_type > tmp_work(
                traits::detail::to_int( opt_size_work ) );
        invoke( side, ilo, ihi, a, tau, c, workspace( tmp_work ) );
    }

    //
    // Static member function that returns the minimum size of
    // workspace-array work.
    //
    static std::ptrdiff_t min_size_work( const char side,
            const std::ptrdiff_t m, const std::ptrdiff_t n ) {
        if ( side == 'L' )
            return std::max< std::ptrdiff_t >( 1, n );
        else
            return std::max< std::ptrdiff_t >( 1, m );
    }
};


//
// Functions for direct use. These functions are overloaded for temporaries,
// so that wrapped types can still be passed and used for write-access. In
// addition, if applicable, they are overloaded for user-defined workspaces.
// Calls to these functions are passed to the ormhr_impl classes. In the 
// documentation, most overloads are collapsed to avoid a large number of
// prototypes which are very similar.
//

//
// Overloaded function for ormhr. Its overload differs for
// * MatrixC&
// * User-defined workspace
//
template< typename MatrixA, typename VectorTAU, typename MatrixC,
        typename Workspace >
inline std::ptrdiff_t ormhr( const char side, const fortran_int_t ilo,
        const fortran_int_t ihi, const MatrixA& a, const VectorTAU& tau,
        MatrixC& c, Workspace work ) {
    return ormhr_impl< typename value< MatrixA >::type >::invoke( side,
            ilo, ihi, a, tau, c, work );
}

//
// Overloaded function for ormhr. Its overload differs for
// * MatrixC&
// * Default workspace-type (optimal)
//
template< typename MatrixA, typename VectorTAU, typename MatrixC >
inline std::ptrdiff_t ormhr( const char side, const fortran_int_t ilo,
        const fortran_int_t ihi, const MatrixA& a, const VectorTAU& tau,
        MatrixC& c ) {
    return ormhr_impl< typename value< MatrixA >::type >::invoke( side,
            ilo, ihi, a, tau, c, optimal_workspace() );
}

//
// Overloaded function for ormhr. Its overload differs for
// * const MatrixC&
// * User-defined workspace
//
template< typename MatrixA, typename VectorTAU, typename MatrixC,
        typename Workspace >
inline std::ptrdiff_t ormhr( const char side, const fortran_int_t ilo,
        const fortran_int_t ihi, const MatrixA& a, const VectorTAU& tau,
        const MatrixC& c, Workspace work ) {
    return ormhr_impl< typename value< MatrixA >::type >::invoke( side,
            ilo, ihi, a, tau, c, work );
}

//
// Overloaded function for ormhr. Its overload differs for
// * const MatrixC&
// * Default workspace-type (optimal)
//
template< typename MatrixA, typename VectorTAU, typename MatrixC >
inline std::ptrdiff_t ormhr( const char side, const fortran_int_t ilo,
        const fortran_int_t ihi, const MatrixA& a, const VectorTAU& tau,
        const MatrixC& c ) {
    return ormhr_impl< typename value< MatrixA >::type >::invoke( side,
            ilo, ihi, a, tau, c, optimal_workspace() );
}

} // namespace lapack
} // namespace bindings
} // namespace numeric
} // namespace boost

#endif
