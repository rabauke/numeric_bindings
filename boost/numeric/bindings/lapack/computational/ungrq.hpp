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

#ifndef BOOST_NUMERIC_BINDINGS_LAPACK_COMPUTATIONAL_UNGRQ_HPP
#define BOOST_NUMERIC_BINDINGS_LAPACK_COMPUTATIONAL_UNGRQ_HPP

#include <boost/assert.hpp>
#include <boost/numeric/bindings/begin.hpp>
#include <boost/numeric/bindings/detail/array.hpp>
#include <boost/numeric/bindings/is_column_major.hpp>
#include <boost/numeric/bindings/is_mutable.hpp>
#include <boost/numeric/bindings/lapack/workspace.hpp>
#include <boost/numeric/bindings/remove_imaginary.hpp>
#include <boost/numeric/bindings/size.hpp>
#include <boost/numeric/bindings/stride.hpp>
#include <boost/numeric/bindings/traits/detail/utils.hpp>
#include <boost/numeric/bindings/value_type.hpp>
#include <boost/static_assert.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/type_traits/remove_const.hpp>

//
// The LAPACK-backend for ungrq is the netlib-compatible backend.
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
inline std::ptrdiff_t ungrq( const fortran_int_t m, const fortran_int_t n,
        const fortran_int_t k, std::complex<float>* a,
        const fortran_int_t lda, const std::complex<float>* tau,
        std::complex<float>* work, const fortran_int_t lwork ) {
    fortran_int_t info(0);
    LAPACK_CUNGRQ( &m, &n, &k, a, &lda, tau, work, &lwork, &info );
    return info;
}

//
// Overloaded function for dispatching to
// * netlib-compatible LAPACK backend (the default), and
// * complex<double> value-type.
//
inline std::ptrdiff_t ungrq( const fortran_int_t m, const fortran_int_t n,
        const fortran_int_t k, std::complex<double>* a,
        const fortran_int_t lda, const std::complex<double>* tau,
        std::complex<double>* work, const fortran_int_t lwork ) {
    fortran_int_t info(0);
    LAPACK_ZUNGRQ( &m, &n, &k, a, &lda, tau, work, &lwork, &info );
    return info;
}

} // namespace detail

//
// Value-type based template class. Use this class if you need a type
// for dispatching to ungrq.
//
template< typename Value >
struct ungrq_impl {

    typedef Value value_type;
    typedef typename remove_imaginary< Value >::type real_type;

    //
    // Static member function for user-defined workspaces, that
    // * Deduces the required arguments for dispatching to LAPACK, and
    // * Asserts that most arguments make sense.
    //
    template< typename MatrixA, typename VectorTAU, typename WORK >
    static std::ptrdiff_t invoke( const fortran_int_t m,
            const fortran_int_t n, const fortran_int_t k, MatrixA& a,
            const VectorTAU& tau, detail::workspace1< WORK > work ) {
        namespace bindings = ::boost::numeric::bindings;
        BOOST_STATIC_ASSERT( (bindings::is_column_major< MatrixA >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename remove_const<
                typename bindings::value_type< MatrixA >::type >::type,
                typename remove_const< typename bindings::value_type<
                VectorTAU >::type >::type >::value) );
        BOOST_STATIC_ASSERT( (bindings::is_mutable< MatrixA >::value) );
        BOOST_ASSERT( bindings::size(tau) >= k );
        BOOST_ASSERT( bindings::size(work.select(value_type())) >=
                min_size_work( m ));
        BOOST_ASSERT( bindings::size_minor(a) == 1 ||
                bindings::stride_minor(a) == 1 );
        BOOST_ASSERT( bindings::stride_major(a) >= std::max< std::ptrdiff_t >(1,
                m) );
        BOOST_ASSERT( m >= 0 );
        BOOST_ASSERT( n >= m );
        return detail::ungrq( m, n, k, bindings::begin_value(a),
                bindings::stride_major(a), bindings::begin_value(tau),
                bindings::begin_value(work.select(value_type())),
                bindings::size(work.select(value_type())) );
    }

    //
    // Static member function that
    // * Figures out the minimal workspace requirements, and passes
    //   the results to the user-defined workspace overload of the 
    //   invoke static member function
    // * Enables the unblocked algorithm (BLAS level 2)
    //
    template< typename MatrixA, typename VectorTAU >
    static std::ptrdiff_t invoke( const fortran_int_t m,
            const fortran_int_t n, const fortran_int_t k, MatrixA& a,
            const VectorTAU& tau, minimal_workspace ) {
        namespace bindings = ::boost::numeric::bindings;
        bindings::detail::array< value_type > tmp_work( min_size_work( m ) );
        return invoke( m, n, k, a, tau, workspace( tmp_work ) );
    }

    //
    // Static member function that
    // * Figures out the optimal workspace requirements, and passes
    //   the results to the user-defined workspace overload of the 
    //   invoke static member
    // * Enables the blocked algorithm (BLAS level 3)
    //
    template< typename MatrixA, typename VectorTAU >
    static std::ptrdiff_t invoke( const fortran_int_t m,
            const fortran_int_t n, const fortran_int_t k, MatrixA& a,
            const VectorTAU& tau, optimal_workspace ) {
        namespace bindings = ::boost::numeric::bindings;
        value_type opt_size_work;
        detail::ungrq( m, n, k, bindings::begin_value(a),
                bindings::stride_major(a), bindings::begin_value(tau),
                &opt_size_work, -1 );
        bindings::detail::array< value_type > tmp_work(
                traits::detail::to_int( opt_size_work ) );
        return invoke( m, n, k, a, tau, workspace( tmp_work ) );
    }

    //
    // Static member function that returns the minimum size of
    // workspace-array work.
    //
    static std::ptrdiff_t min_size_work( const std::ptrdiff_t m ) {
        return std::max< std::ptrdiff_t >( 1, m );
    }
};


//
// Functions for direct use. These functions are overloaded for temporaries,
// so that wrapped types can still be passed and used for write-access. In
// addition, if applicable, they are overloaded for user-defined workspaces.
// Calls to these functions are passed to the ungrq_impl classes. In the 
// documentation, most overloads are collapsed to avoid a large number of
// prototypes which are very similar.
//

//
// Overloaded function for ungrq. Its overload differs for
// * MatrixA&
// * User-defined workspace
//
template< typename MatrixA, typename VectorTAU, typename Workspace >
inline typename boost::enable_if< detail::is_workspace< Workspace >,
        std::ptrdiff_t >::type
ungrq( const fortran_int_t m, const fortran_int_t n,
        const fortran_int_t k, MatrixA& a, const VectorTAU& tau,
        Workspace work ) {
    return ungrq_impl< typename bindings::value_type<
            MatrixA >::type >::invoke( m, n, k, a, tau, work );
}

//
// Overloaded function for ungrq. Its overload differs for
// * MatrixA&
// * Default workspace-type (optimal)
//
template< typename MatrixA, typename VectorTAU >
inline typename boost::disable_if< detail::is_workspace< VectorTAU >,
        std::ptrdiff_t >::type
ungrq( const fortran_int_t m, const fortran_int_t n,
        const fortran_int_t k, MatrixA& a, const VectorTAU& tau ) {
    return ungrq_impl< typename bindings::value_type<
            MatrixA >::type >::invoke( m, n, k, a, tau, optimal_workspace() );
}

//
// Overloaded function for ungrq. Its overload differs for
// * const MatrixA&
// * User-defined workspace
//
template< typename MatrixA, typename VectorTAU, typename Workspace >
inline typename boost::enable_if< detail::is_workspace< Workspace >,
        std::ptrdiff_t >::type
ungrq( const fortran_int_t m, const fortran_int_t n,
        const fortran_int_t k, const MatrixA& a, const VectorTAU& tau,
        Workspace work ) {
    return ungrq_impl< typename bindings::value_type<
            MatrixA >::type >::invoke( m, n, k, a, tau, work );
}

//
// Overloaded function for ungrq. Its overload differs for
// * const MatrixA&
// * Default workspace-type (optimal)
//
template< typename MatrixA, typename VectorTAU >
inline typename boost::disable_if< detail::is_workspace< VectorTAU >,
        std::ptrdiff_t >::type
ungrq( const fortran_int_t m, const fortran_int_t n,
        const fortran_int_t k, const MatrixA& a, const VectorTAU& tau ) {
    return ungrq_impl< typename bindings::value_type<
            MatrixA >::type >::invoke( m, n, k, a, tau, optimal_workspace() );
}

} // namespace lapack
} // namespace bindings
} // namespace numeric
} // namespace boost

#endif
