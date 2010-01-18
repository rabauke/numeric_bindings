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

#ifndef BOOST_NUMERIC_BINDINGS_LAPACK_DRIVER_CGESV_HPP
#define BOOST_NUMERIC_BINDINGS_LAPACK_DRIVER_CGESV_HPP

#include <boost/assert.hpp>
#include <boost/numeric/bindings/begin.hpp>
#include <boost/numeric/bindings/detail/array.hpp>
#include <boost/numeric/bindings/is_mutable.hpp>
#include <boost/numeric/bindings/lapack/workspace.hpp>
#include <boost/numeric/bindings/remove_imaginary.hpp>
#include <boost/numeric/bindings/size.hpp>
#include <boost/numeric/bindings/stride.hpp>
#include <boost/numeric/bindings/value_type.hpp>
#include <boost/static_assert.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/type_traits/remove_const.hpp>

//
// The LAPACK-backend for cgesv is the netlib-compatible backend.
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
// * complex<double> value-type.
//
inline std::ptrdiff_t cgesv( const fortran_int_t n, const fortran_int_t nrhs,
        std::complex<double>* a, const fortran_int_t lda, fortran_int_t* ipiv,
        const std::complex<double>* b, const fortran_int_t ldb,
        std::complex<double>* x, const fortran_int_t ldx,
        std::complex<double>* work, std::complex<float>* swork, double* rwork,
        fortran_int_t& iter ) {
    fortran_int_t info(0);
    LAPACK_ZCGESV( &n, &nrhs, a, &lda, ipiv, b, &ldb, x, &ldx, work, swork,
            rwork, &iter, &info );
    return info;
}

} // namespace detail

//
// Value-type based template class. Use this class if you need a type
// for dispatching to cgesv.
//
template< typename Value >
struct cgesv_impl {

    typedef Value value_type;
    typedef typename remove_imaginary< Value >::type real_type;
    typedef tag::column_major order;

    //
    // Static member function for user-defined workspaces, that
    // * Deduces the required arguments for dispatching to LAPACK, and
    // * Asserts that most arguments make sense.
    //
    template< typename MatrixA, typename VectorIPIV, typename MatrixB,
            typename MatrixX, typename WORK, typename SWORK, typename RWORK >
    static std::ptrdiff_t invoke( MatrixA& a, VectorIPIV& ipiv,
            const MatrixB& b, MatrixX& x, fortran_int_t& iter,
            detail::workspace3< WORK, SWORK, RWORK > work ) {
        namespace bindings = ::boost::numeric::bindings;
        BOOST_STATIC_ASSERT( (boost::is_same< typename remove_const<
                typename bindings::value_type< MatrixA >::type >::type,
                typename remove_const< typename bindings::value_type<
                MatrixB >::type >::type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename remove_const<
                typename bindings::value_type< MatrixA >::type >::type,
                typename remove_const< typename bindings::value_type<
                MatrixX >::type >::type >::value) );
        BOOST_STATIC_ASSERT( (bindings::is_mutable< VectorIPIV >::value) );
        BOOST_STATIC_ASSERT( (bindings::is_mutable< MatrixX >::value) );
        BOOST_ASSERT( bindings::size(ipiv) >= bindings::stride_major(work) );
        BOOST_ASSERT( bindings::size(work.select(real_type())) >=
                min_size_rwork( bindings::stride_major(work) ));
        BOOST_ASSERT( bindings::size(work.select(value_type())) >=
                min_size_swork( bindings::stride_major(work),
                bindings::size_column(b) ));
        BOOST_ASSERT( bindings::size(work.select(value_type())) >=
                min_size_work( $CALL_MIN_SIZE ));
        BOOST_ASSERT( bindings::size_column(b) >= 0 );
        BOOST_ASSERT( bindings::size_minor(a) == 1 ||
                bindings::stride_minor(a) == 1 );
        BOOST_ASSERT( bindings::size_minor(b) == 1 ||
                bindings::stride_minor(b) == 1 );
        BOOST_ASSERT( bindings::size_minor(x) == 1 ||
                bindings::stride_minor(x) == 1 );
        BOOST_ASSERT( bindings::stride_major(a) >= std::max< std::ptrdiff_t >(1,
                bindings::stride_major(work)) );
        BOOST_ASSERT( bindings::stride_major(b) >= std::max< std::ptrdiff_t >(1,
                bindings::stride_major(work)) );
        BOOST_ASSERT( bindings::stride_major(work) >= 0 );
        BOOST_ASSERT( bindings::stride_major(x) >= std::max< std::ptrdiff_t >(1,
                bindings::stride_major(work)) );
        return detail::cgesv( bindings::stride_major(work),
                bindings::size_column(b), bindings::begin_value(a),
                bindings::stride_major(a), bindings::begin_value(ipiv),
                bindings::begin_value(b), bindings::stride_major(b),
                bindings::begin_value(x), bindings::stride_major(x),
                bindings::begin_value(work),
                bindings::begin_value(work.select(value_type())),
                bindings::begin_value(work.select(real_type())), iter );
    }

    //
    // Static member function that
    // * Figures out the minimal workspace requirements, and passes
    //   the results to the user-defined workspace overload of the 
    //   invoke static member function
    // * Enables the unblocked algorithm (BLAS level 2)
    //
    template< typename MatrixA, typename VectorIPIV, typename MatrixB,
            typename MatrixX >
    static std::ptrdiff_t invoke( MatrixA& a, VectorIPIV& ipiv,
            const MatrixB& b, MatrixX& x, fortran_int_t& iter,
            minimal_workspace work ) {
        namespace bindings = ::boost::numeric::bindings;
        bindings::detail::array< value_type > tmp_work( min_size_work(
                $CALL_MIN_SIZE ) );
        bindings::detail::array< value_type > tmp_swork( min_size_swork(
                bindings::stride_major(work), bindings::size_column(b) ) );
        bindings::detail::array< real_type > tmp_rwork( min_size_rwork(
                bindings::stride_major(work) ) );
        return invoke( a, ipiv, b, x, iter, workspace( tmp_work, tmp_swork,
                tmp_rwork ) );
    }

    //
    // Static member function that
    // * Figures out the optimal workspace requirements, and passes
    //   the results to the user-defined workspace overload of the 
    //   invoke static member
    // * Enables the blocked algorithm (BLAS level 3)
    //
    template< typename MatrixA, typename VectorIPIV, typename MatrixB,
            typename MatrixX >
    static std::ptrdiff_t invoke( MatrixA& a, VectorIPIV& ipiv,
            const MatrixB& b, MatrixX& x, fortran_int_t& iter,
            optimal_workspace work ) {
        namespace bindings = ::boost::numeric::bindings;
        return invoke( a, ipiv, b, x, iter, minimal_workspace() );
    }

    //
    // Static member function that returns the minimum size of
    // workspace-array work.
    //
    static std::ptrdiff_t min_size_work( $ARGUMENTS ) {
        return n*nrhs;
    }

    //
    // Static member function that returns the minimum size of
    // workspace-array swork.
    //
    static std::ptrdiff_t min_size_swork( const std::ptrdiff_t n,
            const std::ptrdiff_t nrhs ) {
        return n*(n+nrhs);
    }

    //
    // Static member function that returns the minimum size of
    // workspace-array rwork.
    //
    static std::ptrdiff_t min_size_rwork( const std::ptrdiff_t n ) {
        return n;
    }
};


//
// Functions for direct use. These functions are overloaded for temporaries,
// so that wrapped types can still be passed and used for write-access. In
// addition, if applicable, they are overloaded for user-defined workspaces.
// Calls to these functions are passed to the cgesv_impl classes. In the 
// documentation, most overloads are collapsed to avoid a large number of
// prototypes which are very similar.
//

//
// Overloaded function for cgesv. Its overload differs for
// * MatrixA&
// * VectorIPIV&
// * MatrixX&
// * User-defined workspace
//
template< typename MatrixA, typename VectorIPIV, typename MatrixB,
        typename MatrixX, typename Workspace >
inline typename boost::enable_if< detail::is_workspace< Workspace >,
        std::ptrdiff_t >::type
cgesv( MatrixA& a, VectorIPIV& ipiv, const MatrixB& b, MatrixX& x,
        fortran_int_t& iter, Workspace work ) {
    return cgesv_impl< typename bindings::value_type<
            MatrixA >::type >::invoke( a, ipiv, b, x, iter, work );
}

//
// Overloaded function for cgesv. Its overload differs for
// * MatrixA&
// * VectorIPIV&
// * MatrixX&
// * Default workspace-type (optimal)
//
template< typename MatrixA, typename VectorIPIV, typename MatrixB,
        typename MatrixX >
inline typename boost::disable_if< detail::is_workspace< MatrixX >,
        std::ptrdiff_t >::type
cgesv( MatrixA& a, VectorIPIV& ipiv, const MatrixB& b, MatrixX& x,
        fortran_int_t& iter ) {
    return cgesv_impl< typename bindings::value_type<
            MatrixA >::type >::invoke( a, ipiv, b, x, iter,
            optimal_workspace() );
}

//
// Overloaded function for cgesv. Its overload differs for
// * const MatrixA&
// * VectorIPIV&
// * MatrixX&
// * User-defined workspace
//
template< typename MatrixA, typename VectorIPIV, typename MatrixB,
        typename MatrixX, typename Workspace >
inline typename boost::enable_if< detail::is_workspace< Workspace >,
        std::ptrdiff_t >::type
cgesv( const MatrixA& a, VectorIPIV& ipiv, const MatrixB& b, MatrixX& x,
        fortran_int_t& iter, Workspace work ) {
    return cgesv_impl< typename bindings::value_type<
            MatrixA >::type >::invoke( a, ipiv, b, x, iter, work );
}

//
// Overloaded function for cgesv. Its overload differs for
// * const MatrixA&
// * VectorIPIV&
// * MatrixX&
// * Default workspace-type (optimal)
//
template< typename MatrixA, typename VectorIPIV, typename MatrixB,
        typename MatrixX >
inline typename boost::disable_if< detail::is_workspace< MatrixX >,
        std::ptrdiff_t >::type
cgesv( const MatrixA& a, VectorIPIV& ipiv, const MatrixB& b, MatrixX& x,
        fortran_int_t& iter ) {
    return cgesv_impl< typename bindings::value_type<
            MatrixA >::type >::invoke( a, ipiv, b, x, iter,
            optimal_workspace() );
}

//
// Overloaded function for cgesv. Its overload differs for
// * MatrixA&
// * const VectorIPIV&
// * MatrixX&
// * User-defined workspace
//
template< typename MatrixA, typename VectorIPIV, typename MatrixB,
        typename MatrixX, typename Workspace >
inline typename boost::enable_if< detail::is_workspace< Workspace >,
        std::ptrdiff_t >::type
cgesv( MatrixA& a, const VectorIPIV& ipiv, const MatrixB& b, MatrixX& x,
        fortran_int_t& iter, Workspace work ) {
    return cgesv_impl< typename bindings::value_type<
            MatrixA >::type >::invoke( a, ipiv, b, x, iter, work );
}

//
// Overloaded function for cgesv. Its overload differs for
// * MatrixA&
// * const VectorIPIV&
// * MatrixX&
// * Default workspace-type (optimal)
//
template< typename MatrixA, typename VectorIPIV, typename MatrixB,
        typename MatrixX >
inline typename boost::disable_if< detail::is_workspace< MatrixX >,
        std::ptrdiff_t >::type
cgesv( MatrixA& a, const VectorIPIV& ipiv, const MatrixB& b, MatrixX& x,
        fortran_int_t& iter ) {
    return cgesv_impl< typename bindings::value_type<
            MatrixA >::type >::invoke( a, ipiv, b, x, iter,
            optimal_workspace() );
}

//
// Overloaded function for cgesv. Its overload differs for
// * const MatrixA&
// * const VectorIPIV&
// * MatrixX&
// * User-defined workspace
//
template< typename MatrixA, typename VectorIPIV, typename MatrixB,
        typename MatrixX, typename Workspace >
inline typename boost::enable_if< detail::is_workspace< Workspace >,
        std::ptrdiff_t >::type
cgesv( const MatrixA& a, const VectorIPIV& ipiv, const MatrixB& b,
        MatrixX& x, fortran_int_t& iter, Workspace work ) {
    return cgesv_impl< typename bindings::value_type<
            MatrixA >::type >::invoke( a, ipiv, b, x, iter, work );
}

//
// Overloaded function for cgesv. Its overload differs for
// * const MatrixA&
// * const VectorIPIV&
// * MatrixX&
// * Default workspace-type (optimal)
//
template< typename MatrixA, typename VectorIPIV, typename MatrixB,
        typename MatrixX >
inline typename boost::disable_if< detail::is_workspace< MatrixX >,
        std::ptrdiff_t >::type
cgesv( const MatrixA& a, const VectorIPIV& ipiv, const MatrixB& b,
        MatrixX& x, fortran_int_t& iter ) {
    return cgesv_impl< typename bindings::value_type<
            MatrixA >::type >::invoke( a, ipiv, b, x, iter,
            optimal_workspace() );
}

//
// Overloaded function for cgesv. Its overload differs for
// * MatrixA&
// * VectorIPIV&
// * const MatrixX&
// * User-defined workspace
//
template< typename MatrixA, typename VectorIPIV, typename MatrixB,
        typename MatrixX, typename Workspace >
inline typename boost::enable_if< detail::is_workspace< Workspace >,
        std::ptrdiff_t >::type
cgesv( MatrixA& a, VectorIPIV& ipiv, const MatrixB& b, const MatrixX& x,
        fortran_int_t& iter, Workspace work ) {
    return cgesv_impl< typename bindings::value_type<
            MatrixA >::type >::invoke( a, ipiv, b, x, iter, work );
}

//
// Overloaded function for cgesv. Its overload differs for
// * MatrixA&
// * VectorIPIV&
// * const MatrixX&
// * Default workspace-type (optimal)
//
template< typename MatrixA, typename VectorIPIV, typename MatrixB,
        typename MatrixX >
inline typename boost::disable_if< detail::is_workspace< MatrixX >,
        std::ptrdiff_t >::type
cgesv( MatrixA& a, VectorIPIV& ipiv, const MatrixB& b, const MatrixX& x,
        fortran_int_t& iter ) {
    return cgesv_impl< typename bindings::value_type<
            MatrixA >::type >::invoke( a, ipiv, b, x, iter,
            optimal_workspace() );
}

//
// Overloaded function for cgesv. Its overload differs for
// * const MatrixA&
// * VectorIPIV&
// * const MatrixX&
// * User-defined workspace
//
template< typename MatrixA, typename VectorIPIV, typename MatrixB,
        typename MatrixX, typename Workspace >
inline typename boost::enable_if< detail::is_workspace< Workspace >,
        std::ptrdiff_t >::type
cgesv( const MatrixA& a, VectorIPIV& ipiv, const MatrixB& b,
        const MatrixX& x, fortran_int_t& iter, Workspace work ) {
    return cgesv_impl< typename bindings::value_type<
            MatrixA >::type >::invoke( a, ipiv, b, x, iter, work );
}

//
// Overloaded function for cgesv. Its overload differs for
// * const MatrixA&
// * VectorIPIV&
// * const MatrixX&
// * Default workspace-type (optimal)
//
template< typename MatrixA, typename VectorIPIV, typename MatrixB,
        typename MatrixX >
inline typename boost::disable_if< detail::is_workspace< MatrixX >,
        std::ptrdiff_t >::type
cgesv( const MatrixA& a, VectorIPIV& ipiv, const MatrixB& b,
        const MatrixX& x, fortran_int_t& iter ) {
    return cgesv_impl< typename bindings::value_type<
            MatrixA >::type >::invoke( a, ipiv, b, x, iter,
            optimal_workspace() );
}

//
// Overloaded function for cgesv. Its overload differs for
// * MatrixA&
// * const VectorIPIV&
// * const MatrixX&
// * User-defined workspace
//
template< typename MatrixA, typename VectorIPIV, typename MatrixB,
        typename MatrixX, typename Workspace >
inline typename boost::enable_if< detail::is_workspace< Workspace >,
        std::ptrdiff_t >::type
cgesv( MatrixA& a, const VectorIPIV& ipiv, const MatrixB& b,
        const MatrixX& x, fortran_int_t& iter, Workspace work ) {
    return cgesv_impl< typename bindings::value_type<
            MatrixA >::type >::invoke( a, ipiv, b, x, iter, work );
}

//
// Overloaded function for cgesv. Its overload differs for
// * MatrixA&
// * const VectorIPIV&
// * const MatrixX&
// * Default workspace-type (optimal)
//
template< typename MatrixA, typename VectorIPIV, typename MatrixB,
        typename MatrixX >
inline typename boost::disable_if< detail::is_workspace< MatrixX >,
        std::ptrdiff_t >::type
cgesv( MatrixA& a, const VectorIPIV& ipiv, const MatrixB& b,
        const MatrixX& x, fortran_int_t& iter ) {
    return cgesv_impl< typename bindings::value_type<
            MatrixA >::type >::invoke( a, ipiv, b, x, iter,
            optimal_workspace() );
}

//
// Overloaded function for cgesv. Its overload differs for
// * const MatrixA&
// * const VectorIPIV&
// * const MatrixX&
// * User-defined workspace
//
template< typename MatrixA, typename VectorIPIV, typename MatrixB,
        typename MatrixX, typename Workspace >
inline typename boost::enable_if< detail::is_workspace< Workspace >,
        std::ptrdiff_t >::type
cgesv( const MatrixA& a, const VectorIPIV& ipiv, const MatrixB& b,
        const MatrixX& x, fortran_int_t& iter, Workspace work ) {
    return cgesv_impl< typename bindings::value_type<
            MatrixA >::type >::invoke( a, ipiv, b, x, iter, work );
}

//
// Overloaded function for cgesv. Its overload differs for
// * const MatrixA&
// * const VectorIPIV&
// * const MatrixX&
// * Default workspace-type (optimal)
//
template< typename MatrixA, typename VectorIPIV, typename MatrixB,
        typename MatrixX >
inline typename boost::disable_if< detail::is_workspace< MatrixX >,
        std::ptrdiff_t >::type
cgesv( const MatrixA& a, const VectorIPIV& ipiv, const MatrixB& b,
        const MatrixX& x, fortran_int_t& iter ) {
    return cgesv_impl< typename bindings::value_type<
            MatrixA >::type >::invoke( a, ipiv, b, x, iter,
            optimal_workspace() );
}

} // namespace lapack
} // namespace bindings
} // namespace numeric
} // namespace boost

#endif
