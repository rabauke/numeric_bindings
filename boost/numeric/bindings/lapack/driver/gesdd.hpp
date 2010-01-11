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

#ifndef BOOST_NUMERIC_BINDINGS_LAPACK_DRIVER_GESDD_HPP
#define BOOST_NUMERIC_BINDINGS_LAPACK_DRIVER_GESDD_HPP

#include <boost/assert.hpp>
#include <boost/numeric/bindings/begin.hpp>
#include <boost/numeric/bindings/detail/array.hpp>
#include <boost/numeric/bindings/is_complex.hpp>
#include <boost/numeric/bindings/is_mutable.hpp>
#include <boost/numeric/bindings/is_real.hpp>
#include <boost/numeric/bindings/lapack/workspace.hpp>
#include <boost/numeric/bindings/remove_imaginary.hpp>
#include <boost/numeric/bindings/size.hpp>
#include <boost/numeric/bindings/stride.hpp>
#include <boost/numeric/bindings/traits/detail/utils.hpp>
#include <boost/numeric/bindings/value.hpp>
#include <boost/static_assert.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/type_traits/remove_const.hpp>
#include <boost/utility/enable_if.hpp>

//
// The LAPACK-backend for gesdd is the netlib-compatible backend.
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
inline std::ptrdiff_t gesdd( char jobz, fortran_int_t m, fortran_int_t n,
        float* a, fortran_int_t lda, float* s, float* u, fortran_int_t ldu,
        float* vt, fortran_int_t ldvt, float* work, fortran_int_t lwork,
        fortran_int_t* iwork ) {
    fortran_int_t info(0);
    LAPACK_SGESDD( &jobz, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work,
            &lwork, iwork, &info );
    return info;
}

//
// Overloaded function for dispatching to
// * netlib-compatible LAPACK backend (the default), and
// * double value-type.
//
inline std::ptrdiff_t gesdd( char jobz, fortran_int_t m, fortran_int_t n,
        double* a, fortran_int_t lda, double* s, double* u, fortran_int_t ldu,
        double* vt, fortran_int_t ldvt, double* work, fortran_int_t lwork,
        fortran_int_t* iwork ) {
    fortran_int_t info(0);
    LAPACK_DGESDD( &jobz, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work,
            &lwork, iwork, &info );
    return info;
}

//
// Overloaded function for dispatching to
// * netlib-compatible LAPACK backend (the default), and
// * complex<float> value-type.
//
inline std::ptrdiff_t gesdd( char jobz, fortran_int_t m, fortran_int_t n,
        std::complex<float>* a, fortran_int_t lda, float* s,
        std::complex<float>* u, fortran_int_t ldu, std::complex<float>* vt,
        fortran_int_t ldvt, std::complex<float>* work, fortran_int_t lwork,
        float* rwork, fortran_int_t* iwork ) {
    fortran_int_t info(0);
    LAPACK_CGESDD( &jobz, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work,
            &lwork, rwork, iwork, &info );
    return info;
}

//
// Overloaded function for dispatching to
// * netlib-compatible LAPACK backend (the default), and
// * complex<double> value-type.
//
inline std::ptrdiff_t gesdd( char jobz, fortran_int_t m, fortran_int_t n,
        std::complex<double>* a, fortran_int_t lda, double* s,
        std::complex<double>* u, fortran_int_t ldu, std::complex<double>* vt,
        fortran_int_t ldvt, std::complex<double>* work, fortran_int_t lwork,
        double* rwork, fortran_int_t* iwork ) {
    fortran_int_t info(0);
    LAPACK_ZGESDD( &jobz, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work,
            &lwork, rwork, iwork, &info );
    return info;
}

} // namespace detail

//
// Value-type based template class. Use this class if you need a type
// for dispatching to gesdd.
//
template< typename Value, typename Enable = void >
struct gesdd_impl {};

//
// This implementation is enabled if Value is a real type.
//
template< typename Value >
struct gesdd_impl< Value, typename boost::enable_if< is_real< Value > >::type > {

    typedef Value value_type;
    typedef typename remove_imaginary< Value >::type real_type;
    typedef tag::column_major order;

    //
    // Static member function for user-defined workspaces, that
    // * Deduces the required arguments for dispatching to LAPACK, and
    // * Asserts that most arguments make sense.
    //
    template< typename MatrixA, typename VectorS, typename MatrixU,
            typename MatrixVT, typename WORK, typename IWORK >
    static std::ptrdiff_t invoke( const char jobz, MatrixA& a, VectorS& s,
            MatrixU& u, MatrixVT& vt, detail::workspace2< WORK,
            IWORK > work ) {
        BOOST_STATIC_ASSERT( (boost::is_same< typename remove_const<
                typename value< MatrixA >::type >::type,
                typename remove_const< typename value<
                VectorS >::type >::type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename remove_const<
                typename value< MatrixA >::type >::type,
                typename remove_const< typename value<
                MatrixU >::type >::type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename remove_const<
                typename value< MatrixA >::type >::type,
                typename remove_const< typename value<
                MatrixVT >::type >::type >::value) );
        BOOST_STATIC_ASSERT( (is_mutable< MatrixA >::value) );
        BOOST_STATIC_ASSERT( (is_mutable< VectorS >::value) );
        BOOST_STATIC_ASSERT( (is_mutable< MatrixU >::value) );
        BOOST_STATIC_ASSERT( (is_mutable< MatrixVT >::value) );
        integer_t minmn = std::min< std::ptrdiff_t >( size_row(a),
                size_column(a) );
        BOOST_ASSERT( jobz == 'A' || jobz == 'S' || jobz == 'O' ||
                jobz == 'N' );
        BOOST_ASSERT( size(s) >= std::min< std::ptrdiff_t >(size_row(a),
                size_column(a)) );
        BOOST_ASSERT( size(work.select(fortran_int_t())) >=
                min_size_iwork( minmn ));
        BOOST_ASSERT( size(work.select(real_type())) >= min_size_work(
                size_row(a), size_column(a), jobz, minmn ));
        BOOST_ASSERT( size_column(a) >= 0 );
        BOOST_ASSERT( size_minor(a) == 1 || stride_minor(a) == 1 );
        BOOST_ASSERT( size_minor(u) == 1 || stride_minor(u) == 1 );
        BOOST_ASSERT( size_minor(vt) == 1 || stride_minor(vt) == 1 );
        BOOST_ASSERT( size_row(a) >= 0 );
        BOOST_ASSERT( stride_major(a) >= std::max< std::ptrdiff_t >(1,
                size_row(a)) );
        return detail::gesdd( jobz, size_row(a), size_column(a),
                begin_value(a), stride_major(a), begin_value(s),
                begin_value(u), stride_major(u), begin_value(vt),
                stride_major(vt), begin_value(work.select(real_type())),
                size(work.select(real_type())),
                begin_value(work.select(fortran_int_t())) );
    }

    //
    // Static member function that
    // * Figures out the minimal workspace requirements, and passes
    //   the results to the user-defined workspace overload of the 
    //   invoke static member function
    // * Enables the unblocked algorithm (BLAS level 2)
    //
    template< typename MatrixA, typename VectorS, typename MatrixU,
            typename MatrixVT >
    static std::ptrdiff_t invoke( const char jobz, MatrixA& a, VectorS& s,
            MatrixU& u, MatrixVT& vt, minimal_workspace work ) {
        integer_t minmn = std::min< std::ptrdiff_t >( size_row(a),
                size_column(a) );
        bindings::detail::array< real_type > tmp_work( min_size_work(
                size_row(a), size_column(a), jobz, minmn ) );
        bindings::detail::array< fortran_int_t > tmp_iwork(
                min_size_iwork( minmn ) );
        return invoke( jobz, a, s, u, vt, workspace( tmp_work, tmp_iwork ) );
    }

    //
    // Static member function that
    // * Figures out the optimal workspace requirements, and passes
    //   the results to the user-defined workspace overload of the 
    //   invoke static member
    // * Enables the blocked algorithm (BLAS level 3)
    //
    template< typename MatrixA, typename VectorS, typename MatrixU,
            typename MatrixVT >
    static std::ptrdiff_t invoke( const char jobz, MatrixA& a, VectorS& s,
            MatrixU& u, MatrixVT& vt, optimal_workspace work ) {
        return invoke( jobz, a, s, u, vt, minimal_workspace() );
    }

    //
    // Static member function that returns the minimum size of
    // workspace-array work.
    //
    static std::ptrdiff_t min_size_work( const std::ptrdiff_t m,
            const std::ptrdiff_t n, const char jobz,
            const std::ptrdiff_t minmn ) {
        if ( n == 0 ) return 1;
        if ( jobz == 'N' ) return 3*minmn + std::max<
                std::ptrdiff_t >( std::max< std::ptrdiff_t >(m,n), 7*minmn );
        if ( jobz == 'O' ) return 3*minmn*minmn + std::max<
                std::ptrdiff_t >( std::max< std::ptrdiff_t >( m,n ),
                5*minmn*minmn + 4*minmn );
        return 3*minmn*minmn + std::max< std::ptrdiff_t >( std::max<
                std::ptrdiff_t >( m,n ), 4*minmn*minmn + 4*minmn );
    }

    //
    // Static member function that returns the minimum size of
    // workspace-array iwork.
    //
    static std::ptrdiff_t min_size_iwork( const std::ptrdiff_t minmn ) {
            return 8*minmn;
    }
};

//
// This implementation is enabled if Value is a complex type.
//
template< typename Value >
struct gesdd_impl< Value, typename boost::enable_if< is_complex< Value > >::type > {

    typedef Value value_type;
    typedef typename remove_imaginary< Value >::type real_type;
    typedef tag::column_major order;

    //
    // Static member function for user-defined workspaces, that
    // * Deduces the required arguments for dispatching to LAPACK, and
    // * Asserts that most arguments make sense.
    //
    template< typename MatrixA, typename VectorS, typename MatrixU,
            typename MatrixVT, typename WORK, typename RWORK, typename IWORK >
    static std::ptrdiff_t invoke( const char jobz, MatrixA& a, VectorS& s,
            MatrixU& u, MatrixVT& vt, detail::workspace3< WORK, RWORK,
            IWORK > work ) {
        BOOST_STATIC_ASSERT( (boost::is_same< typename remove_const<
                typename value< MatrixA >::type >::type,
                typename remove_const< typename value<
                MatrixU >::type >::type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename remove_const<
                typename value< MatrixA >::type >::type,
                typename remove_const< typename value<
                MatrixVT >::type >::type >::value) );
        BOOST_STATIC_ASSERT( (is_mutable< MatrixA >::value) );
        BOOST_STATIC_ASSERT( (is_mutable< VectorS >::value) );
        BOOST_STATIC_ASSERT( (is_mutable< MatrixU >::value) );
        BOOST_STATIC_ASSERT( (is_mutable< MatrixVT >::value) );
        integer_t minmn = std::min< std::ptrdiff_t >( size_row(a),
                size_column(a) );
        BOOST_ASSERT( jobz == 'A' || jobz == 'S' || jobz == 'O' ||
                jobz == 'N' );
        BOOST_ASSERT( size(s) >= std::min< std::ptrdiff_t >(size_row(a),
                size_column(a)) );
        BOOST_ASSERT( size(work.select(fortran_int_t())) >=
                min_size_iwork( minmn ));
        BOOST_ASSERT( size(work.select(real_type())) >= min_size_rwork( minmn,
                jobz ));
        BOOST_ASSERT( size(work.select(value_type())) >= min_size_work(
                size_row(a), size_column(a), jobz, minmn ));
        BOOST_ASSERT( size_column(a) >= 0 );
        BOOST_ASSERT( size_minor(a) == 1 || stride_minor(a) == 1 );
        BOOST_ASSERT( size_minor(u) == 1 || stride_minor(u) == 1 );
        BOOST_ASSERT( size_minor(vt) == 1 || stride_minor(vt) == 1 );
        BOOST_ASSERT( size_row(a) >= 0 );
        BOOST_ASSERT( stride_major(a) >= std::max< std::ptrdiff_t >(1,
                size_row(a)) );
        return detail::gesdd( jobz, size_row(a), size_column(a),
                begin_value(a), stride_major(a), begin_value(s),
                begin_value(u), stride_major(u), begin_value(vt),
                stride_major(vt), begin_value(work.select(value_type())),
                size(work.select(value_type())),
                begin_value(work.select(real_type())),
                begin_value(work.select(fortran_int_t())) );
    }

    //
    // Static member function that
    // * Figures out the minimal workspace requirements, and passes
    //   the results to the user-defined workspace overload of the 
    //   invoke static member function
    // * Enables the unblocked algorithm (BLAS level 2)
    //
    template< typename MatrixA, typename VectorS, typename MatrixU,
            typename MatrixVT >
    static std::ptrdiff_t invoke( const char jobz, MatrixA& a, VectorS& s,
            MatrixU& u, MatrixVT& vt, minimal_workspace work ) {
        integer_t minmn = std::min< std::ptrdiff_t >( size_row(a),
                size_column(a) );
        bindings::detail::array< value_type > tmp_work( min_size_work(
                size_row(a), size_column(a), jobz, minmn ) );
        bindings::detail::array< real_type > tmp_rwork( min_size_rwork( minmn,
                jobz ) );
        bindings::detail::array< fortran_int_t > tmp_iwork(
                min_size_iwork( minmn ) );
        return invoke( jobz, a, s, u, vt, workspace( tmp_work, tmp_rwork,
                tmp_iwork ) );
    }

    //
    // Static member function that
    // * Figures out the optimal workspace requirements, and passes
    //   the results to the user-defined workspace overload of the 
    //   invoke static member
    // * Enables the blocked algorithm (BLAS level 3)
    //
    template< typename MatrixA, typename VectorS, typename MatrixU,
            typename MatrixVT >
    static std::ptrdiff_t invoke( const char jobz, MatrixA& a, VectorS& s,
            MatrixU& u, MatrixVT& vt, optimal_workspace work ) {
        integer_t minmn = std::min< std::ptrdiff_t >( size_row(a),
                size_column(a) );
        value_type opt_size_work;
        bindings::detail::array< real_type > tmp_rwork( min_size_rwork( minmn,
                jobz ) );
        bindings::detail::array< fortran_int_t > tmp_iwork(
                min_size_iwork( minmn ) );
        detail::gesdd( jobz, size_row(a), size_column(a), begin_value(a),
                stride_major(a), begin_value(s), begin_value(u),
                stride_major(u), begin_value(vt), stride_major(vt),
                &opt_size_work, -1, begin_value(tmp_rwork),
                begin_value(tmp_iwork) );
        bindings::detail::array< value_type > tmp_work(
                traits::detail::to_int( opt_size_work ) );
        return invoke( jobz, a, s, u, vt, workspace( tmp_work, tmp_rwork,
                tmp_iwork ) );
    }

    //
    // Static member function that returns the minimum size of
    // workspace-array work.
    //
    static std::ptrdiff_t min_size_work( const std::ptrdiff_t m,
            const std::ptrdiff_t n, const char jobz,
            const std::ptrdiff_t minmn ) {
        if ( n == 0 ) return 1;
        if ( jobz == 'N' ) return 2*minmn + std::max< std::ptrdiff_t >( m,n );
        if ( jobz == 'O' ) return 2*(minmn*minmn + minmn) + std::max<
                std::ptrdiff_t >( m, n );
        return minmn*minmn + 2*minmn + std::max< std::ptrdiff_t >( m, n );
    }

    //
    // Static member function that returns the minimum size of
    // workspace-array rwork.
    //
    static std::ptrdiff_t min_size_rwork( const std::ptrdiff_t minmn,
            const char jobz ) {
        if ( jobz == 'N' ) return 5*minmn;
        return 5*minmn*minmn + 7*minmn;
    }

    //
    // Static member function that returns the minimum size of
    // workspace-array iwork.
    //
    static std::ptrdiff_t min_size_iwork( const std::ptrdiff_t minmn ) {
            return 8*minmn;
    }
};


//
// Functions for direct use. These functions are overloaded for temporaries,
// so that wrapped types can still be passed and used for write-access. In
// addition, if applicable, they are overloaded for user-defined workspaces.
// Calls to these functions are passed to the gesdd_impl classes. In the 
// documentation, most overloads are collapsed to avoid a large number of
// prototypes which are very similar.
//

//
// Overloaded function for gesdd. Its overload differs for
// * MatrixA&
// * VectorS&
// * MatrixU&
// * MatrixVT&
// * User-defined workspace
//
template< typename MatrixA, typename VectorS, typename MatrixU,
        typename MatrixVT, typename Workspace >
inline std::ptrdiff_t gesdd( const char jobz, MatrixA& a, VectorS& s,
        MatrixU& u, MatrixVT& vt, Workspace work ) {
    return gesdd_impl< typename value< MatrixA >::type >::invoke( jobz,
            a, s, u, vt, work );
}

//
// Overloaded function for gesdd. Its overload differs for
// * MatrixA&
// * VectorS&
// * MatrixU&
// * MatrixVT&
// * Default workspace-type (optimal)
//
template< typename MatrixA, typename VectorS, typename MatrixU,
        typename MatrixVT >
inline std::ptrdiff_t gesdd( const char jobz, MatrixA& a, VectorS& s,
        MatrixU& u, MatrixVT& vt ) {
    return gesdd_impl< typename value< MatrixA >::type >::invoke( jobz,
            a, s, u, vt, optimal_workspace() );
}

//
// Overloaded function for gesdd. Its overload differs for
// * const MatrixA&
// * VectorS&
// * MatrixU&
// * MatrixVT&
// * User-defined workspace
//
template< typename MatrixA, typename VectorS, typename MatrixU,
        typename MatrixVT, typename Workspace >
inline std::ptrdiff_t gesdd( const char jobz, const MatrixA& a,
        VectorS& s, MatrixU& u, MatrixVT& vt, Workspace work ) {
    return gesdd_impl< typename value< MatrixA >::type >::invoke( jobz,
            a, s, u, vt, work );
}

//
// Overloaded function for gesdd. Its overload differs for
// * const MatrixA&
// * VectorS&
// * MatrixU&
// * MatrixVT&
// * Default workspace-type (optimal)
//
template< typename MatrixA, typename VectorS, typename MatrixU,
        typename MatrixVT >
inline std::ptrdiff_t gesdd( const char jobz, const MatrixA& a,
        VectorS& s, MatrixU& u, MatrixVT& vt ) {
    return gesdd_impl< typename value< MatrixA >::type >::invoke( jobz,
            a, s, u, vt, optimal_workspace() );
}

//
// Overloaded function for gesdd. Its overload differs for
// * MatrixA&
// * const VectorS&
// * MatrixU&
// * MatrixVT&
// * User-defined workspace
//
template< typename MatrixA, typename VectorS, typename MatrixU,
        typename MatrixVT, typename Workspace >
inline std::ptrdiff_t gesdd( const char jobz, MatrixA& a,
        const VectorS& s, MatrixU& u, MatrixVT& vt, Workspace work ) {
    return gesdd_impl< typename value< MatrixA >::type >::invoke( jobz,
            a, s, u, vt, work );
}

//
// Overloaded function for gesdd. Its overload differs for
// * MatrixA&
// * const VectorS&
// * MatrixU&
// * MatrixVT&
// * Default workspace-type (optimal)
//
template< typename MatrixA, typename VectorS, typename MatrixU,
        typename MatrixVT >
inline std::ptrdiff_t gesdd( const char jobz, MatrixA& a,
        const VectorS& s, MatrixU& u, MatrixVT& vt ) {
    return gesdd_impl< typename value< MatrixA >::type >::invoke( jobz,
            a, s, u, vt, optimal_workspace() );
}

//
// Overloaded function for gesdd. Its overload differs for
// * const MatrixA&
// * const VectorS&
// * MatrixU&
// * MatrixVT&
// * User-defined workspace
//
template< typename MatrixA, typename VectorS, typename MatrixU,
        typename MatrixVT, typename Workspace >
inline std::ptrdiff_t gesdd( const char jobz, const MatrixA& a,
        const VectorS& s, MatrixU& u, MatrixVT& vt, Workspace work ) {
    return gesdd_impl< typename value< MatrixA >::type >::invoke( jobz,
            a, s, u, vt, work );
}

//
// Overloaded function for gesdd. Its overload differs for
// * const MatrixA&
// * const VectorS&
// * MatrixU&
// * MatrixVT&
// * Default workspace-type (optimal)
//
template< typename MatrixA, typename VectorS, typename MatrixU,
        typename MatrixVT >
inline std::ptrdiff_t gesdd( const char jobz, const MatrixA& a,
        const VectorS& s, MatrixU& u, MatrixVT& vt ) {
    return gesdd_impl< typename value< MatrixA >::type >::invoke( jobz,
            a, s, u, vt, optimal_workspace() );
}

//
// Overloaded function for gesdd. Its overload differs for
// * MatrixA&
// * VectorS&
// * const MatrixU&
// * MatrixVT&
// * User-defined workspace
//
template< typename MatrixA, typename VectorS, typename MatrixU,
        typename MatrixVT, typename Workspace >
inline std::ptrdiff_t gesdd( const char jobz, MatrixA& a, VectorS& s,
        const MatrixU& u, MatrixVT& vt, Workspace work ) {
    return gesdd_impl< typename value< MatrixA >::type >::invoke( jobz,
            a, s, u, vt, work );
}

//
// Overloaded function for gesdd. Its overload differs for
// * MatrixA&
// * VectorS&
// * const MatrixU&
// * MatrixVT&
// * Default workspace-type (optimal)
//
template< typename MatrixA, typename VectorS, typename MatrixU,
        typename MatrixVT >
inline std::ptrdiff_t gesdd( const char jobz, MatrixA& a, VectorS& s,
        const MatrixU& u, MatrixVT& vt ) {
    return gesdd_impl< typename value< MatrixA >::type >::invoke( jobz,
            a, s, u, vt, optimal_workspace() );
}

//
// Overloaded function for gesdd. Its overload differs for
// * const MatrixA&
// * VectorS&
// * const MatrixU&
// * MatrixVT&
// * User-defined workspace
//
template< typename MatrixA, typename VectorS, typename MatrixU,
        typename MatrixVT, typename Workspace >
inline std::ptrdiff_t gesdd( const char jobz, const MatrixA& a,
        VectorS& s, const MatrixU& u, MatrixVT& vt, Workspace work ) {
    return gesdd_impl< typename value< MatrixA >::type >::invoke( jobz,
            a, s, u, vt, work );
}

//
// Overloaded function for gesdd. Its overload differs for
// * const MatrixA&
// * VectorS&
// * const MatrixU&
// * MatrixVT&
// * Default workspace-type (optimal)
//
template< typename MatrixA, typename VectorS, typename MatrixU,
        typename MatrixVT >
inline std::ptrdiff_t gesdd( const char jobz, const MatrixA& a,
        VectorS& s, const MatrixU& u, MatrixVT& vt ) {
    return gesdd_impl< typename value< MatrixA >::type >::invoke( jobz,
            a, s, u, vt, optimal_workspace() );
}

//
// Overloaded function for gesdd. Its overload differs for
// * MatrixA&
// * const VectorS&
// * const MatrixU&
// * MatrixVT&
// * User-defined workspace
//
template< typename MatrixA, typename VectorS, typename MatrixU,
        typename MatrixVT, typename Workspace >
inline std::ptrdiff_t gesdd( const char jobz, MatrixA& a,
        const VectorS& s, const MatrixU& u, MatrixVT& vt, Workspace work ) {
    return gesdd_impl< typename value< MatrixA >::type >::invoke( jobz,
            a, s, u, vt, work );
}

//
// Overloaded function for gesdd. Its overload differs for
// * MatrixA&
// * const VectorS&
// * const MatrixU&
// * MatrixVT&
// * Default workspace-type (optimal)
//
template< typename MatrixA, typename VectorS, typename MatrixU,
        typename MatrixVT >
inline std::ptrdiff_t gesdd( const char jobz, MatrixA& a,
        const VectorS& s, const MatrixU& u, MatrixVT& vt ) {
    return gesdd_impl< typename value< MatrixA >::type >::invoke( jobz,
            a, s, u, vt, optimal_workspace() );
}

//
// Overloaded function for gesdd. Its overload differs for
// * const MatrixA&
// * const VectorS&
// * const MatrixU&
// * MatrixVT&
// * User-defined workspace
//
template< typename MatrixA, typename VectorS, typename MatrixU,
        typename MatrixVT, typename Workspace >
inline std::ptrdiff_t gesdd( const char jobz, const MatrixA& a,
        const VectorS& s, const MatrixU& u, MatrixVT& vt, Workspace work ) {
    return gesdd_impl< typename value< MatrixA >::type >::invoke( jobz,
            a, s, u, vt, work );
}

//
// Overloaded function for gesdd. Its overload differs for
// * const MatrixA&
// * const VectorS&
// * const MatrixU&
// * MatrixVT&
// * Default workspace-type (optimal)
//
template< typename MatrixA, typename VectorS, typename MatrixU,
        typename MatrixVT >
inline std::ptrdiff_t gesdd( const char jobz, const MatrixA& a,
        const VectorS& s, const MatrixU& u, MatrixVT& vt ) {
    return gesdd_impl< typename value< MatrixA >::type >::invoke( jobz,
            a, s, u, vt, optimal_workspace() );
}

//
// Overloaded function for gesdd. Its overload differs for
// * MatrixA&
// * VectorS&
// * MatrixU&
// * const MatrixVT&
// * User-defined workspace
//
template< typename MatrixA, typename VectorS, typename MatrixU,
        typename MatrixVT, typename Workspace >
inline std::ptrdiff_t gesdd( const char jobz, MatrixA& a, VectorS& s,
        MatrixU& u, const MatrixVT& vt, Workspace work ) {
    return gesdd_impl< typename value< MatrixA >::type >::invoke( jobz,
            a, s, u, vt, work );
}

//
// Overloaded function for gesdd. Its overload differs for
// * MatrixA&
// * VectorS&
// * MatrixU&
// * const MatrixVT&
// * Default workspace-type (optimal)
//
template< typename MatrixA, typename VectorS, typename MatrixU,
        typename MatrixVT >
inline std::ptrdiff_t gesdd( const char jobz, MatrixA& a, VectorS& s,
        MatrixU& u, const MatrixVT& vt ) {
    return gesdd_impl< typename value< MatrixA >::type >::invoke( jobz,
            a, s, u, vt, optimal_workspace() );
}

//
// Overloaded function for gesdd. Its overload differs for
// * const MatrixA&
// * VectorS&
// * MatrixU&
// * const MatrixVT&
// * User-defined workspace
//
template< typename MatrixA, typename VectorS, typename MatrixU,
        typename MatrixVT, typename Workspace >
inline std::ptrdiff_t gesdd( const char jobz, const MatrixA& a,
        VectorS& s, MatrixU& u, const MatrixVT& vt, Workspace work ) {
    return gesdd_impl< typename value< MatrixA >::type >::invoke( jobz,
            a, s, u, vt, work );
}

//
// Overloaded function for gesdd. Its overload differs for
// * const MatrixA&
// * VectorS&
// * MatrixU&
// * const MatrixVT&
// * Default workspace-type (optimal)
//
template< typename MatrixA, typename VectorS, typename MatrixU,
        typename MatrixVT >
inline std::ptrdiff_t gesdd( const char jobz, const MatrixA& a,
        VectorS& s, MatrixU& u, const MatrixVT& vt ) {
    return gesdd_impl< typename value< MatrixA >::type >::invoke( jobz,
            a, s, u, vt, optimal_workspace() );
}

//
// Overloaded function for gesdd. Its overload differs for
// * MatrixA&
// * const VectorS&
// * MatrixU&
// * const MatrixVT&
// * User-defined workspace
//
template< typename MatrixA, typename VectorS, typename MatrixU,
        typename MatrixVT, typename Workspace >
inline std::ptrdiff_t gesdd( const char jobz, MatrixA& a,
        const VectorS& s, MatrixU& u, const MatrixVT& vt, Workspace work ) {
    return gesdd_impl< typename value< MatrixA >::type >::invoke( jobz,
            a, s, u, vt, work );
}

//
// Overloaded function for gesdd. Its overload differs for
// * MatrixA&
// * const VectorS&
// * MatrixU&
// * const MatrixVT&
// * Default workspace-type (optimal)
//
template< typename MatrixA, typename VectorS, typename MatrixU,
        typename MatrixVT >
inline std::ptrdiff_t gesdd( const char jobz, MatrixA& a,
        const VectorS& s, MatrixU& u, const MatrixVT& vt ) {
    return gesdd_impl< typename value< MatrixA >::type >::invoke( jobz,
            a, s, u, vt, optimal_workspace() );
}

//
// Overloaded function for gesdd. Its overload differs for
// * const MatrixA&
// * const VectorS&
// * MatrixU&
// * const MatrixVT&
// * User-defined workspace
//
template< typename MatrixA, typename VectorS, typename MatrixU,
        typename MatrixVT, typename Workspace >
inline std::ptrdiff_t gesdd( const char jobz, const MatrixA& a,
        const VectorS& s, MatrixU& u, const MatrixVT& vt, Workspace work ) {
    return gesdd_impl< typename value< MatrixA >::type >::invoke( jobz,
            a, s, u, vt, work );
}

//
// Overloaded function for gesdd. Its overload differs for
// * const MatrixA&
// * const VectorS&
// * MatrixU&
// * const MatrixVT&
// * Default workspace-type (optimal)
//
template< typename MatrixA, typename VectorS, typename MatrixU,
        typename MatrixVT >
inline std::ptrdiff_t gesdd( const char jobz, const MatrixA& a,
        const VectorS& s, MatrixU& u, const MatrixVT& vt ) {
    return gesdd_impl< typename value< MatrixA >::type >::invoke( jobz,
            a, s, u, vt, optimal_workspace() );
}

//
// Overloaded function for gesdd. Its overload differs for
// * MatrixA&
// * VectorS&
// * const MatrixU&
// * const MatrixVT&
// * User-defined workspace
//
template< typename MatrixA, typename VectorS, typename MatrixU,
        typename MatrixVT, typename Workspace >
inline std::ptrdiff_t gesdd( const char jobz, MatrixA& a, VectorS& s,
        const MatrixU& u, const MatrixVT& vt, Workspace work ) {
    return gesdd_impl< typename value< MatrixA >::type >::invoke( jobz,
            a, s, u, vt, work );
}

//
// Overloaded function for gesdd. Its overload differs for
// * MatrixA&
// * VectorS&
// * const MatrixU&
// * const MatrixVT&
// * Default workspace-type (optimal)
//
template< typename MatrixA, typename VectorS, typename MatrixU,
        typename MatrixVT >
inline std::ptrdiff_t gesdd( const char jobz, MatrixA& a, VectorS& s,
        const MatrixU& u, const MatrixVT& vt ) {
    return gesdd_impl< typename value< MatrixA >::type >::invoke( jobz,
            a, s, u, vt, optimal_workspace() );
}

//
// Overloaded function for gesdd. Its overload differs for
// * const MatrixA&
// * VectorS&
// * const MatrixU&
// * const MatrixVT&
// * User-defined workspace
//
template< typename MatrixA, typename VectorS, typename MatrixU,
        typename MatrixVT, typename Workspace >
inline std::ptrdiff_t gesdd( const char jobz, const MatrixA& a,
        VectorS& s, const MatrixU& u, const MatrixVT& vt, Workspace work ) {
    return gesdd_impl< typename value< MatrixA >::type >::invoke( jobz,
            a, s, u, vt, work );
}

//
// Overloaded function for gesdd. Its overload differs for
// * const MatrixA&
// * VectorS&
// * const MatrixU&
// * const MatrixVT&
// * Default workspace-type (optimal)
//
template< typename MatrixA, typename VectorS, typename MatrixU,
        typename MatrixVT >
inline std::ptrdiff_t gesdd( const char jobz, const MatrixA& a,
        VectorS& s, const MatrixU& u, const MatrixVT& vt ) {
    return gesdd_impl< typename value< MatrixA >::type >::invoke( jobz,
            a, s, u, vt, optimal_workspace() );
}

//
// Overloaded function for gesdd. Its overload differs for
// * MatrixA&
// * const VectorS&
// * const MatrixU&
// * const MatrixVT&
// * User-defined workspace
//
template< typename MatrixA, typename VectorS, typename MatrixU,
        typename MatrixVT, typename Workspace >
inline std::ptrdiff_t gesdd( const char jobz, MatrixA& a,
        const VectorS& s, const MatrixU& u, const MatrixVT& vt,
        Workspace work ) {
    return gesdd_impl< typename value< MatrixA >::type >::invoke( jobz,
            a, s, u, vt, work );
}

//
// Overloaded function for gesdd. Its overload differs for
// * MatrixA&
// * const VectorS&
// * const MatrixU&
// * const MatrixVT&
// * Default workspace-type (optimal)
//
template< typename MatrixA, typename VectorS, typename MatrixU,
        typename MatrixVT >
inline std::ptrdiff_t gesdd( const char jobz, MatrixA& a,
        const VectorS& s, const MatrixU& u, const MatrixVT& vt ) {
    return gesdd_impl< typename value< MatrixA >::type >::invoke( jobz,
            a, s, u, vt, optimal_workspace() );
}

//
// Overloaded function for gesdd. Its overload differs for
// * const MatrixA&
// * const VectorS&
// * const MatrixU&
// * const MatrixVT&
// * User-defined workspace
//
template< typename MatrixA, typename VectorS, typename MatrixU,
        typename MatrixVT, typename Workspace >
inline std::ptrdiff_t gesdd( const char jobz, const MatrixA& a,
        const VectorS& s, const MatrixU& u, const MatrixVT& vt,
        Workspace work ) {
    return gesdd_impl< typename value< MatrixA >::type >::invoke( jobz,
            a, s, u, vt, work );
}

//
// Overloaded function for gesdd. Its overload differs for
// * const MatrixA&
// * const VectorS&
// * const MatrixU&
// * const MatrixVT&
// * Default workspace-type (optimal)
//
template< typename MatrixA, typename VectorS, typename MatrixU,
        typename MatrixVT >
inline std::ptrdiff_t gesdd( const char jobz, const MatrixA& a,
        const VectorS& s, const MatrixU& u, const MatrixVT& vt ) {
    return gesdd_impl< typename value< MatrixA >::type >::invoke( jobz,
            a, s, u, vt, optimal_workspace() );
}

} // namespace lapack
} // namespace bindings
} // namespace numeric
} // namespace boost

#endif
