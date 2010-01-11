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

#ifndef BOOST_NUMERIC_BINDINGS_LAPACK_DRIVER_GELSS_HPP
#define BOOST_NUMERIC_BINDINGS_LAPACK_DRIVER_GELSS_HPP

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
// The LAPACK-backend for gelss is the netlib-compatible backend.
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
inline std::ptrdiff_t gelss( fortran_int_t m, fortran_int_t n,
        fortran_int_t nrhs, float* a, fortran_int_t lda, float* b,
        fortran_int_t ldb, float* s, float rcond, fortran_int_t& rank,
        float* work, fortran_int_t lwork ) {
    fortran_int_t info(0);
    LAPACK_SGELSS( &m, &n, &nrhs, a, &lda, b, &ldb, s, &rcond, &rank, work,
            &lwork, &info );
    return info;
}

//
// Overloaded function for dispatching to
// * netlib-compatible LAPACK backend (the default), and
// * double value-type.
//
inline std::ptrdiff_t gelss( fortran_int_t m, fortran_int_t n,
        fortran_int_t nrhs, double* a, fortran_int_t lda, double* b,
        fortran_int_t ldb, double* s, double rcond, fortran_int_t& rank,
        double* work, fortran_int_t lwork ) {
    fortran_int_t info(0);
    LAPACK_DGELSS( &m, &n, &nrhs, a, &lda, b, &ldb, s, &rcond, &rank, work,
            &lwork, &info );
    return info;
}

//
// Overloaded function for dispatching to
// * netlib-compatible LAPACK backend (the default), and
// * complex<float> value-type.
//
inline std::ptrdiff_t gelss( fortran_int_t m, fortran_int_t n,
        fortran_int_t nrhs, std::complex<float>* a, fortran_int_t lda,
        std::complex<float>* b, fortran_int_t ldb, float* s, float rcond,
        fortran_int_t& rank, std::complex<float>* work, fortran_int_t lwork,
        float* rwork ) {
    fortran_int_t info(0);
    LAPACK_CGELSS( &m, &n, &nrhs, a, &lda, b, &ldb, s, &rcond, &rank, work,
            &lwork, rwork, &info );
    return info;
}

//
// Overloaded function for dispatching to
// * netlib-compatible LAPACK backend (the default), and
// * complex<double> value-type.
//
inline std::ptrdiff_t gelss( fortran_int_t m, fortran_int_t n,
        fortran_int_t nrhs, std::complex<double>* a, fortran_int_t lda,
        std::complex<double>* b, fortran_int_t ldb, double* s, double rcond,
        fortran_int_t& rank, std::complex<double>* work, fortran_int_t lwork,
        double* rwork ) {
    fortran_int_t info(0);
    LAPACK_ZGELSS( &m, &n, &nrhs, a, &lda, b, &ldb, s, &rcond, &rank, work,
            &lwork, rwork, &info );
    return info;
}

} // namespace detail

//
// Value-type based template class. Use this class if you need a type
// for dispatching to gelss.
//
template< typename Value, typename Enable = void >
struct gelss_impl {};

//
// This implementation is enabled if Value is a real type.
//
template< typename Value >
struct gelss_impl< Value, typename boost::enable_if< is_real< Value > >::type > {

    typedef Value value_type;
    typedef typename remove_imaginary< Value >::type real_type;
    typedef tag::column_major order;

    //
    // Static member function for user-defined workspaces, that
    // * Deduces the required arguments for dispatching to LAPACK, and
    // * Asserts that most arguments make sense.
    //
    template< typename MatrixA, typename MatrixB, typename VectorS,
            typename WORK >
    static std::ptrdiff_t invoke( MatrixA& a, MatrixB& b, VectorS& s,
            const real_type rcond, fortran_int_t& rank,
            detail::workspace1< WORK > work ) {
        BOOST_STATIC_ASSERT( (boost::is_same< typename remove_const<
                typename value< MatrixA >::type >::type,
                typename remove_const< typename value<
                MatrixB >::type >::type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename remove_const<
                typename value< MatrixA >::type >::type,
                typename remove_const< typename value<
                VectorS >::type >::type >::value) );
        BOOST_STATIC_ASSERT( (is_mutable< MatrixA >::value) );
        BOOST_STATIC_ASSERT( (is_mutable< MatrixB >::value) );
        BOOST_STATIC_ASSERT( (is_mutable< VectorS >::value) );
        BOOST_ASSERT( size(s) >= std::min< std::ptrdiff_t >(size_row(a),
                size_column(a)) );
        BOOST_ASSERT( size(work.select(real_type())) >= min_size_work(
                size_row(a), size_column(a), size_column(b) ));
        BOOST_ASSERT( size_column(a) >= 0 );
        BOOST_ASSERT( size_column(b) >= 0 );
        BOOST_ASSERT( size_minor(a) == 1 || stride_minor(a) == 1 );
        BOOST_ASSERT( size_minor(b) == 1 || stride_minor(b) == 1 );
        BOOST_ASSERT( size_row(a) >= 0 );
        BOOST_ASSERT( stride_major(a) >= std::max< std::ptrdiff_t >(1,
                size_row(a)) );
        BOOST_ASSERT( stride_major(b) >= std::max< std::ptrdiff_t >(1,std::max<
                std::ptrdiff_t >(size_row(a),size_column(a))) );
        return detail::gelss( size_row(a), size_column(a), size_column(b),
                begin_value(a), stride_major(a), begin_value(b),
                stride_major(b), begin_value(s), rcond, rank,
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
    template< typename MatrixA, typename MatrixB, typename VectorS >
    static std::ptrdiff_t invoke( MatrixA& a, MatrixB& b, VectorS& s,
            const real_type rcond, fortran_int_t& rank,
            minimal_workspace work ) {
        bindings::detail::array< real_type > tmp_work( min_size_work(
                size_row(a), size_column(a), size_column(b) ) );
        return invoke( a, b, s, rcond, rank, workspace( tmp_work ) );
    }

    //
    // Static member function that
    // * Figures out the optimal workspace requirements, and passes
    //   the results to the user-defined workspace overload of the 
    //   invoke static member
    // * Enables the blocked algorithm (BLAS level 3)
    //
    template< typename MatrixA, typename MatrixB, typename VectorS >
    static std::ptrdiff_t invoke( MatrixA& a, MatrixB& b, VectorS& s,
            const real_type rcond, fortran_int_t& rank,
            optimal_workspace work ) {
        real_type opt_size_work;
        detail::gelss( size_row(a), size_column(a), size_column(b),
                begin_value(a), stride_major(a), begin_value(b),
                stride_major(b), begin_value(s), rcond, rank, &opt_size_work,
                -1 );
        bindings::detail::array< real_type > tmp_work(
                traits::detail::to_int( opt_size_work ) );
        return invoke( a, b, s, rcond, rank, workspace( tmp_work ) );
    }

    //
    // Static member function that returns the minimum size of
    // workspace-array work.
    //
    static std::ptrdiff_t min_size_work( const std::ptrdiff_t m,
            const std::ptrdiff_t n, const std::ptrdiff_t nrhs ) {
        std::ptrdiff_t minmn = std::min< std::ptrdiff_t >( m, n );
        return std::max< std::ptrdiff_t >( 1, 3*minmn + std::max<
                std::ptrdiff_t >( std::max< std::ptrdiff_t >( 2*minmn, std::max<
                std::ptrdiff_t >(m,n) ), nrhs ) );
    }
};

//
// This implementation is enabled if Value is a complex type.
//
template< typename Value >
struct gelss_impl< Value, typename boost::enable_if< is_complex< Value > >::type > {

    typedef Value value_type;
    typedef typename remove_imaginary< Value >::type real_type;
    typedef tag::column_major order;

    //
    // Static member function for user-defined workspaces, that
    // * Deduces the required arguments for dispatching to LAPACK, and
    // * Asserts that most arguments make sense.
    //
    template< typename MatrixA, typename MatrixB, typename VectorS,
            typename WORK, typename RWORK >
    static std::ptrdiff_t invoke( MatrixA& a, MatrixB& b, VectorS& s,
            const real_type rcond, fortran_int_t& rank,
            detail::workspace2< WORK, RWORK > work ) {
        BOOST_STATIC_ASSERT( (boost::is_same< typename remove_const<
                typename value< MatrixA >::type >::type,
                typename remove_const< typename value<
                MatrixB >::type >::type >::value) );
        BOOST_STATIC_ASSERT( (is_mutable< MatrixA >::value) );
        BOOST_STATIC_ASSERT( (is_mutable< MatrixB >::value) );
        BOOST_STATIC_ASSERT( (is_mutable< VectorS >::value) );
        std::ptrdiff_t minmn = std::min< std::ptrdiff_t >( size_row(a),
                size_column(a) );
        BOOST_ASSERT( size(s) >= std::min< std::ptrdiff_t >(size_row(a),
                size_column(a)) );
        BOOST_ASSERT( size(work.select(real_type())) >= min_size_rwork(
                minmn ));
        BOOST_ASSERT( size(work.select(value_type())) >= min_size_work(
                size_row(a), size_column(a), size_column(b), minmn ));
        BOOST_ASSERT( size_column(a) >= 0 );
        BOOST_ASSERT( size_column(b) >= 0 );
        BOOST_ASSERT( size_minor(a) == 1 || stride_minor(a) == 1 );
        BOOST_ASSERT( size_minor(b) == 1 || stride_minor(b) == 1 );
        BOOST_ASSERT( size_row(a) >= 0 );
        BOOST_ASSERT( stride_major(a) >= std::max< std::ptrdiff_t >(1,
                size_row(a)) );
        BOOST_ASSERT( stride_major(b) >= std::max< std::ptrdiff_t >(1,std::max<
                std::ptrdiff_t >(size_row(a),size_column(a))) );
        return detail::gelss( size_row(a), size_column(a), size_column(b),
                begin_value(a), stride_major(a), begin_value(b),
                stride_major(b), begin_value(s), rcond, rank,
                begin_value(work.select(value_type())),
                size(work.select(value_type())),
                begin_value(work.select(real_type())) );
    }

    //
    // Static member function that
    // * Figures out the minimal workspace requirements, and passes
    //   the results to the user-defined workspace overload of the 
    //   invoke static member function
    // * Enables the unblocked algorithm (BLAS level 2)
    //
    template< typename MatrixA, typename MatrixB, typename VectorS >
    static std::ptrdiff_t invoke( MatrixA& a, MatrixB& b, VectorS& s,
            const real_type rcond, fortran_int_t& rank,
            minimal_workspace work ) {
        std::ptrdiff_t minmn = std::min< std::ptrdiff_t >( size_row(a),
                size_column(a) );
        bindings::detail::array< value_type > tmp_work( min_size_work(
                size_row(a), size_column(a), size_column(b), minmn ) );
        bindings::detail::array< real_type > tmp_rwork( min_size_rwork(
                minmn ) );
        return invoke( a, b, s, rcond, rank, workspace( tmp_work,
                tmp_rwork ) );
    }

    //
    // Static member function that
    // * Figures out the optimal workspace requirements, and passes
    //   the results to the user-defined workspace overload of the 
    //   invoke static member
    // * Enables the blocked algorithm (BLAS level 3)
    //
    template< typename MatrixA, typename MatrixB, typename VectorS >
    static std::ptrdiff_t invoke( MatrixA& a, MatrixB& b, VectorS& s,
            const real_type rcond, fortran_int_t& rank,
            optimal_workspace work ) {
        std::ptrdiff_t minmn = std::min< std::ptrdiff_t >( size_row(a),
                size_column(a) );
        value_type opt_size_work;
        bindings::detail::array< real_type > tmp_rwork( min_size_rwork(
                minmn ) );
        detail::gelss( size_row(a), size_column(a), size_column(b),
                begin_value(a), stride_major(a), begin_value(b),
                stride_major(b), begin_value(s), rcond, rank, &opt_size_work,
                -1, begin_value(tmp_rwork) );
        bindings::detail::array< value_type > tmp_work(
                traits::detail::to_int( opt_size_work ) );
        return invoke( a, b, s, rcond, rank, workspace( tmp_work,
                tmp_rwork ) );
    }

    //
    // Static member function that returns the minimum size of
    // workspace-array work.
    //
    static std::ptrdiff_t min_size_work( const std::ptrdiff_t m,
            const std::ptrdiff_t n, const std::ptrdiff_t nrhs,
            const std::ptrdiff_t minmn ) {
        return std::max< std::ptrdiff_t >( 1, 2*minmn + std::max<
                std::ptrdiff_t >( std::max< std::ptrdiff_t >( m,n ), nrhs ) );
    }

    //
    // Static member function that returns the minimum size of
    // workspace-array rwork.
    //
    static std::ptrdiff_t min_size_rwork( const std::ptrdiff_t minmn ) {
        return 5*minmn;
    }
};


//
// Functions for direct use. These functions are overloaded for temporaries,
// so that wrapped types can still be passed and used for write-access. In
// addition, if applicable, they are overloaded for user-defined workspaces.
// Calls to these functions are passed to the gelss_impl classes. In the 
// documentation, most overloads are collapsed to avoid a large number of
// prototypes which are very similar.
//

//
// Overloaded function for gelss. Its overload differs for
// * MatrixA&
// * MatrixB&
// * VectorS&
// * User-defined workspace
//
template< typename MatrixA, typename MatrixB, typename VectorS,
        typename Workspace >
inline std::ptrdiff_t gelss( MatrixA& a, MatrixB& b, VectorS& s,
        const typename remove_imaginary< typename value<
        MatrixA >::type >::type rcond, fortran_int_t& rank,
        Workspace work ) {
    return gelss_impl< typename value< MatrixA >::type >::invoke( a, b,
            s, rcond, rank, work );
}

//
// Overloaded function for gelss. Its overload differs for
// * MatrixA&
// * MatrixB&
// * VectorS&
// * Default workspace-type (optimal)
//
template< typename MatrixA, typename MatrixB, typename VectorS >
inline std::ptrdiff_t gelss( MatrixA& a, MatrixB& b, VectorS& s,
        const typename remove_imaginary< typename value<
        MatrixA >::type >::type rcond, fortran_int_t& rank ) {
    return gelss_impl< typename value< MatrixA >::type >::invoke( a, b,
            s, rcond, rank, optimal_workspace() );
}

//
// Overloaded function for gelss. Its overload differs for
// * const MatrixA&
// * MatrixB&
// * VectorS&
// * User-defined workspace
//
template< typename MatrixA, typename MatrixB, typename VectorS,
        typename Workspace >
inline std::ptrdiff_t gelss( const MatrixA& a, MatrixB& b, VectorS& s,
        const typename remove_imaginary< typename value<
        MatrixA >::type >::type rcond, fortran_int_t& rank,
        Workspace work ) {
    return gelss_impl< typename value< MatrixA >::type >::invoke( a, b,
            s, rcond, rank, work );
}

//
// Overloaded function for gelss. Its overload differs for
// * const MatrixA&
// * MatrixB&
// * VectorS&
// * Default workspace-type (optimal)
//
template< typename MatrixA, typename MatrixB, typename VectorS >
inline std::ptrdiff_t gelss( const MatrixA& a, MatrixB& b, VectorS& s,
        const typename remove_imaginary< typename value<
        MatrixA >::type >::type rcond, fortran_int_t& rank ) {
    return gelss_impl< typename value< MatrixA >::type >::invoke( a, b,
            s, rcond, rank, optimal_workspace() );
}

//
// Overloaded function for gelss. Its overload differs for
// * MatrixA&
// * const MatrixB&
// * VectorS&
// * User-defined workspace
//
template< typename MatrixA, typename MatrixB, typename VectorS,
        typename Workspace >
inline std::ptrdiff_t gelss( MatrixA& a, const MatrixB& b, VectorS& s,
        const typename remove_imaginary< typename value<
        MatrixA >::type >::type rcond, fortran_int_t& rank,
        Workspace work ) {
    return gelss_impl< typename value< MatrixA >::type >::invoke( a, b,
            s, rcond, rank, work );
}

//
// Overloaded function for gelss. Its overload differs for
// * MatrixA&
// * const MatrixB&
// * VectorS&
// * Default workspace-type (optimal)
//
template< typename MatrixA, typename MatrixB, typename VectorS >
inline std::ptrdiff_t gelss( MatrixA& a, const MatrixB& b, VectorS& s,
        const typename remove_imaginary< typename value<
        MatrixA >::type >::type rcond, fortran_int_t& rank ) {
    return gelss_impl< typename value< MatrixA >::type >::invoke( a, b,
            s, rcond, rank, optimal_workspace() );
}

//
// Overloaded function for gelss. Its overload differs for
// * const MatrixA&
// * const MatrixB&
// * VectorS&
// * User-defined workspace
//
template< typename MatrixA, typename MatrixB, typename VectorS,
        typename Workspace >
inline std::ptrdiff_t gelss( const MatrixA& a, const MatrixB& b,
        VectorS& s, const typename remove_imaginary< typename value<
        MatrixA >::type >::type rcond, fortran_int_t& rank,
        Workspace work ) {
    return gelss_impl< typename value< MatrixA >::type >::invoke( a, b,
            s, rcond, rank, work );
}

//
// Overloaded function for gelss. Its overload differs for
// * const MatrixA&
// * const MatrixB&
// * VectorS&
// * Default workspace-type (optimal)
//
template< typename MatrixA, typename MatrixB, typename VectorS >
inline std::ptrdiff_t gelss( const MatrixA& a, const MatrixB& b,
        VectorS& s, const typename remove_imaginary< typename value<
        MatrixA >::type >::type rcond, fortran_int_t& rank ) {
    return gelss_impl< typename value< MatrixA >::type >::invoke( a, b,
            s, rcond, rank, optimal_workspace() );
}

//
// Overloaded function for gelss. Its overload differs for
// * MatrixA&
// * MatrixB&
// * const VectorS&
// * User-defined workspace
//
template< typename MatrixA, typename MatrixB, typename VectorS,
        typename Workspace >
inline std::ptrdiff_t gelss( MatrixA& a, MatrixB& b, const VectorS& s,
        const typename remove_imaginary< typename value<
        MatrixA >::type >::type rcond, fortran_int_t& rank,
        Workspace work ) {
    return gelss_impl< typename value< MatrixA >::type >::invoke( a, b,
            s, rcond, rank, work );
}

//
// Overloaded function for gelss. Its overload differs for
// * MatrixA&
// * MatrixB&
// * const VectorS&
// * Default workspace-type (optimal)
//
template< typename MatrixA, typename MatrixB, typename VectorS >
inline std::ptrdiff_t gelss( MatrixA& a, MatrixB& b, const VectorS& s,
        const typename remove_imaginary< typename value<
        MatrixA >::type >::type rcond, fortran_int_t& rank ) {
    return gelss_impl< typename value< MatrixA >::type >::invoke( a, b,
            s, rcond, rank, optimal_workspace() );
}

//
// Overloaded function for gelss. Its overload differs for
// * const MatrixA&
// * MatrixB&
// * const VectorS&
// * User-defined workspace
//
template< typename MatrixA, typename MatrixB, typename VectorS,
        typename Workspace >
inline std::ptrdiff_t gelss( const MatrixA& a, MatrixB& b,
        const VectorS& s, const typename remove_imaginary< typename value<
        MatrixA >::type >::type rcond, fortran_int_t& rank,
        Workspace work ) {
    return gelss_impl< typename value< MatrixA >::type >::invoke( a, b,
            s, rcond, rank, work );
}

//
// Overloaded function for gelss. Its overload differs for
// * const MatrixA&
// * MatrixB&
// * const VectorS&
// * Default workspace-type (optimal)
//
template< typename MatrixA, typename MatrixB, typename VectorS >
inline std::ptrdiff_t gelss( const MatrixA& a, MatrixB& b,
        const VectorS& s, const typename remove_imaginary< typename value<
        MatrixA >::type >::type rcond, fortran_int_t& rank ) {
    return gelss_impl< typename value< MatrixA >::type >::invoke( a, b,
            s, rcond, rank, optimal_workspace() );
}

//
// Overloaded function for gelss. Its overload differs for
// * MatrixA&
// * const MatrixB&
// * const VectorS&
// * User-defined workspace
//
template< typename MatrixA, typename MatrixB, typename VectorS,
        typename Workspace >
inline std::ptrdiff_t gelss( MatrixA& a, const MatrixB& b,
        const VectorS& s, const typename remove_imaginary< typename value<
        MatrixA >::type >::type rcond, fortran_int_t& rank,
        Workspace work ) {
    return gelss_impl< typename value< MatrixA >::type >::invoke( a, b,
            s, rcond, rank, work );
}

//
// Overloaded function for gelss. Its overload differs for
// * MatrixA&
// * const MatrixB&
// * const VectorS&
// * Default workspace-type (optimal)
//
template< typename MatrixA, typename MatrixB, typename VectorS >
inline std::ptrdiff_t gelss( MatrixA& a, const MatrixB& b,
        const VectorS& s, const typename remove_imaginary< typename value<
        MatrixA >::type >::type rcond, fortran_int_t& rank ) {
    return gelss_impl< typename value< MatrixA >::type >::invoke( a, b,
            s, rcond, rank, optimal_workspace() );
}

//
// Overloaded function for gelss. Its overload differs for
// * const MatrixA&
// * const MatrixB&
// * const VectorS&
// * User-defined workspace
//
template< typename MatrixA, typename MatrixB, typename VectorS,
        typename Workspace >
inline std::ptrdiff_t gelss( const MatrixA& a, const MatrixB& b,
        const VectorS& s, const typename remove_imaginary< typename value<
        MatrixA >::type >::type rcond, fortran_int_t& rank,
        Workspace work ) {
    return gelss_impl< typename value< MatrixA >::type >::invoke( a, b,
            s, rcond, rank, work );
}

//
// Overloaded function for gelss. Its overload differs for
// * const MatrixA&
// * const MatrixB&
// * const VectorS&
// * Default workspace-type (optimal)
//
template< typename MatrixA, typename MatrixB, typename VectorS >
inline std::ptrdiff_t gelss( const MatrixA& a, const MatrixB& b,
        const VectorS& s, const typename remove_imaginary< typename value<
        MatrixA >::type >::type rcond, fortran_int_t& rank ) {
    return gelss_impl< typename value< MatrixA >::type >::invoke( a, b,
            s, rcond, rank, optimal_workspace() );
}

} // namespace lapack
} // namespace bindings
} // namespace numeric
} // namespace boost

#endif
