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

#ifndef BOOST_NUMERIC_BINDINGS_LAPACK_COMPUTATIONAL_GEHRD_HPP
#define BOOST_NUMERIC_BINDINGS_LAPACK_COMPUTATIONAL_GEHRD_HPP

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
// The LAPACK-backend for gehrd is the netlib-compatible backend.
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
inline std::ptrdiff_t gehrd( const fortran_int_t n, const fortran_int_t ilo,
        const fortran_int_t ihi, float* a, const fortran_int_t lda,
        float* tau, float* work, const fortran_int_t lwork ) {
    fortran_int_t info(0);
    LAPACK_SGEHRD( &n, &ilo, &ihi, a, &lda, tau, work, &lwork, &info );
    return info;
}

//
// Overloaded function for dispatching to
// * netlib-compatible LAPACK backend (the default), and
// * double value-type.
//
inline std::ptrdiff_t gehrd( const fortran_int_t n, const fortran_int_t ilo,
        const fortran_int_t ihi, double* a, const fortran_int_t lda,
        double* tau, double* work, const fortran_int_t lwork ) {
    fortran_int_t info(0);
    LAPACK_DGEHRD( &n, &ilo, &ihi, a, &lda, tau, work, &lwork, &info );
    return info;
}

//
// Overloaded function for dispatching to
// * netlib-compatible LAPACK backend (the default), and
// * complex<float> value-type.
//
inline std::ptrdiff_t gehrd( const fortran_int_t n, const fortran_int_t ilo,
        const fortran_int_t ihi, std::complex<float>* a,
        const fortran_int_t lda, std::complex<float>* tau,
        std::complex<float>* work, const fortran_int_t lwork ) {
    fortran_int_t info(0);
    LAPACK_CGEHRD( &n, &ilo, &ihi, a, &lda, tau, work, &lwork, &info );
    return info;
}

//
// Overloaded function for dispatching to
// * netlib-compatible LAPACK backend (the default), and
// * complex<double> value-type.
//
inline std::ptrdiff_t gehrd( const fortran_int_t n, const fortran_int_t ilo,
        const fortran_int_t ihi, std::complex<double>* a,
        const fortran_int_t lda, std::complex<double>* tau,
        std::complex<double>* work, const fortran_int_t lwork ) {
    fortran_int_t info(0);
    LAPACK_ZGEHRD( &n, &ilo, &ihi, a, &lda, tau, work, &lwork, &info );
    return info;
}

} // namespace detail

//
// Value-type based template class. Use this class if you need a type
// for dispatching to gehrd.
//
template< typename Value, typename Enable = void >
struct gehrd_impl {};

//
// This implementation is enabled if Value is a real type.
//
template< typename Value >
struct gehrd_impl< Value, typename boost::enable_if< is_real< Value > >::type > {

    typedef Value value_type;
    typedef typename remove_imaginary< Value >::type real_type;
    typedef tag::column_major order;

    //
    // Static member function for user-defined workspaces, that
    // * Deduces the required arguments for dispatching to LAPACK, and
    // * Asserts that most arguments make sense.
    //
    template< typename MatrixA, typename VectorTAU, typename WORK >
    static std::ptrdiff_t invoke( const fortran_int_t ilo,
            const fortran_int_t ihi, MatrixA& a, VectorTAU& tau,
            detail::workspace1< WORK > work ) {
        BOOST_STATIC_ASSERT( (boost::is_same< typename remove_const<
                typename value< MatrixA >::type >::type,
                typename remove_const< typename value<
                VectorTAU >::type >::type >::value) );
        BOOST_STATIC_ASSERT( (is_mutable< MatrixA >::value) );
        BOOST_STATIC_ASSERT( (is_mutable< VectorTAU >::value) );
        BOOST_ASSERT( size(tau) >= size_column(a)-1 );
        BOOST_ASSERT( size(work.select(real_type())) >= min_size_work(
                $CALL_MIN_SIZE ));
        BOOST_ASSERT( size_column(a) >= 0 );
        BOOST_ASSERT( size_minor(a) == 1 || stride_minor(a) == 1 );
        BOOST_ASSERT( stride_major(a) >= std::max< std::ptrdiff_t >(1,
                size_column(a)) );
        return detail::gehrd( size_column(a), ilo, ihi, begin_value(a),
                stride_major(a), begin_value(tau),
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
    template< typename MatrixA, typename VectorTAU >
    static std::ptrdiff_t invoke( const fortran_int_t ilo,
            const fortran_int_t ihi, MatrixA& a, VectorTAU& tau,
            minimal_workspace work ) {
        bindings::detail::array< real_type > tmp_work( min_size_work(
                $CALL_MIN_SIZE ) );
        return invoke( ilo, ihi, a, tau, workspace( tmp_work ) );
    }

    //
    // Static member function that
    // * Figures out the optimal workspace requirements, and passes
    //   the results to the user-defined workspace overload of the 
    //   invoke static member
    // * Enables the blocked algorithm (BLAS level 3)
    //
    template< typename MatrixA, typename VectorTAU >
    static std::ptrdiff_t invoke( const fortran_int_t ilo,
            const fortran_int_t ihi, MatrixA& a, VectorTAU& tau,
            optimal_workspace work ) {
        real_type opt_size_work;
        detail::gehrd( size_column(a), ilo, ihi, begin_value(a),
                stride_major(a), begin_value(tau), &opt_size_work, -1 );
        bindings::detail::array< real_type > tmp_work(
                traits::detail::to_int( opt_size_work ) );
        return invoke( ilo, ihi, a, tau, workspace( tmp_work ) );
    }

    //
    // Static member function that returns the minimum size of
    // workspace-array work.
    //
    static std::ptrdiff_t min_size_work( $ARGUMENTS ) {
        $MIN_SIZE
    }
};

//
// This implementation is enabled if Value is a complex type.
//
template< typename Value >
struct gehrd_impl< Value, typename boost::enable_if< is_complex< Value > >::type > {

    typedef Value value_type;
    typedef typename remove_imaginary< Value >::type real_type;
    typedef tag::column_major order;

    //
    // Static member function for user-defined workspaces, that
    // * Deduces the required arguments for dispatching to LAPACK, and
    // * Asserts that most arguments make sense.
    //
    template< typename MatrixA, typename VectorTAU, typename WORK >
    static std::ptrdiff_t invoke( const fortran_int_t ilo,
            const fortran_int_t ihi, MatrixA& a, VectorTAU& tau,
            detail::workspace1< WORK > work ) {
        BOOST_STATIC_ASSERT( (boost::is_same< typename remove_const<
                typename value< MatrixA >::type >::type,
                typename remove_const< typename value<
                VectorTAU >::type >::type >::value) );
        BOOST_STATIC_ASSERT( (is_mutable< MatrixA >::value) );
        BOOST_STATIC_ASSERT( (is_mutable< VectorTAU >::value) );
        BOOST_ASSERT( size(tau) >= size_column(a)-1 );
        BOOST_ASSERT( size(work.select(value_type())) >= min_size_work(
                $CALL_MIN_SIZE ));
        BOOST_ASSERT( size_column(a) >= 0 );
        BOOST_ASSERT( size_minor(a) == 1 || stride_minor(a) == 1 );
        BOOST_ASSERT( stride_major(a) >= std::max< std::ptrdiff_t >(1,
                size_column(a)) );
        return detail::gehrd( size_column(a), ilo, ihi, begin_value(a),
                stride_major(a), begin_value(tau),
                begin_value(work.select(value_type())),
                size(work.select(value_type())) );
    }

    //
    // Static member function that
    // * Figures out the minimal workspace requirements, and passes
    //   the results to the user-defined workspace overload of the 
    //   invoke static member function
    // * Enables the unblocked algorithm (BLAS level 2)
    //
    template< typename MatrixA, typename VectorTAU >
    static std::ptrdiff_t invoke( const fortran_int_t ilo,
            const fortran_int_t ihi, MatrixA& a, VectorTAU& tau,
            minimal_workspace work ) {
        bindings::detail::array< value_type > tmp_work( min_size_work(
                $CALL_MIN_SIZE ) );
        return invoke( ilo, ihi, a, tau, workspace( tmp_work ) );
    }

    //
    // Static member function that
    // * Figures out the optimal workspace requirements, and passes
    //   the results to the user-defined workspace overload of the 
    //   invoke static member
    // * Enables the blocked algorithm (BLAS level 3)
    //
    template< typename MatrixA, typename VectorTAU >
    static std::ptrdiff_t invoke( const fortran_int_t ilo,
            const fortran_int_t ihi, MatrixA& a, VectorTAU& tau,
            optimal_workspace work ) {
        value_type opt_size_work;
        detail::gehrd( size_column(a), ilo, ihi, begin_value(a),
                stride_major(a), begin_value(tau), &opt_size_work, -1 );
        bindings::detail::array< value_type > tmp_work(
                traits::detail::to_int( opt_size_work ) );
        return invoke( ilo, ihi, a, tau, workspace( tmp_work ) );
    }

    //
    // Static member function that returns the minimum size of
    // workspace-array work.
    //
    static std::ptrdiff_t min_size_work( $ARGUMENTS ) {
        $MIN_SIZE
    }
};


//
// Functions for direct use. These functions are overloaded for temporaries,
// so that wrapped types can still be passed and used for write-access. In
// addition, if applicable, they are overloaded for user-defined workspaces.
// Calls to these functions are passed to the gehrd_impl classes. In the 
// documentation, most overloads are collapsed to avoid a large number of
// prototypes which are very similar.
//

//
// Overloaded function for gehrd. Its overload differs for
// * MatrixA&
// * VectorTAU&
// * User-defined workspace
//
template< typename MatrixA, typename VectorTAU, typename Workspace >
inline std::ptrdiff_t gehrd( const fortran_int_t ilo,
        const fortran_int_t ihi, MatrixA& a, VectorTAU& tau,
        Workspace work ) {
    return gehrd_impl< typename value< MatrixA >::type >::invoke( ilo,
            ihi, a, tau, work );
}

//
// Overloaded function for gehrd. Its overload differs for
// * MatrixA&
// * VectorTAU&
// * Default workspace-type (optimal)
//
template< typename MatrixA, typename VectorTAU >
inline std::ptrdiff_t gehrd( const fortran_int_t ilo,
        const fortran_int_t ihi, MatrixA& a, VectorTAU& tau ) {
    return gehrd_impl< typename value< MatrixA >::type >::invoke( ilo,
            ihi, a, tau, optimal_workspace() );
}

//
// Overloaded function for gehrd. Its overload differs for
// * const MatrixA&
// * VectorTAU&
// * User-defined workspace
//
template< typename MatrixA, typename VectorTAU, typename Workspace >
inline std::ptrdiff_t gehrd( const fortran_int_t ilo,
        const fortran_int_t ihi, const MatrixA& a, VectorTAU& tau,
        Workspace work ) {
    return gehrd_impl< typename value< MatrixA >::type >::invoke( ilo,
            ihi, a, tau, work );
}

//
// Overloaded function for gehrd. Its overload differs for
// * const MatrixA&
// * VectorTAU&
// * Default workspace-type (optimal)
//
template< typename MatrixA, typename VectorTAU >
inline std::ptrdiff_t gehrd( const fortran_int_t ilo,
        const fortran_int_t ihi, const MatrixA& a, VectorTAU& tau ) {
    return gehrd_impl< typename value< MatrixA >::type >::invoke( ilo,
            ihi, a, tau, optimal_workspace() );
}

//
// Overloaded function for gehrd. Its overload differs for
// * MatrixA&
// * const VectorTAU&
// * User-defined workspace
//
template< typename MatrixA, typename VectorTAU, typename Workspace >
inline std::ptrdiff_t gehrd( const fortran_int_t ilo,
        const fortran_int_t ihi, MatrixA& a, const VectorTAU& tau,
        Workspace work ) {
    return gehrd_impl< typename value< MatrixA >::type >::invoke( ilo,
            ihi, a, tau, work );
}

//
// Overloaded function for gehrd. Its overload differs for
// * MatrixA&
// * const VectorTAU&
// * Default workspace-type (optimal)
//
template< typename MatrixA, typename VectorTAU >
inline std::ptrdiff_t gehrd( const fortran_int_t ilo,
        const fortran_int_t ihi, MatrixA& a, const VectorTAU& tau ) {
    return gehrd_impl< typename value< MatrixA >::type >::invoke( ilo,
            ihi, a, tau, optimal_workspace() );
}

//
// Overloaded function for gehrd. Its overload differs for
// * const MatrixA&
// * const VectorTAU&
// * User-defined workspace
//
template< typename MatrixA, typename VectorTAU, typename Workspace >
inline std::ptrdiff_t gehrd( const fortran_int_t ilo,
        const fortran_int_t ihi, const MatrixA& a, const VectorTAU& tau,
        Workspace work ) {
    return gehrd_impl< typename value< MatrixA >::type >::invoke( ilo,
            ihi, a, tau, work );
}

//
// Overloaded function for gehrd. Its overload differs for
// * const MatrixA&
// * const VectorTAU&
// * Default workspace-type (optimal)
//
template< typename MatrixA, typename VectorTAU >
inline std::ptrdiff_t gehrd( const fortran_int_t ilo,
        const fortran_int_t ihi, const MatrixA& a, const VectorTAU& tau ) {
    return gehrd_impl< typename value< MatrixA >::type >::invoke( ilo,
            ihi, a, tau, optimal_workspace() );
}

} // namespace lapack
} // namespace bindings
} // namespace numeric
} // namespace boost

#endif
