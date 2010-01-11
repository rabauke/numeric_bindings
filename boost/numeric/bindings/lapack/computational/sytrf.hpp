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

#ifndef BOOST_NUMERIC_BINDINGS_LAPACK_COMPUTATIONAL_SYTRF_HPP
#define BOOST_NUMERIC_BINDINGS_LAPACK_COMPUTATIONAL_SYTRF_HPP

#include <boost/assert.hpp>
#include <boost/numeric/bindings/begin.hpp>
#include <boost/numeric/bindings/data_side.hpp>
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
// The LAPACK-backend for sytrf is the netlib-compatible backend.
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
template< typename UpLo >
inline std::ptrdiff_t sytrf( UpLo, fortran_int_t n, float* a,
        fortran_int_t lda, fortran_int_t* ipiv, float* work,
        fortran_int_t lwork ) {
    fortran_int_t info(0);
    LAPACK_SSYTRF( &lapack_option< UpLo >::value, &n, a, &lda, ipiv, work,
            &lwork, &info );
    return info;
}

//
// Overloaded function for dispatching to
// * netlib-compatible LAPACK backend (the default), and
// * double value-type.
//
template< typename UpLo >
inline std::ptrdiff_t sytrf( UpLo, fortran_int_t n, double* a,
        fortran_int_t lda, fortran_int_t* ipiv, double* work,
        fortran_int_t lwork ) {
    fortran_int_t info(0);
    LAPACK_DSYTRF( &lapack_option< UpLo >::value, &n, a, &lda, ipiv, work,
            &lwork, &info );
    return info;
}

//
// Overloaded function for dispatching to
// * netlib-compatible LAPACK backend (the default), and
// * complex<float> value-type.
//
template< typename UpLo >
inline std::ptrdiff_t sytrf( UpLo, fortran_int_t n, std::complex<float>* a,
        fortran_int_t lda, fortran_int_t* ipiv, std::complex<float>* work,
        fortran_int_t lwork ) {
    fortran_int_t info(0);
    LAPACK_CSYTRF( &lapack_option< UpLo >::value, &n, a, &lda, ipiv, work,
            &lwork, &info );
    return info;
}

//
// Overloaded function for dispatching to
// * netlib-compatible LAPACK backend (the default), and
// * complex<double> value-type.
//
template< typename UpLo >
inline std::ptrdiff_t sytrf( UpLo, fortran_int_t n, std::complex<double>* a,
        fortran_int_t lda, fortran_int_t* ipiv, std::complex<double>* work,
        fortran_int_t lwork ) {
    fortran_int_t info(0);
    LAPACK_ZSYTRF( &lapack_option< UpLo >::value, &n, a, &lda, ipiv, work,
            &lwork, &info );
    return info;
}

} // namespace detail

//
// Value-type based template class. Use this class if you need a type
// for dispatching to sytrf.
//
template< typename Value, typename Enable = void >
struct sytrf_impl {};

//
// This implementation is enabled if Value is a real type.
//
template< typename Value >
struct sytrf_impl< Value, typename boost::enable_if< is_real< Value > >::type > {

    typedef Value value_type;
    typedef typename remove_imaginary< Value >::type real_type;
    typedef tag::column_major order;

    //
    // Static member function for user-defined workspaces, that
    // * Deduces the required arguments for dispatching to LAPACK, and
    // * Asserts that most arguments make sense.
    //
    template< typename MatrixA, typename VectorIPIV, typename WORK >
    static std::ptrdiff_t invoke( MatrixA& a, VectorIPIV& ipiv,
            detail::workspace1< WORK > work ) {
        typedef typename result_of::data_side< MatrixA >::type uplo;
        BOOST_STATIC_ASSERT( (is_mutable< MatrixA >::value) );
        BOOST_STATIC_ASSERT( (is_mutable< VectorIPIV >::value) );
        BOOST_ASSERT( size(work.select(real_type())) >= min_size_work());
        BOOST_ASSERT( size_column(a) >= 0 );
        BOOST_ASSERT( size_minor(a) == 1 || stride_minor(a) == 1 );
        BOOST_ASSERT( stride_major(a) >= std::max< std::ptrdiff_t >(1,
                size_column(a)) );
        return detail::sytrf( uplo(), size_column(a), begin_value(a),
                stride_major(a), begin_value(ipiv),
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
    template< typename MatrixA, typename VectorIPIV >
    static std::ptrdiff_t invoke( MatrixA& a, VectorIPIV& ipiv,
            minimal_workspace work ) {
        typedef typename result_of::data_side< MatrixA >::type uplo;
        bindings::detail::array< real_type > tmp_work( min_size_work() );
        return invoke( a, ipiv, workspace( tmp_work ) );
    }

    //
    // Static member function that
    // * Figures out the optimal workspace requirements, and passes
    //   the results to the user-defined workspace overload of the 
    //   invoke static member
    // * Enables the blocked algorithm (BLAS level 3)
    //
    template< typename MatrixA, typename VectorIPIV >
    static std::ptrdiff_t invoke( MatrixA& a, VectorIPIV& ipiv,
            optimal_workspace work ) {
        typedef typename result_of::data_side< MatrixA >::type uplo;
        real_type opt_size_work;
        detail::sytrf( uplo(), size_column(a), begin_value(a),
                stride_major(a), begin_value(ipiv), &opt_size_work, -1 );
        bindings::detail::array< real_type > tmp_work(
                traits::detail::to_int( opt_size_work ) );
        invoke( a, ipiv, workspace( tmp_work ) );
    }

    //
    // Static member function that returns the minimum size of
    // workspace-array work.
    //
    static std::ptrdiff_t min_size_work() {
        return 1;
    }
};

//
// This implementation is enabled if Value is a complex type.
//
template< typename Value >
struct sytrf_impl< Value, typename boost::enable_if< is_complex< Value > >::type > {

    typedef Value value_type;
    typedef typename remove_imaginary< Value >::type real_type;
    typedef tag::column_major order;

    //
    // Static member function for user-defined workspaces, that
    // * Deduces the required arguments for dispatching to LAPACK, and
    // * Asserts that most arguments make sense.
    //
    template< typename MatrixA, typename VectorIPIV, typename WORK >
    static std::ptrdiff_t invoke( MatrixA& a, VectorIPIV& ipiv,
            detail::workspace1< WORK > work ) {
        typedef typename result_of::data_side< MatrixA >::type uplo;
        BOOST_STATIC_ASSERT( (is_mutable< MatrixA >::value) );
        BOOST_STATIC_ASSERT( (is_mutable< VectorIPIV >::value) );
        BOOST_ASSERT( size(work.select(value_type())) >= min_size_work());
        BOOST_ASSERT( size_column(a) >= 0 );
        BOOST_ASSERT( size_minor(a) == 1 || stride_minor(a) == 1 );
        BOOST_ASSERT( stride_major(a) >= std::max< std::ptrdiff_t >(1,
                size_column(a)) );
        return detail::sytrf( uplo(), size_column(a), begin_value(a),
                stride_major(a), begin_value(ipiv),
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
    template< typename MatrixA, typename VectorIPIV >
    static std::ptrdiff_t invoke( MatrixA& a, VectorIPIV& ipiv,
            minimal_workspace work ) {
        typedef typename result_of::data_side< MatrixA >::type uplo;
        bindings::detail::array< value_type > tmp_work( min_size_work() );
        return invoke( a, ipiv, workspace( tmp_work ) );
    }

    //
    // Static member function that
    // * Figures out the optimal workspace requirements, and passes
    //   the results to the user-defined workspace overload of the 
    //   invoke static member
    // * Enables the blocked algorithm (BLAS level 3)
    //
    template< typename MatrixA, typename VectorIPIV >
    static std::ptrdiff_t invoke( MatrixA& a, VectorIPIV& ipiv,
            optimal_workspace work ) {
        typedef typename result_of::data_side< MatrixA >::type uplo;
        value_type opt_size_work;
        detail::sytrf( uplo(), size_column(a), begin_value(a),
                stride_major(a), begin_value(ipiv), &opt_size_work, -1 );
        bindings::detail::array< value_type > tmp_work(
                traits::detail::to_int( opt_size_work ) );
        invoke( a, ipiv, workspace( tmp_work ) );
    }

    //
    // Static member function that returns the minimum size of
    // workspace-array work.
    //
    static std::ptrdiff_t min_size_work() {
        return 1;
    }
};


//
// Functions for direct use. These functions are overloaded for temporaries,
// so that wrapped types can still be passed and used for write-access. In
// addition, if applicable, they are overloaded for user-defined workspaces.
// Calls to these functions are passed to the sytrf_impl classes. In the 
// documentation, most overloads are collapsed to avoid a large number of
// prototypes which are very similar.
//

//
// Overloaded function for sytrf. Its overload differs for
// * MatrixA&
// * VectorIPIV&
// * User-defined workspace
//
template< typename MatrixA, typename VectorIPIV, typename Workspace >
inline std::ptrdiff_t sytrf( MatrixA& a, VectorIPIV& ipiv,
        Workspace work ) {
    return sytrf_impl< typename value< MatrixA >::type >::invoke( a,
            ipiv, work );
}

//
// Overloaded function for sytrf. Its overload differs for
// * MatrixA&
// * VectorIPIV&
// * Default workspace-type (optimal)
//
template< typename MatrixA, typename VectorIPIV >
inline std::ptrdiff_t sytrf( MatrixA& a, VectorIPIV& ipiv ) {
    return sytrf_impl< typename value< MatrixA >::type >::invoke( a,
            ipiv, optimal_workspace() );
}

//
// Overloaded function for sytrf. Its overload differs for
// * const MatrixA&
// * VectorIPIV&
// * User-defined workspace
//
template< typename MatrixA, typename VectorIPIV, typename Workspace >
inline std::ptrdiff_t sytrf( const MatrixA& a, VectorIPIV& ipiv,
        Workspace work ) {
    return sytrf_impl< typename value< MatrixA >::type >::invoke( a,
            ipiv, work );
}

//
// Overloaded function for sytrf. Its overload differs for
// * const MatrixA&
// * VectorIPIV&
// * Default workspace-type (optimal)
//
template< typename MatrixA, typename VectorIPIV >
inline std::ptrdiff_t sytrf( const MatrixA& a, VectorIPIV& ipiv ) {
    return sytrf_impl< typename value< MatrixA >::type >::invoke( a,
            ipiv, optimal_workspace() );
}

//
// Overloaded function for sytrf. Its overload differs for
// * MatrixA&
// * const VectorIPIV&
// * User-defined workspace
//
template< typename MatrixA, typename VectorIPIV, typename Workspace >
inline std::ptrdiff_t sytrf( MatrixA& a, const VectorIPIV& ipiv,
        Workspace work ) {
    return sytrf_impl< typename value< MatrixA >::type >::invoke( a,
            ipiv, work );
}

//
// Overloaded function for sytrf. Its overload differs for
// * MatrixA&
// * const VectorIPIV&
// * Default workspace-type (optimal)
//
template< typename MatrixA, typename VectorIPIV >
inline std::ptrdiff_t sytrf( MatrixA& a, const VectorIPIV& ipiv ) {
    return sytrf_impl< typename value< MatrixA >::type >::invoke( a,
            ipiv, optimal_workspace() );
}

//
// Overloaded function for sytrf. Its overload differs for
// * const MatrixA&
// * const VectorIPIV&
// * User-defined workspace
//
template< typename MatrixA, typename VectorIPIV, typename Workspace >
inline std::ptrdiff_t sytrf( const MatrixA& a, const VectorIPIV& ipiv,
        Workspace work ) {
    return sytrf_impl< typename value< MatrixA >::type >::invoke( a,
            ipiv, work );
}

//
// Overloaded function for sytrf. Its overload differs for
// * const MatrixA&
// * const VectorIPIV&
// * Default workspace-type (optimal)
//
template< typename MatrixA, typename VectorIPIV >
inline std::ptrdiff_t sytrf( const MatrixA& a, const VectorIPIV& ipiv ) {
    return sytrf_impl< typename value< MatrixA >::type >::invoke( a,
            ipiv, optimal_workspace() );
}

} // namespace lapack
} // namespace bindings
} // namespace numeric
} // namespace boost

#endif
