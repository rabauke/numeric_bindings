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

#ifndef BOOST_NUMERIC_BINDINGS_LAPACK_DRIVER_HPEVD_HPP
#define BOOST_NUMERIC_BINDINGS_LAPACK_DRIVER_HPEVD_HPP

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
#include <boost/numeric/bindings/uplo_tag.hpp>
#include <boost/numeric/bindings/value_type.hpp>
#include <boost/static_assert.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/type_traits/remove_const.hpp>

//
// The LAPACK-backend for hpevd is the netlib-compatible backend.
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
template< typename UpLo >
inline std::ptrdiff_t hpevd( const char jobz, const UpLo uplo,
        const fortran_int_t n, std::complex<float>* ap, float* w,
        std::complex<float>* z, const fortran_int_t ldz,
        std::complex<float>* work, const fortran_int_t lwork, float* rwork,
        const fortran_int_t lrwork, fortran_int_t* iwork,
        const fortran_int_t liwork ) {
    fortran_int_t info(0);
    LAPACK_CHPEVD( &jobz, &lapack_option< UpLo >::value, &n, ap, w, z, &ldz,
            work, &lwork, rwork, &lrwork, iwork, &liwork, &info );
    return info;
}

//
// Overloaded function for dispatching to
// * netlib-compatible LAPACK backend (the default), and
// * complex<double> value-type.
//
template< typename UpLo >
inline std::ptrdiff_t hpevd( const char jobz, const UpLo uplo,
        const fortran_int_t n, std::complex<double>* ap, double* w,
        std::complex<double>* z, const fortran_int_t ldz,
        std::complex<double>* work, const fortran_int_t lwork, double* rwork,
        const fortran_int_t lrwork, fortran_int_t* iwork,
        const fortran_int_t liwork ) {
    fortran_int_t info(0);
    LAPACK_ZHPEVD( &jobz, &lapack_option< UpLo >::value, &n, ap, w, z, &ldz,
            work, &lwork, rwork, &lrwork, iwork, &liwork, &info );
    return info;
}

} // namespace detail

//
// Value-type based template class. Use this class if you need a type
// for dispatching to hpevd.
//
template< typename Value >
struct hpevd_impl {

    typedef Value value_type;
    typedef typename remove_imaginary< Value >::type real_type;

    //
    // Static member function for user-defined workspaces, that
    // * Deduces the required arguments for dispatching to LAPACK, and
    // * Asserts that most arguments make sense.
    //
    template< typename MatrixAP, typename VectorW, typename MatrixZ,
            typename WORK, typename RWORK, typename IWORK >
    static std::ptrdiff_t invoke( const char jobz, MatrixAP& ap, VectorW& w,
            MatrixZ& z, detail::workspace3< WORK, RWORK, IWORK > work ) {
        namespace bindings = ::boost::numeric::bindings;
        typedef typename result_of::uplo_tag< MatrixAP >::type uplo;
        BOOST_STATIC_ASSERT( (bindings::is_column_major< MatrixZ >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename remove_const<
                typename bindings::value_type< MatrixAP >::type >::type,
                typename remove_const< typename bindings::value_type<
                MatrixZ >::type >::type >::value) );
        BOOST_STATIC_ASSERT( (bindings::is_mutable< MatrixAP >::value) );
        BOOST_STATIC_ASSERT( (bindings::is_mutable< VectorW >::value) );
        BOOST_STATIC_ASSERT( (bindings::is_mutable< MatrixZ >::value) );
        BOOST_ASSERT( bindings::size(work.select(fortran_int_t())) >=
                min_size_iwork( jobz, bindings::size_column(ap) ));
        BOOST_ASSERT( bindings::size(work.select(real_type())) >=
                min_size_rwork( jobz, bindings::size_column(ap) ));
        BOOST_ASSERT( bindings::size(work.select(value_type())) >=
                min_size_work( jobz, bindings::size_column(ap) ));
        BOOST_ASSERT( bindings::size_column(ap) >= 0 );
        BOOST_ASSERT( bindings::size_minor(z) == 1 ||
                bindings::stride_minor(z) == 1 );
        BOOST_ASSERT( jobz == 'N' || jobz == 'V' );
        return detail::hpevd( jobz, uplo(), bindings::size_column(ap),
                bindings::begin_value(ap), bindings::begin_value(w),
                bindings::begin_value(z), bindings::stride_major(z),
                bindings::begin_value(work.select(value_type())),
                bindings::size(work.select(value_type())),
                bindings::begin_value(work.select(real_type())),
                bindings::size(work.select(real_type())),
                bindings::begin_value(work.select(fortran_int_t())),
                bindings::size(work.select(fortran_int_t())) );
    }

    //
    // Static member function that
    // * Figures out the minimal workspace requirements, and passes
    //   the results to the user-defined workspace overload of the 
    //   invoke static member function
    // * Enables the unblocked algorithm (BLAS level 2)
    //
    template< typename MatrixAP, typename VectorW, typename MatrixZ >
    static std::ptrdiff_t invoke( const char jobz, MatrixAP& ap, VectorW& w,
            MatrixZ& z, minimal_workspace ) {
        namespace bindings = ::boost::numeric::bindings;
        typedef typename result_of::uplo_tag< MatrixAP >::type uplo;
        bindings::detail::array< value_type > tmp_work( min_size_work( jobz,
                bindings::size_column(ap) ) );
        bindings::detail::array< real_type > tmp_rwork( min_size_rwork( jobz,
                bindings::size_column(ap) ) );
        bindings::detail::array< fortran_int_t > tmp_iwork(
                min_size_iwork( jobz, bindings::size_column(ap) ) );
        return invoke( jobz, ap, w, z, workspace( tmp_work, tmp_rwork,
                tmp_iwork ) );
    }

    //
    // Static member function that
    // * Figures out the optimal workspace requirements, and passes
    //   the results to the user-defined workspace overload of the 
    //   invoke static member
    // * Enables the blocked algorithm (BLAS level 3)
    //
    template< typename MatrixAP, typename VectorW, typename MatrixZ >
    static std::ptrdiff_t invoke( const char jobz, MatrixAP& ap, VectorW& w,
            MatrixZ& z, optimal_workspace ) {
        namespace bindings = ::boost::numeric::bindings;
        typedef typename result_of::uplo_tag< MatrixAP >::type uplo;
        value_type opt_size_work;
        real_type opt_size_rwork;
        fortran_int_t opt_size_iwork;
        detail::hpevd( jobz, uplo(), bindings::size_column(ap),
                bindings::begin_value(ap), bindings::begin_value(w),
                bindings::begin_value(z), bindings::stride_major(z),
                &opt_size_work, -1, &opt_size_rwork, -1, &opt_size_iwork, -1 );
        bindings::detail::array< value_type > tmp_work(
                traits::detail::to_int( opt_size_work ) );
        bindings::detail::array< real_type > tmp_rwork(
                traits::detail::to_int( opt_size_rwork ) );
        bindings::detail::array< fortran_int_t > tmp_iwork(
                opt_size_iwork );
        return invoke( jobz, ap, w, z, workspace( tmp_work, tmp_rwork,
                tmp_iwork ) );
    }

    //
    // Static member function that returns the minimum size of
    // workspace-array work.
    //
    static std::ptrdiff_t min_size_work( const char jobz,
            const std::ptrdiff_t n ) {
        if ( n < 2 )
            return 1;
        else {
            if ( jobz == 'N' )
                return n;
            else
                return 2*n;
        }
    }

    //
    // Static member function that returns the minimum size of
    // workspace-array rwork.
    //
    static std::ptrdiff_t min_size_rwork( const char jobz,
            const std::ptrdiff_t n ) {
        if ( n < 2 )
            return 1;
        else {
            if ( jobz == 'N' )
                return n;
            else
                return 1 + 5*n + 2*n*n;
        }
    }

    //
    // Static member function that returns the minimum size of
    // workspace-array iwork.
    //
    static std::ptrdiff_t min_size_iwork( const char jobz,
            const std::ptrdiff_t n ) {
        if ( jobz == 'N' || n < 2 )
            return 1;
        else
            return 3 + 5*n;
    }
};


//
// Functions for direct use. These functions are overloaded for temporaries,
// so that wrapped types can still be passed and used for write-access. In
// addition, if applicable, they are overloaded for user-defined workspaces.
// Calls to these functions are passed to the hpevd_impl classes. In the 
// documentation, most overloads are collapsed to avoid a large number of
// prototypes which are very similar.
//

//
// Overloaded function for hpevd. Its overload differs for
// * MatrixAP&
// * MatrixZ&
// * User-defined workspace
//
template< typename MatrixAP, typename VectorW, typename MatrixZ,
        typename Workspace >
inline typename boost::enable_if< detail::is_workspace< Workspace >,
        std::ptrdiff_t >::type
hpevd( const char jobz, MatrixAP& ap, VectorW& w, MatrixZ& z,
        Workspace work ) {
    return hpevd_impl< typename bindings::value_type<
            MatrixAP >::type >::invoke( jobz, ap, w, z, work );
}

//
// Overloaded function for hpevd. Its overload differs for
// * MatrixAP&
// * MatrixZ&
// * Default workspace-type (optimal)
//
template< typename MatrixAP, typename VectorW, typename MatrixZ >
inline typename boost::disable_if< detail::is_workspace< MatrixZ >,
        std::ptrdiff_t >::type
hpevd( const char jobz, MatrixAP& ap, VectorW& w, MatrixZ& z ) {
    return hpevd_impl< typename bindings::value_type<
            MatrixAP >::type >::invoke( jobz, ap, w, z, optimal_workspace() );
}

//
// Overloaded function for hpevd. Its overload differs for
// * const MatrixAP&
// * MatrixZ&
// * User-defined workspace
//
template< typename MatrixAP, typename VectorW, typename MatrixZ,
        typename Workspace >
inline typename boost::enable_if< detail::is_workspace< Workspace >,
        std::ptrdiff_t >::type
hpevd( const char jobz, const MatrixAP& ap, VectorW& w, MatrixZ& z,
        Workspace work ) {
    return hpevd_impl< typename bindings::value_type<
            MatrixAP >::type >::invoke( jobz, ap, w, z, work );
}

//
// Overloaded function for hpevd. Its overload differs for
// * const MatrixAP&
// * MatrixZ&
// * Default workspace-type (optimal)
//
template< typename MatrixAP, typename VectorW, typename MatrixZ >
inline typename boost::disable_if< detail::is_workspace< MatrixZ >,
        std::ptrdiff_t >::type
hpevd( const char jobz, const MatrixAP& ap, VectorW& w, MatrixZ& z ) {
    return hpevd_impl< typename bindings::value_type<
            MatrixAP >::type >::invoke( jobz, ap, w, z, optimal_workspace() );
}

//
// Overloaded function for hpevd. Its overload differs for
// * MatrixAP&
// * const MatrixZ&
// * User-defined workspace
//
template< typename MatrixAP, typename VectorW, typename MatrixZ,
        typename Workspace >
inline typename boost::enable_if< detail::is_workspace< Workspace >,
        std::ptrdiff_t >::type
hpevd( const char jobz, MatrixAP& ap, VectorW& w, const MatrixZ& z,
        Workspace work ) {
    return hpevd_impl< typename bindings::value_type<
            MatrixAP >::type >::invoke( jobz, ap, w, z, work );
}

//
// Overloaded function for hpevd. Its overload differs for
// * MatrixAP&
// * const MatrixZ&
// * Default workspace-type (optimal)
//
template< typename MatrixAP, typename VectorW, typename MatrixZ >
inline typename boost::disable_if< detail::is_workspace< MatrixZ >,
        std::ptrdiff_t >::type
hpevd( const char jobz, MatrixAP& ap, VectorW& w, const MatrixZ& z ) {
    return hpevd_impl< typename bindings::value_type<
            MatrixAP >::type >::invoke( jobz, ap, w, z, optimal_workspace() );
}

//
// Overloaded function for hpevd. Its overload differs for
// * const MatrixAP&
// * const MatrixZ&
// * User-defined workspace
//
template< typename MatrixAP, typename VectorW, typename MatrixZ,
        typename Workspace >
inline typename boost::enable_if< detail::is_workspace< Workspace >,
        std::ptrdiff_t >::type
hpevd( const char jobz, const MatrixAP& ap, VectorW& w, const MatrixZ& z,
        Workspace work ) {
    return hpevd_impl< typename bindings::value_type<
            MatrixAP >::type >::invoke( jobz, ap, w, z, work );
}

//
// Overloaded function for hpevd. Its overload differs for
// * const MatrixAP&
// * const MatrixZ&
// * Default workspace-type (optimal)
//
template< typename MatrixAP, typename VectorW, typename MatrixZ >
inline typename boost::disable_if< detail::is_workspace< MatrixZ >,
        std::ptrdiff_t >::type
hpevd( const char jobz, const MatrixAP& ap, VectorW& w,
        const MatrixZ& z ) {
    return hpevd_impl< typename bindings::value_type<
            MatrixAP >::type >::invoke( jobz, ap, w, z, optimal_workspace() );
}

} // namespace lapack
} // namespace bindings
} // namespace numeric
} // namespace boost

#endif
