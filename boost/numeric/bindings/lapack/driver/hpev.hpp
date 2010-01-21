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

#ifndef BOOST_NUMERIC_BINDINGS_LAPACK_DRIVER_HPEV_HPP
#define BOOST_NUMERIC_BINDINGS_LAPACK_DRIVER_HPEV_HPP

#include <boost/assert.hpp>
#include <boost/numeric/bindings/begin.hpp>
#include <boost/numeric/bindings/detail/array.hpp>
#include <boost/numeric/bindings/is_column_major.hpp>
#include <boost/numeric/bindings/is_mutable.hpp>
#include <boost/numeric/bindings/lapack/workspace.hpp>
#include <boost/numeric/bindings/remove_imaginary.hpp>
#include <boost/numeric/bindings/size.hpp>
#include <boost/numeric/bindings/stride.hpp>
#include <boost/numeric/bindings/uplo_tag.hpp>
#include <boost/numeric/bindings/value_type.hpp>
#include <boost/static_assert.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/type_traits/remove_const.hpp>

//
// The LAPACK-backend for hpev is the netlib-compatible backend.
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
inline std::ptrdiff_t hpev( const char jobz, const UpLo uplo,
        const fortran_int_t n, std::complex<float>* ap, float* w,
        std::complex<float>* z, const fortran_int_t ldz,
        std::complex<float>* work, float* rwork ) {
    fortran_int_t info(0);
    LAPACK_CHPEV( &jobz, &lapack_option< UpLo >::value, &n, ap, w, z, &ldz,
            work, rwork, &info );
    return info;
}

//
// Overloaded function for dispatching to
// * netlib-compatible LAPACK backend (the default), and
// * complex<double> value-type.
//
template< typename UpLo >
inline std::ptrdiff_t hpev( const char jobz, const UpLo uplo,
        const fortran_int_t n, std::complex<double>* ap, double* w,
        std::complex<double>* z, const fortran_int_t ldz,
        std::complex<double>* work, double* rwork ) {
    fortran_int_t info(0);
    LAPACK_ZHPEV( &jobz, &lapack_option< UpLo >::value, &n, ap, w, z, &ldz,
            work, rwork, &info );
    return info;
}

} // namespace detail

//
// Value-type based template class. Use this class if you need a type
// for dispatching to hpev.
//
template< typename Value >
struct hpev_impl {

    typedef Value value_type;
    typedef typename remove_imaginary< Value >::type real_type;

    //
    // Static member function for user-defined workspaces, that
    // * Deduces the required arguments for dispatching to LAPACK, and
    // * Asserts that most arguments make sense.
    //
    template< typename MatrixAP, typename VectorW, typename MatrixZ,
            typename WORK, typename RWORK >
    static std::ptrdiff_t invoke( const char jobz, MatrixAP& ap, VectorW& w,
            MatrixZ& z, detail::workspace2< WORK, RWORK > work ) {
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
        BOOST_ASSERT( bindings::size(work.select(real_type())) >=
                min_size_rwork( bindings::size_column(ap) ));
        BOOST_ASSERT( bindings::size(work.select(value_type())) >=
                min_size_work( bindings::size_column(ap) ));
        BOOST_ASSERT( bindings::size_column(ap) >= 0 );
        BOOST_ASSERT( bindings::size_minor(z) == 1 ||
                bindings::stride_minor(z) == 1 );
        BOOST_ASSERT( jobz == 'N' || jobz == 'V' );
        return detail::hpev( jobz, uplo(), bindings::size_column(ap),
                bindings::begin_value(ap), bindings::begin_value(w),
                bindings::begin_value(z), bindings::stride_major(z),
                bindings::begin_value(work.select(value_type())),
                bindings::begin_value(work.select(real_type())) );
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
            MatrixZ& z, minimal_workspace work ) {
        namespace bindings = ::boost::numeric::bindings;
        typedef typename result_of::uplo_tag< MatrixAP >::type uplo;
        bindings::detail::array< value_type > tmp_work( min_size_work(
                bindings::size_column(ap) ) );
        bindings::detail::array< real_type > tmp_rwork( min_size_rwork(
                bindings::size_column(ap) ) );
        return invoke( jobz, ap, w, z, workspace( tmp_work, tmp_rwork ) );
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
            MatrixZ& z, optimal_workspace work ) {
        namespace bindings = ::boost::numeric::bindings;
        typedef typename result_of::uplo_tag< MatrixAP >::type uplo;
        return invoke( jobz, ap, w, z, minimal_workspace() );
    }

    //
    // Static member function that returns the minimum size of
    // workspace-array work.
    //
    static std::ptrdiff_t min_size_work( const std::ptrdiff_t n ) {
        return std::max< std::ptrdiff_t >(1,2*n-1);
    }

    //
    // Static member function that returns the minimum size of
    // workspace-array rwork.
    //
    static std::ptrdiff_t min_size_rwork( const std::ptrdiff_t n ) {
        return std::max< std::ptrdiff_t >(1,3*n-2);
    }
};


//
// Functions for direct use. These functions are overloaded for temporaries,
// so that wrapped types can still be passed and used for write-access. In
// addition, if applicable, they are overloaded for user-defined workspaces.
// Calls to these functions are passed to the hpev_impl classes. In the 
// documentation, most overloads are collapsed to avoid a large number of
// prototypes which are very similar.
//

//
// Overloaded function for hpev. Its overload differs for
// * MatrixAP&
// * VectorW&
// * MatrixZ&
// * User-defined workspace
//
template< typename MatrixAP, typename VectorW, typename MatrixZ,
        typename Workspace >
inline typename boost::enable_if< detail::is_workspace< Workspace >,
        std::ptrdiff_t >::type
hpev( const char jobz, MatrixAP& ap, VectorW& w, MatrixZ& z,
        Workspace work ) {
    return hpev_impl< typename bindings::value_type<
            MatrixAP >::type >::invoke( jobz, ap, w, z, work );
}

//
// Overloaded function for hpev. Its overload differs for
// * MatrixAP&
// * VectorW&
// * MatrixZ&
// * Default workspace-type (optimal)
//
template< typename MatrixAP, typename VectorW, typename MatrixZ >
inline typename boost::disable_if< detail::is_workspace< MatrixZ >,
        std::ptrdiff_t >::type
hpev( const char jobz, MatrixAP& ap, VectorW& w, MatrixZ& z ) {
    return hpev_impl< typename bindings::value_type<
            MatrixAP >::type >::invoke( jobz, ap, w, z, optimal_workspace() );
}

//
// Overloaded function for hpev. Its overload differs for
// * const MatrixAP&
// * VectorW&
// * MatrixZ&
// * User-defined workspace
//
template< typename MatrixAP, typename VectorW, typename MatrixZ,
        typename Workspace >
inline typename boost::enable_if< detail::is_workspace< Workspace >,
        std::ptrdiff_t >::type
hpev( const char jobz, const MatrixAP& ap, VectorW& w, MatrixZ& z,
        Workspace work ) {
    return hpev_impl< typename bindings::value_type<
            MatrixAP >::type >::invoke( jobz, ap, w, z, work );
}

//
// Overloaded function for hpev. Its overload differs for
// * const MatrixAP&
// * VectorW&
// * MatrixZ&
// * Default workspace-type (optimal)
//
template< typename MatrixAP, typename VectorW, typename MatrixZ >
inline typename boost::disable_if< detail::is_workspace< MatrixZ >,
        std::ptrdiff_t >::type
hpev( const char jobz, const MatrixAP& ap, VectorW& w, MatrixZ& z ) {
    return hpev_impl< typename bindings::value_type<
            MatrixAP >::type >::invoke( jobz, ap, w, z, optimal_workspace() );
}

//
// Overloaded function for hpev. Its overload differs for
// * MatrixAP&
// * const VectorW&
// * MatrixZ&
// * User-defined workspace
//
template< typename MatrixAP, typename VectorW, typename MatrixZ,
        typename Workspace >
inline typename boost::enable_if< detail::is_workspace< Workspace >,
        std::ptrdiff_t >::type
hpev( const char jobz, MatrixAP& ap, const VectorW& w, MatrixZ& z,
        Workspace work ) {
    return hpev_impl< typename bindings::value_type<
            MatrixAP >::type >::invoke( jobz, ap, w, z, work );
}

//
// Overloaded function for hpev. Its overload differs for
// * MatrixAP&
// * const VectorW&
// * MatrixZ&
// * Default workspace-type (optimal)
//
template< typename MatrixAP, typename VectorW, typename MatrixZ >
inline typename boost::disable_if< detail::is_workspace< MatrixZ >,
        std::ptrdiff_t >::type
hpev( const char jobz, MatrixAP& ap, const VectorW& w, MatrixZ& z ) {
    return hpev_impl< typename bindings::value_type<
            MatrixAP >::type >::invoke( jobz, ap, w, z, optimal_workspace() );
}

//
// Overloaded function for hpev. Its overload differs for
// * const MatrixAP&
// * const VectorW&
// * MatrixZ&
// * User-defined workspace
//
template< typename MatrixAP, typename VectorW, typename MatrixZ,
        typename Workspace >
inline typename boost::enable_if< detail::is_workspace< Workspace >,
        std::ptrdiff_t >::type
hpev( const char jobz, const MatrixAP& ap, const VectorW& w, MatrixZ& z,
        Workspace work ) {
    return hpev_impl< typename bindings::value_type<
            MatrixAP >::type >::invoke( jobz, ap, w, z, work );
}

//
// Overloaded function for hpev. Its overload differs for
// * const MatrixAP&
// * const VectorW&
// * MatrixZ&
// * Default workspace-type (optimal)
//
template< typename MatrixAP, typename VectorW, typename MatrixZ >
inline typename boost::disable_if< detail::is_workspace< MatrixZ >,
        std::ptrdiff_t >::type
hpev( const char jobz, const MatrixAP& ap, const VectorW& w,
        MatrixZ& z ) {
    return hpev_impl< typename bindings::value_type<
            MatrixAP >::type >::invoke( jobz, ap, w, z, optimal_workspace() );
}

//
// Overloaded function for hpev. Its overload differs for
// * MatrixAP&
// * VectorW&
// * const MatrixZ&
// * User-defined workspace
//
template< typename MatrixAP, typename VectorW, typename MatrixZ,
        typename Workspace >
inline typename boost::enable_if< detail::is_workspace< Workspace >,
        std::ptrdiff_t >::type
hpev( const char jobz, MatrixAP& ap, VectorW& w, const MatrixZ& z,
        Workspace work ) {
    return hpev_impl< typename bindings::value_type<
            MatrixAP >::type >::invoke( jobz, ap, w, z, work );
}

//
// Overloaded function for hpev. Its overload differs for
// * MatrixAP&
// * VectorW&
// * const MatrixZ&
// * Default workspace-type (optimal)
//
template< typename MatrixAP, typename VectorW, typename MatrixZ >
inline typename boost::disable_if< detail::is_workspace< MatrixZ >,
        std::ptrdiff_t >::type
hpev( const char jobz, MatrixAP& ap, VectorW& w, const MatrixZ& z ) {
    return hpev_impl< typename bindings::value_type<
            MatrixAP >::type >::invoke( jobz, ap, w, z, optimal_workspace() );
}

//
// Overloaded function for hpev. Its overload differs for
// * const MatrixAP&
// * VectorW&
// * const MatrixZ&
// * User-defined workspace
//
template< typename MatrixAP, typename VectorW, typename MatrixZ,
        typename Workspace >
inline typename boost::enable_if< detail::is_workspace< Workspace >,
        std::ptrdiff_t >::type
hpev( const char jobz, const MatrixAP& ap, VectorW& w, const MatrixZ& z,
        Workspace work ) {
    return hpev_impl< typename bindings::value_type<
            MatrixAP >::type >::invoke( jobz, ap, w, z, work );
}

//
// Overloaded function for hpev. Its overload differs for
// * const MatrixAP&
// * VectorW&
// * const MatrixZ&
// * Default workspace-type (optimal)
//
template< typename MatrixAP, typename VectorW, typename MatrixZ >
inline typename boost::disable_if< detail::is_workspace< MatrixZ >,
        std::ptrdiff_t >::type
hpev( const char jobz, const MatrixAP& ap, VectorW& w,
        const MatrixZ& z ) {
    return hpev_impl< typename bindings::value_type<
            MatrixAP >::type >::invoke( jobz, ap, w, z, optimal_workspace() );
}

//
// Overloaded function for hpev. Its overload differs for
// * MatrixAP&
// * const VectorW&
// * const MatrixZ&
// * User-defined workspace
//
template< typename MatrixAP, typename VectorW, typename MatrixZ,
        typename Workspace >
inline typename boost::enable_if< detail::is_workspace< Workspace >,
        std::ptrdiff_t >::type
hpev( const char jobz, MatrixAP& ap, const VectorW& w, const MatrixZ& z,
        Workspace work ) {
    return hpev_impl< typename bindings::value_type<
            MatrixAP >::type >::invoke( jobz, ap, w, z, work );
}

//
// Overloaded function for hpev. Its overload differs for
// * MatrixAP&
// * const VectorW&
// * const MatrixZ&
// * Default workspace-type (optimal)
//
template< typename MatrixAP, typename VectorW, typename MatrixZ >
inline typename boost::disable_if< detail::is_workspace< MatrixZ >,
        std::ptrdiff_t >::type
hpev( const char jobz, MatrixAP& ap, const VectorW& w,
        const MatrixZ& z ) {
    return hpev_impl< typename bindings::value_type<
            MatrixAP >::type >::invoke( jobz, ap, w, z, optimal_workspace() );
}

//
// Overloaded function for hpev. Its overload differs for
// * const MatrixAP&
// * const VectorW&
// * const MatrixZ&
// * User-defined workspace
//
template< typename MatrixAP, typename VectorW, typename MatrixZ,
        typename Workspace >
inline typename boost::enable_if< detail::is_workspace< Workspace >,
        std::ptrdiff_t >::type
hpev( const char jobz, const MatrixAP& ap, const VectorW& w,
        const MatrixZ& z, Workspace work ) {
    return hpev_impl< typename bindings::value_type<
            MatrixAP >::type >::invoke( jobz, ap, w, z, work );
}

//
// Overloaded function for hpev. Its overload differs for
// * const MatrixAP&
// * const VectorW&
// * const MatrixZ&
// * Default workspace-type (optimal)
//
template< typename MatrixAP, typename VectorW, typename MatrixZ >
inline typename boost::disable_if< detail::is_workspace< MatrixZ >,
        std::ptrdiff_t >::type
hpev( const char jobz, const MatrixAP& ap, const VectorW& w,
        const MatrixZ& z ) {
    return hpev_impl< typename bindings::value_type<
            MatrixAP >::type >::invoke( jobz, ap, w, z, optimal_workspace() );
}

} // namespace lapack
} // namespace bindings
} // namespace numeric
} // namespace boost

#endif
