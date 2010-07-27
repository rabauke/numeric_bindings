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

#ifndef BOOST_NUMERIC_BINDINGS_LAPACK_DRIVER_HBGV_HPP
#define BOOST_NUMERIC_BINDINGS_LAPACK_DRIVER_HBGV_HPP

#include <boost/assert.hpp>
#include <boost/numeric/bindings/bandwidth.hpp>
#include <boost/numeric/bindings/begin.hpp>
#include <boost/numeric/bindings/detail/array.hpp>
#include <boost/numeric/bindings/is_column_major.hpp>
#include <boost/numeric/bindings/is_complex.hpp>
#include <boost/numeric/bindings/is_mutable.hpp>
#include <boost/numeric/bindings/is_real.hpp>
#include <boost/numeric/bindings/lapack/workspace.hpp>
#include <boost/numeric/bindings/remove_imaginary.hpp>
#include <boost/numeric/bindings/size.hpp>
#include <boost/numeric/bindings/stride.hpp>
#include <boost/numeric/bindings/uplo_tag.hpp>
#include <boost/numeric/bindings/value_type.hpp>
#include <boost/static_assert.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/type_traits/remove_const.hpp>
#include <boost/utility/enable_if.hpp>

//
// The LAPACK-backend for hbgv is the netlib-compatible backend.
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
inline std::ptrdiff_t hbgv( const char jobz, const UpLo uplo,
        const fortran_int_t n, const fortran_int_t ka, const fortran_int_t kb,
        float* ab, const fortran_int_t ldab, float* bb,
        const fortran_int_t ldbb, float* w, float* z, const fortran_int_t ldz,
        float* work ) {
    fortran_int_t info(0);
    LAPACK_SSBGV( &jobz, &lapack_option< UpLo >::value, &n, &ka, &kb, ab,
            &ldab, bb, &ldbb, w, z, &ldz, work, &info );
    return info;
}

//
// Overloaded function for dispatching to
// * netlib-compatible LAPACK backend (the default), and
// * double value-type.
//
template< typename UpLo >
inline std::ptrdiff_t hbgv( const char jobz, const UpLo uplo,
        const fortran_int_t n, const fortran_int_t ka, const fortran_int_t kb,
        double* ab, const fortran_int_t ldab, double* bb,
        const fortran_int_t ldbb, double* w, double* z,
        const fortran_int_t ldz, double* work ) {
    fortran_int_t info(0);
    LAPACK_DSBGV( &jobz, &lapack_option< UpLo >::value, &n, &ka, &kb, ab,
            &ldab, bb, &ldbb, w, z, &ldz, work, &info );
    return info;
}

//
// Overloaded function for dispatching to
// * netlib-compatible LAPACK backend (the default), and
// * complex<float> value-type.
//
template< typename UpLo >
inline std::ptrdiff_t hbgv( const char jobz, const UpLo uplo,
        const fortran_int_t n, const fortran_int_t ka, const fortran_int_t kb,
        std::complex<float>* ab, const fortran_int_t ldab,
        std::complex<float>* bb, const fortran_int_t ldbb, float* w,
        std::complex<float>* z, const fortran_int_t ldz,
        std::complex<float>* work, float* rwork ) {
    fortran_int_t info(0);
    LAPACK_CHBGV( &jobz, &lapack_option< UpLo >::value, &n, &ka, &kb, ab,
            &ldab, bb, &ldbb, w, z, &ldz, work, rwork, &info );
    return info;
}

//
// Overloaded function for dispatching to
// * netlib-compatible LAPACK backend (the default), and
// * complex<double> value-type.
//
template< typename UpLo >
inline std::ptrdiff_t hbgv( const char jobz, const UpLo uplo,
        const fortran_int_t n, const fortran_int_t ka, const fortran_int_t kb,
        std::complex<double>* ab, const fortran_int_t ldab,
        std::complex<double>* bb, const fortran_int_t ldbb, double* w,
        std::complex<double>* z, const fortran_int_t ldz,
        std::complex<double>* work, double* rwork ) {
    fortran_int_t info(0);
    LAPACK_ZHBGV( &jobz, &lapack_option< UpLo >::value, &n, &ka, &kb, ab,
            &ldab, bb, &ldbb, w, z, &ldz, work, rwork, &info );
    return info;
}

} // namespace detail

//
// Value-type based template class. Use this class if you need a type
// for dispatching to hbgv.
//
template< typename Value, typename Enable = void >
struct hbgv_impl {};

//
// This implementation is enabled if Value is a real type.
//
template< typename Value >
struct hbgv_impl< Value, typename boost::enable_if< is_real< Value > >::type > {

    typedef Value value_type;
    typedef typename remove_imaginary< Value >::type real_type;

    //
    // Static member function for user-defined workspaces, that
    // * Deduces the required arguments for dispatching to LAPACK, and
    // * Asserts that most arguments make sense.
    //
    template< typename MatrixAB, typename MatrixBB, typename VectorW,
            typename MatrixZ, typename WORK >
    static std::ptrdiff_t invoke( const char jobz, MatrixAB& ab, MatrixBB& bb,
            VectorW& w, MatrixZ& z, detail::workspace1< WORK > work ) {
        namespace bindings = ::boost::numeric::bindings;
        typedef typename result_of::uplo_tag< MatrixAB >::type uplo;
        BOOST_STATIC_ASSERT( (bindings::is_column_major< MatrixAB >::value) );
        BOOST_STATIC_ASSERT( (bindings::is_column_major< MatrixBB >::value) );
        BOOST_STATIC_ASSERT( (bindings::is_column_major< MatrixZ >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename remove_const<
                typename bindings::value_type< MatrixAB >::type >::type,
                typename remove_const< typename bindings::value_type<
                MatrixBB >::type >::type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename remove_const<
                typename bindings::value_type< MatrixAB >::type >::type,
                typename remove_const< typename bindings::value_type<
                VectorW >::type >::type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename remove_const<
                typename bindings::value_type< MatrixAB >::type >::type,
                typename remove_const< typename bindings::value_type<
                MatrixZ >::type >::type >::value) );
        BOOST_STATIC_ASSERT( (bindings::is_mutable< MatrixAB >::value) );
        BOOST_STATIC_ASSERT( (bindings::is_mutable< MatrixBB >::value) );
        BOOST_STATIC_ASSERT( (bindings::is_mutable< VectorW >::value) );
        BOOST_STATIC_ASSERT( (bindings::is_mutable< MatrixZ >::value) );
        BOOST_ASSERT( bindings::bandwidth(ab, uplo()) >= 0 );
        BOOST_ASSERT( bindings::bandwidth(bb, uplo()) >= 0 );
        BOOST_ASSERT( bindings::size(work.select(real_type())) >=
                min_size_work( bindings::size_column(ab) ));
        BOOST_ASSERT( bindings::size_column(ab) >= 0 );
        BOOST_ASSERT( bindings::size_minor(ab) == 1 ||
                bindings::stride_minor(ab) == 1 );
        BOOST_ASSERT( bindings::size_minor(bb) == 1 ||
                bindings::stride_minor(bb) == 1 );
        BOOST_ASSERT( bindings::size_minor(z) == 1 ||
                bindings::stride_minor(z) == 1 );
        BOOST_ASSERT( bindings::stride_major(ab) >= bindings::bandwidth(ab,
                uplo())+1 );
        BOOST_ASSERT( bindings::stride_major(bb) >= bindings::bandwidth(bb,
                uplo())+1 );
        BOOST_ASSERT( jobz == 'N' || jobz == 'V' );
        return detail::hbgv( jobz, uplo(), bindings::size_column(ab),
                bindings::bandwidth(ab, uplo()), bindings::bandwidth(bb,
                uplo()), bindings::begin_value(ab),
                bindings::stride_major(ab), bindings::begin_value(bb),
                bindings::stride_major(bb), bindings::begin_value(w),
                bindings::begin_value(z), bindings::stride_major(z),
                bindings::begin_value(work.select(real_type())) );
    }

    //
    // Static member function that
    // * Figures out the minimal workspace requirements, and passes
    //   the results to the user-defined workspace overload of the 
    //   invoke static member function
    // * Enables the unblocked algorithm (BLAS level 2)
    //
    template< typename MatrixAB, typename MatrixBB, typename VectorW,
            typename MatrixZ >
    static std::ptrdiff_t invoke( const char jobz, MatrixAB& ab, MatrixBB& bb,
            VectorW& w, MatrixZ& z, minimal_workspace ) {
        namespace bindings = ::boost::numeric::bindings;
        typedef typename result_of::uplo_tag< MatrixAB >::type uplo;
        bindings::detail::array< real_type > tmp_work( min_size_work(
                bindings::size_column(ab) ) );
        return invoke( jobz, ab, bb, w, z, workspace( tmp_work ) );
    }

    //
    // Static member function that
    // * Figures out the optimal workspace requirements, and passes
    //   the results to the user-defined workspace overload of the 
    //   invoke static member
    // * Enables the blocked algorithm (BLAS level 3)
    //
    template< typename MatrixAB, typename MatrixBB, typename VectorW,
            typename MatrixZ >
    static std::ptrdiff_t invoke( const char jobz, MatrixAB& ab, MatrixBB& bb,
            VectorW& w, MatrixZ& z, optimal_workspace ) {
        namespace bindings = ::boost::numeric::bindings;
        typedef typename result_of::uplo_tag< MatrixAB >::type uplo;
        return invoke( jobz, ab, bb, w, z, minimal_workspace() );
    }

    //
    // Static member function that returns the minimum size of
    // workspace-array work.
    //
    static std::ptrdiff_t min_size_work( const std::ptrdiff_t n ) {
        return 3*n;
    }
};

//
// This implementation is enabled if Value is a complex type.
//
template< typename Value >
struct hbgv_impl< Value, typename boost::enable_if< is_complex< Value > >::type > {

    typedef Value value_type;
    typedef typename remove_imaginary< Value >::type real_type;

    //
    // Static member function for user-defined workspaces, that
    // * Deduces the required arguments for dispatching to LAPACK, and
    // * Asserts that most arguments make sense.
    //
    template< typename MatrixAB, typename MatrixBB, typename VectorW,
            typename MatrixZ, typename WORK, typename RWORK >
    static std::ptrdiff_t invoke( const char jobz, MatrixAB& ab, MatrixBB& bb,
            VectorW& w, MatrixZ& z, detail::workspace2< WORK, RWORK > work ) {
        namespace bindings = ::boost::numeric::bindings;
        typedef typename result_of::uplo_tag< MatrixAB >::type uplo;
        BOOST_STATIC_ASSERT( (bindings::is_column_major< MatrixAB >::value) );
        BOOST_STATIC_ASSERT( (bindings::is_column_major< MatrixBB >::value) );
        BOOST_STATIC_ASSERT( (bindings::is_column_major< MatrixZ >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename remove_const<
                typename bindings::value_type< MatrixAB >::type >::type,
                typename remove_const< typename bindings::value_type<
                MatrixBB >::type >::type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename remove_const<
                typename bindings::value_type< MatrixAB >::type >::type,
                typename remove_const< typename bindings::value_type<
                MatrixZ >::type >::type >::value) );
        BOOST_STATIC_ASSERT( (bindings::is_mutable< MatrixAB >::value) );
        BOOST_STATIC_ASSERT( (bindings::is_mutable< MatrixBB >::value) );
        BOOST_STATIC_ASSERT( (bindings::is_mutable< VectorW >::value) );
        BOOST_STATIC_ASSERT( (bindings::is_mutable< MatrixZ >::value) );
        BOOST_ASSERT( bindings::bandwidth(ab, uplo()) >= 0 );
        BOOST_ASSERT( bindings::bandwidth(bb, uplo()) >= 0 );
        BOOST_ASSERT( bindings::size(work.select(real_type())) >=
                min_size_rwork( bindings::size_column(ab) ));
        BOOST_ASSERT( bindings::size(work.select(value_type())) >=
                min_size_work( bindings::size_column(ab) ));
        BOOST_ASSERT( bindings::size_column(ab) >= 0 );
        BOOST_ASSERT( bindings::size_minor(ab) == 1 ||
                bindings::stride_minor(ab) == 1 );
        BOOST_ASSERT( bindings::size_minor(bb) == 1 ||
                bindings::stride_minor(bb) == 1 );
        BOOST_ASSERT( bindings::size_minor(z) == 1 ||
                bindings::stride_minor(z) == 1 );
        BOOST_ASSERT( bindings::stride_major(ab) >= bindings::bandwidth(ab,
                uplo())+1 );
        BOOST_ASSERT( bindings::stride_major(bb) >= bindings::bandwidth(bb,
                uplo())+1 );
        BOOST_ASSERT( jobz == 'N' || jobz == 'V' );
        return detail::hbgv( jobz, uplo(), bindings::size_column(ab),
                bindings::bandwidth(ab, uplo()), bindings::bandwidth(bb,
                uplo()), bindings::begin_value(ab),
                bindings::stride_major(ab), bindings::begin_value(bb),
                bindings::stride_major(bb), bindings::begin_value(w),
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
    template< typename MatrixAB, typename MatrixBB, typename VectorW,
            typename MatrixZ >
    static std::ptrdiff_t invoke( const char jobz, MatrixAB& ab, MatrixBB& bb,
            VectorW& w, MatrixZ& z, minimal_workspace ) {
        namespace bindings = ::boost::numeric::bindings;
        typedef typename result_of::uplo_tag< MatrixAB >::type uplo;
        bindings::detail::array< value_type > tmp_work( min_size_work(
                bindings::size_column(ab) ) );
        bindings::detail::array< real_type > tmp_rwork( min_size_rwork(
                bindings::size_column(ab) ) );
        return invoke( jobz, ab, bb, w, z, workspace( tmp_work, tmp_rwork ) );
    }

    //
    // Static member function that
    // * Figures out the optimal workspace requirements, and passes
    //   the results to the user-defined workspace overload of the 
    //   invoke static member
    // * Enables the blocked algorithm (BLAS level 3)
    //
    template< typename MatrixAB, typename MatrixBB, typename VectorW,
            typename MatrixZ >
    static std::ptrdiff_t invoke( const char jobz, MatrixAB& ab, MatrixBB& bb,
            VectorW& w, MatrixZ& z, optimal_workspace ) {
        namespace bindings = ::boost::numeric::bindings;
        typedef typename result_of::uplo_tag< MatrixAB >::type uplo;
        return invoke( jobz, ab, bb, w, z, minimal_workspace() );
    }

    //
    // Static member function that returns the minimum size of
    // workspace-array work.
    //
    static std::ptrdiff_t min_size_work( const std::ptrdiff_t n ) {
        return n;
    }

    //
    // Static member function that returns the minimum size of
    // workspace-array rwork.
    //
    static std::ptrdiff_t min_size_rwork( const std::ptrdiff_t n ) {
        return 3*n;
    }
};


//
// Functions for direct use. These functions are overloaded for temporaries,
// so that wrapped types can still be passed and used for write-access. In
// addition, if applicable, they are overloaded for user-defined workspaces.
// Calls to these functions are passed to the hbgv_impl classes. In the 
// documentation, most overloads are collapsed to avoid a large number of
// prototypes which are very similar.
//

//
// Overloaded function for hbgv. Its overload differs for
// * MatrixAB&
// * MatrixBB&
// * MatrixZ&
// * User-defined workspace
//
template< typename MatrixAB, typename MatrixBB, typename VectorW,
        typename MatrixZ, typename Workspace >
inline typename boost::enable_if< detail::is_workspace< Workspace >,
        std::ptrdiff_t >::type
hbgv( const char jobz, MatrixAB& ab, MatrixBB& bb, VectorW& w,
        MatrixZ& z, Workspace work ) {
    return hbgv_impl< typename bindings::value_type<
            MatrixAB >::type >::invoke( jobz, ab, bb, w, z, work );
}

//
// Overloaded function for hbgv. Its overload differs for
// * MatrixAB&
// * MatrixBB&
// * MatrixZ&
// * Default workspace-type (optimal)
//
template< typename MatrixAB, typename MatrixBB, typename VectorW,
        typename MatrixZ >
inline typename boost::disable_if< detail::is_workspace< MatrixZ >,
        std::ptrdiff_t >::type
hbgv( const char jobz, MatrixAB& ab, MatrixBB& bb, VectorW& w,
        MatrixZ& z ) {
    return hbgv_impl< typename bindings::value_type<
            MatrixAB >::type >::invoke( jobz, ab, bb, w, z,
            optimal_workspace() );
}

//
// Overloaded function for hbgv. Its overload differs for
// * const MatrixAB&
// * MatrixBB&
// * MatrixZ&
// * User-defined workspace
//
template< typename MatrixAB, typename MatrixBB, typename VectorW,
        typename MatrixZ, typename Workspace >
inline typename boost::enable_if< detail::is_workspace< Workspace >,
        std::ptrdiff_t >::type
hbgv( const char jobz, const MatrixAB& ab, MatrixBB& bb, VectorW& w,
        MatrixZ& z, Workspace work ) {
    return hbgv_impl< typename bindings::value_type<
            MatrixAB >::type >::invoke( jobz, ab, bb, w, z, work );
}

//
// Overloaded function for hbgv. Its overload differs for
// * const MatrixAB&
// * MatrixBB&
// * MatrixZ&
// * Default workspace-type (optimal)
//
template< typename MatrixAB, typename MatrixBB, typename VectorW,
        typename MatrixZ >
inline typename boost::disable_if< detail::is_workspace< MatrixZ >,
        std::ptrdiff_t >::type
hbgv( const char jobz, const MatrixAB& ab, MatrixBB& bb, VectorW& w,
        MatrixZ& z ) {
    return hbgv_impl< typename bindings::value_type<
            MatrixAB >::type >::invoke( jobz, ab, bb, w, z,
            optimal_workspace() );
}

//
// Overloaded function for hbgv. Its overload differs for
// * MatrixAB&
// * const MatrixBB&
// * MatrixZ&
// * User-defined workspace
//
template< typename MatrixAB, typename MatrixBB, typename VectorW,
        typename MatrixZ, typename Workspace >
inline typename boost::enable_if< detail::is_workspace< Workspace >,
        std::ptrdiff_t >::type
hbgv( const char jobz, MatrixAB& ab, const MatrixBB& bb, VectorW& w,
        MatrixZ& z, Workspace work ) {
    return hbgv_impl< typename bindings::value_type<
            MatrixAB >::type >::invoke( jobz, ab, bb, w, z, work );
}

//
// Overloaded function for hbgv. Its overload differs for
// * MatrixAB&
// * const MatrixBB&
// * MatrixZ&
// * Default workspace-type (optimal)
//
template< typename MatrixAB, typename MatrixBB, typename VectorW,
        typename MatrixZ >
inline typename boost::disable_if< detail::is_workspace< MatrixZ >,
        std::ptrdiff_t >::type
hbgv( const char jobz, MatrixAB& ab, const MatrixBB& bb, VectorW& w,
        MatrixZ& z ) {
    return hbgv_impl< typename bindings::value_type<
            MatrixAB >::type >::invoke( jobz, ab, bb, w, z,
            optimal_workspace() );
}

//
// Overloaded function for hbgv. Its overload differs for
// * const MatrixAB&
// * const MatrixBB&
// * MatrixZ&
// * User-defined workspace
//
template< typename MatrixAB, typename MatrixBB, typename VectorW,
        typename MatrixZ, typename Workspace >
inline typename boost::enable_if< detail::is_workspace< Workspace >,
        std::ptrdiff_t >::type
hbgv( const char jobz, const MatrixAB& ab, const MatrixBB& bb,
        VectorW& w, MatrixZ& z, Workspace work ) {
    return hbgv_impl< typename bindings::value_type<
            MatrixAB >::type >::invoke( jobz, ab, bb, w, z, work );
}

//
// Overloaded function for hbgv. Its overload differs for
// * const MatrixAB&
// * const MatrixBB&
// * MatrixZ&
// * Default workspace-type (optimal)
//
template< typename MatrixAB, typename MatrixBB, typename VectorW,
        typename MatrixZ >
inline typename boost::disable_if< detail::is_workspace< MatrixZ >,
        std::ptrdiff_t >::type
hbgv( const char jobz, const MatrixAB& ab, const MatrixBB& bb,
        VectorW& w, MatrixZ& z ) {
    return hbgv_impl< typename bindings::value_type<
            MatrixAB >::type >::invoke( jobz, ab, bb, w, z,
            optimal_workspace() );
}

//
// Overloaded function for hbgv. Its overload differs for
// * MatrixAB&
// * MatrixBB&
// * const MatrixZ&
// * User-defined workspace
//
template< typename MatrixAB, typename MatrixBB, typename VectorW,
        typename MatrixZ, typename Workspace >
inline typename boost::enable_if< detail::is_workspace< Workspace >,
        std::ptrdiff_t >::type
hbgv( const char jobz, MatrixAB& ab, MatrixBB& bb, VectorW& w,
        const MatrixZ& z, Workspace work ) {
    return hbgv_impl< typename bindings::value_type<
            MatrixAB >::type >::invoke( jobz, ab, bb, w, z, work );
}

//
// Overloaded function for hbgv. Its overload differs for
// * MatrixAB&
// * MatrixBB&
// * const MatrixZ&
// * Default workspace-type (optimal)
//
template< typename MatrixAB, typename MatrixBB, typename VectorW,
        typename MatrixZ >
inline typename boost::disable_if< detail::is_workspace< MatrixZ >,
        std::ptrdiff_t >::type
hbgv( const char jobz, MatrixAB& ab, MatrixBB& bb, VectorW& w,
        const MatrixZ& z ) {
    return hbgv_impl< typename bindings::value_type<
            MatrixAB >::type >::invoke( jobz, ab, bb, w, z,
            optimal_workspace() );
}

//
// Overloaded function for hbgv. Its overload differs for
// * const MatrixAB&
// * MatrixBB&
// * const MatrixZ&
// * User-defined workspace
//
template< typename MatrixAB, typename MatrixBB, typename VectorW,
        typename MatrixZ, typename Workspace >
inline typename boost::enable_if< detail::is_workspace< Workspace >,
        std::ptrdiff_t >::type
hbgv( const char jobz, const MatrixAB& ab, MatrixBB& bb, VectorW& w,
        const MatrixZ& z, Workspace work ) {
    return hbgv_impl< typename bindings::value_type<
            MatrixAB >::type >::invoke( jobz, ab, bb, w, z, work );
}

//
// Overloaded function for hbgv. Its overload differs for
// * const MatrixAB&
// * MatrixBB&
// * const MatrixZ&
// * Default workspace-type (optimal)
//
template< typename MatrixAB, typename MatrixBB, typename VectorW,
        typename MatrixZ >
inline typename boost::disable_if< detail::is_workspace< MatrixZ >,
        std::ptrdiff_t >::type
hbgv( const char jobz, const MatrixAB& ab, MatrixBB& bb, VectorW& w,
        const MatrixZ& z ) {
    return hbgv_impl< typename bindings::value_type<
            MatrixAB >::type >::invoke( jobz, ab, bb, w, z,
            optimal_workspace() );
}

//
// Overloaded function for hbgv. Its overload differs for
// * MatrixAB&
// * const MatrixBB&
// * const MatrixZ&
// * User-defined workspace
//
template< typename MatrixAB, typename MatrixBB, typename VectorW,
        typename MatrixZ, typename Workspace >
inline typename boost::enable_if< detail::is_workspace< Workspace >,
        std::ptrdiff_t >::type
hbgv( const char jobz, MatrixAB& ab, const MatrixBB& bb, VectorW& w,
        const MatrixZ& z, Workspace work ) {
    return hbgv_impl< typename bindings::value_type<
            MatrixAB >::type >::invoke( jobz, ab, bb, w, z, work );
}

//
// Overloaded function for hbgv. Its overload differs for
// * MatrixAB&
// * const MatrixBB&
// * const MatrixZ&
// * Default workspace-type (optimal)
//
template< typename MatrixAB, typename MatrixBB, typename VectorW,
        typename MatrixZ >
inline typename boost::disable_if< detail::is_workspace< MatrixZ >,
        std::ptrdiff_t >::type
hbgv( const char jobz, MatrixAB& ab, const MatrixBB& bb, VectorW& w,
        const MatrixZ& z ) {
    return hbgv_impl< typename bindings::value_type<
            MatrixAB >::type >::invoke( jobz, ab, bb, w, z,
            optimal_workspace() );
}

//
// Overloaded function for hbgv. Its overload differs for
// * const MatrixAB&
// * const MatrixBB&
// * const MatrixZ&
// * User-defined workspace
//
template< typename MatrixAB, typename MatrixBB, typename VectorW,
        typename MatrixZ, typename Workspace >
inline typename boost::enable_if< detail::is_workspace< Workspace >,
        std::ptrdiff_t >::type
hbgv( const char jobz, const MatrixAB& ab, const MatrixBB& bb,
        VectorW& w, const MatrixZ& z, Workspace work ) {
    return hbgv_impl< typename bindings::value_type<
            MatrixAB >::type >::invoke( jobz, ab, bb, w, z, work );
}

//
// Overloaded function for hbgv. Its overload differs for
// * const MatrixAB&
// * const MatrixBB&
// * const MatrixZ&
// * Default workspace-type (optimal)
//
template< typename MatrixAB, typename MatrixBB, typename VectorW,
        typename MatrixZ >
inline typename boost::disable_if< detail::is_workspace< MatrixZ >,
        std::ptrdiff_t >::type
hbgv( const char jobz, const MatrixAB& ab, const MatrixBB& bb,
        VectorW& w, const MatrixZ& z ) {
    return hbgv_impl< typename bindings::value_type<
            MatrixAB >::type >::invoke( jobz, ab, bb, w, z,
            optimal_workspace() );
}

} // namespace lapack
} // namespace bindings
} // namespace numeric
} // namespace boost

#endif
