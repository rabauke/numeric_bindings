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

#ifndef BOOST_NUMERIC_BINDINGS_LAPACK_COMPUTATIONAL_HSEQR_HPP
#define BOOST_NUMERIC_BINDINGS_LAPACK_COMPUTATIONAL_HSEQR_HPP

#include <boost/assert.hpp>
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
#include <boost/numeric/bindings/traits/detail/utils.hpp>
#include <boost/numeric/bindings/value_type.hpp>
#include <boost/static_assert.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/type_traits/remove_const.hpp>
#include <boost/utility/enable_if.hpp>

//
// The LAPACK-backend for hseqr is the netlib-compatible backend.
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
inline std::ptrdiff_t hseqr( const char job, const char compz,
        const fortran_int_t n, const fortran_int_t ilo,
        const fortran_int_t ihi, float* h, const fortran_int_t ldh, float* wr,
        float* wi, float* z, const fortran_int_t ldz, float* work,
        const fortran_int_t lwork ) {
    fortran_int_t info(0);
    LAPACK_SHSEQR( &job, &compz, &n, &ilo, &ihi, h, &ldh, wr, wi, z, &ldz,
            work, &lwork, &info );
    return info;
}

//
// Overloaded function for dispatching to
// * netlib-compatible LAPACK backend (the default), and
// * double value-type.
//
inline std::ptrdiff_t hseqr( const char job, const char compz,
        const fortran_int_t n, const fortran_int_t ilo,
        const fortran_int_t ihi, double* h, const fortran_int_t ldh,
        double* wr, double* wi, double* z, const fortran_int_t ldz,
        double* work, const fortran_int_t lwork ) {
    fortran_int_t info(0);
    LAPACK_DHSEQR( &job, &compz, &n, &ilo, &ihi, h, &ldh, wr, wi, z, &ldz,
            work, &lwork, &info );
    return info;
}

//
// Overloaded function for dispatching to
// * netlib-compatible LAPACK backend (the default), and
// * complex<float> value-type.
//
inline std::ptrdiff_t hseqr( const char job, const char compz,
        const fortran_int_t n, const fortran_int_t ilo,
        const fortran_int_t ihi, std::complex<float>* h,
        const fortran_int_t ldh, std::complex<float>* w,
        std::complex<float>* z, const fortran_int_t ldz,
        std::complex<float>* work, const fortran_int_t lwork ) {
    fortran_int_t info(0);
    LAPACK_CHSEQR( &job, &compz, &n, &ilo, &ihi, h, &ldh, w, z, &ldz, work,
            &lwork, &info );
    return info;
}

//
// Overloaded function for dispatching to
// * netlib-compatible LAPACK backend (the default), and
// * complex<double> value-type.
//
inline std::ptrdiff_t hseqr( const char job, const char compz,
        const fortran_int_t n, const fortran_int_t ilo,
        const fortran_int_t ihi, std::complex<double>* h,
        const fortran_int_t ldh, std::complex<double>* w,
        std::complex<double>* z, const fortran_int_t ldz,
        std::complex<double>* work, const fortran_int_t lwork ) {
    fortran_int_t info(0);
    LAPACK_ZHSEQR( &job, &compz, &n, &ilo, &ihi, h, &ldh, w, z, &ldz, work,
            &lwork, &info );
    return info;
}

} // namespace detail

//
// Value-type based template class. Use this class if you need a type
// for dispatching to hseqr.
//
template< typename Value, typename Enable = void >
struct hseqr_impl {};

//
// This implementation is enabled if Value is a real type.
//
template< typename Value >
struct hseqr_impl< Value, typename boost::enable_if< is_real< Value > >::type > {

    typedef Value value_type;
    typedef typename remove_imaginary< Value >::type real_type;

    //
    // Static member function for user-defined workspaces, that
    // * Deduces the required arguments for dispatching to LAPACK, and
    // * Asserts that most arguments make sense.
    //
    template< typename MatrixH, typename VectorWR, typename VectorWI,
            typename MatrixZ, typename WORK >
    static std::ptrdiff_t invoke( const char job, const char compz,
            const fortran_int_t ilo, const fortran_int_t ihi,
            MatrixH& h, VectorWR& wr, VectorWI& wi, MatrixZ& z,
            detail::workspace1< WORK > work ) {
        namespace bindings = ::boost::numeric::bindings;
        BOOST_STATIC_ASSERT( (bindings::is_column_major< MatrixH >::value) );
        BOOST_STATIC_ASSERT( (bindings::is_column_major< MatrixZ >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename remove_const<
                typename bindings::value_type< MatrixH >::type >::type,
                typename remove_const< typename bindings::value_type<
                VectorWR >::type >::type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename remove_const<
                typename bindings::value_type< MatrixH >::type >::type,
                typename remove_const< typename bindings::value_type<
                VectorWI >::type >::type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename remove_const<
                typename bindings::value_type< MatrixH >::type >::type,
                typename remove_const< typename bindings::value_type<
                MatrixZ >::type >::type >::value) );
        BOOST_STATIC_ASSERT( (bindings::is_mutable< MatrixH >::value) );
        BOOST_STATIC_ASSERT( (bindings::is_mutable< VectorWR >::value) );
        BOOST_STATIC_ASSERT( (bindings::is_mutable< VectorWI >::value) );
        BOOST_STATIC_ASSERT( (bindings::is_mutable< MatrixZ >::value) );
        BOOST_ASSERT( bindings::size(work.select(real_type())) >=
                min_size_work( bindings::size_column(h) ));
        BOOST_ASSERT( bindings::size(wr) >= bindings::size_column(h) );
        BOOST_ASSERT( bindings::size_column(h) >= 0 );
        BOOST_ASSERT( bindings::size_minor(h) == 1 ||
                bindings::stride_minor(h) == 1 );
        BOOST_ASSERT( bindings::size_minor(z) == 1 ||
                bindings::stride_minor(z) == 1 );
        BOOST_ASSERT( bindings::stride_major(h) >= std::max< std::ptrdiff_t >(1,
                bindings::size_column(h)) );
        BOOST_ASSERT( compz == 'N' || compz == 'I' || compz == 'V' );
        BOOST_ASSERT( job == 'E' || job == 'S' );
        return detail::hseqr( job, compz, bindings::size_column(h), ilo, ihi,
                bindings::begin_value(h), bindings::stride_major(h),
                bindings::begin_value(wr), bindings::begin_value(wi),
                bindings::begin_value(z), bindings::stride_major(z),
                bindings::begin_value(work.select(real_type())),
                bindings::size(work.select(real_type())) );
    }

    //
    // Static member function that
    // * Figures out the minimal workspace requirements, and passes
    //   the results to the user-defined workspace overload of the 
    //   invoke static member function
    // * Enables the unblocked algorithm (BLAS level 2)
    //
    template< typename MatrixH, typename VectorWR, typename VectorWI,
            typename MatrixZ >
    static std::ptrdiff_t invoke( const char job, const char compz,
            const fortran_int_t ilo, const fortran_int_t ihi,
            MatrixH& h, VectorWR& wr, VectorWI& wi, MatrixZ& z,
            minimal_workspace ) {
        namespace bindings = ::boost::numeric::bindings;
        bindings::detail::array< real_type > tmp_work( min_size_work(
                bindings::size_column(h) ) );
        return invoke( job, compz, ilo, ihi, h, wr, wi, z,
                workspace( tmp_work ) );
    }

    //
    // Static member function that
    // * Figures out the optimal workspace requirements, and passes
    //   the results to the user-defined workspace overload of the 
    //   invoke static member
    // * Enables the blocked algorithm (BLAS level 3)
    //
    template< typename MatrixH, typename VectorWR, typename VectorWI,
            typename MatrixZ >
    static std::ptrdiff_t invoke( const char job, const char compz,
            const fortran_int_t ilo, const fortran_int_t ihi,
            MatrixH& h, VectorWR& wr, VectorWI& wi, MatrixZ& z,
            optimal_workspace ) {
        namespace bindings = ::boost::numeric::bindings;
        real_type opt_size_work;
        detail::hseqr( job, compz, bindings::size_column(h), ilo, ihi,
                bindings::begin_value(h), bindings::stride_major(h),
                bindings::begin_value(wr), bindings::begin_value(wi),
                bindings::begin_value(z), bindings::stride_major(z),
                &opt_size_work, -1 );
        bindings::detail::array< real_type > tmp_work(
                traits::detail::to_int( opt_size_work ) );
        return invoke( job, compz, ilo, ihi, h, wr, wi, z,
                workspace( tmp_work ) );
    }

    //
    // Static member function that returns the minimum size of
    // workspace-array work.
    //
    static std::ptrdiff_t min_size_work( const std::ptrdiff_t n ) {
        return std::max< std::ptrdiff_t >(1,n);
    }
};

//
// This implementation is enabled if Value is a complex type.
//
template< typename Value >
struct hseqr_impl< Value, typename boost::enable_if< is_complex< Value > >::type > {

    typedef Value value_type;
    typedef typename remove_imaginary< Value >::type real_type;

    //
    // Static member function for user-defined workspaces, that
    // * Deduces the required arguments for dispatching to LAPACK, and
    // * Asserts that most arguments make sense.
    //
    template< typename MatrixH, typename VectorW, typename MatrixZ,
            typename WORK >
    static std::ptrdiff_t invoke( const char job, const char compz,
            const fortran_int_t ilo, const fortran_int_t ihi,
            MatrixH& h, VectorW& w, MatrixZ& z, detail::workspace1<
            WORK > work ) {
        namespace bindings = ::boost::numeric::bindings;
        BOOST_STATIC_ASSERT( (bindings::is_column_major< MatrixH >::value) );
        BOOST_STATIC_ASSERT( (bindings::is_column_major< MatrixZ >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename remove_const<
                typename bindings::value_type< MatrixH >::type >::type,
                typename remove_const< typename bindings::value_type<
                VectorW >::type >::type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename remove_const<
                typename bindings::value_type< MatrixH >::type >::type,
                typename remove_const< typename bindings::value_type<
                MatrixZ >::type >::type >::value) );
        BOOST_STATIC_ASSERT( (bindings::is_mutable< MatrixH >::value) );
        BOOST_STATIC_ASSERT( (bindings::is_mutable< VectorW >::value) );
        BOOST_STATIC_ASSERT( (bindings::is_mutable< MatrixZ >::value) );
        BOOST_ASSERT( bindings::size(work.select(value_type())) >=
                min_size_work( bindings::size_column(h) ));
        BOOST_ASSERT( bindings::size_column(h) >= 0 );
        BOOST_ASSERT( bindings::size_minor(h) == 1 ||
                bindings::stride_minor(h) == 1 );
        BOOST_ASSERT( bindings::size_minor(z) == 1 ||
                bindings::stride_minor(z) == 1 );
        BOOST_ASSERT( bindings::stride_major(h) >= std::max< std::ptrdiff_t >(1,
                bindings::size_column(h)) );
        BOOST_ASSERT( compz == 'N' || compz == 'I' || compz == 'V' );
        BOOST_ASSERT( job == 'E' || job == 'S' );
        return detail::hseqr( job, compz, bindings::size_column(h), ilo, ihi,
                bindings::begin_value(h), bindings::stride_major(h),
                bindings::begin_value(w), bindings::begin_value(z),
                bindings::stride_major(z),
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
    template< typename MatrixH, typename VectorW, typename MatrixZ >
    static std::ptrdiff_t invoke( const char job, const char compz,
            const fortran_int_t ilo, const fortran_int_t ihi,
            MatrixH& h, VectorW& w, MatrixZ& z, minimal_workspace ) {
        namespace bindings = ::boost::numeric::bindings;
        bindings::detail::array< value_type > tmp_work( min_size_work(
                bindings::size_column(h) ) );
        return invoke( job, compz, ilo, ihi, h, w, z, workspace( tmp_work ) );
    }

    //
    // Static member function that
    // * Figures out the optimal workspace requirements, and passes
    //   the results to the user-defined workspace overload of the 
    //   invoke static member
    // * Enables the blocked algorithm (BLAS level 3)
    //
    template< typename MatrixH, typename VectorW, typename MatrixZ >
    static std::ptrdiff_t invoke( const char job, const char compz,
            const fortran_int_t ilo, const fortran_int_t ihi,
            MatrixH& h, VectorW& w, MatrixZ& z, optimal_workspace ) {
        namespace bindings = ::boost::numeric::bindings;
        value_type opt_size_work;
        detail::hseqr( job, compz, bindings::size_column(h), ilo, ihi,
                bindings::begin_value(h), bindings::stride_major(h),
                bindings::begin_value(w), bindings::begin_value(z),
                bindings::stride_major(z), &opt_size_work, -1 );
        bindings::detail::array< value_type > tmp_work(
                traits::detail::to_int( opt_size_work ) );
        return invoke( job, compz, ilo, ihi, h, w, z, workspace( tmp_work ) );
    }

    //
    // Static member function that returns the minimum size of
    // workspace-array work.
    //
    static std::ptrdiff_t min_size_work( const std::ptrdiff_t n ) {
        return std::max< std::ptrdiff_t >(1,n);
    }
};


//
// Functions for direct use. These functions are overloaded for temporaries,
// so that wrapped types can still be passed and used for write-access. In
// addition, if applicable, they are overloaded for user-defined workspaces.
// Calls to these functions are passed to the hseqr_impl classes. In the 
// documentation, most overloads are collapsed to avoid a large number of
// prototypes which are very similar.
//

//
// Overloaded function for hseqr. Its overload differs for
// * MatrixH&
// * MatrixZ&
// * User-defined workspace
//
template< typename MatrixH, typename VectorWR, typename VectorWI,
        typename MatrixZ, typename Workspace >
inline typename boost::enable_if< detail::is_workspace< Workspace >,
        std::ptrdiff_t >::type
hseqr( const char job, const char compz, const fortran_int_t ilo,
        const fortran_int_t ihi, MatrixH& h, VectorWR& wr, VectorWI& wi,
        MatrixZ& z, Workspace work ) {
    return hseqr_impl< typename bindings::value_type<
            MatrixH >::type >::invoke( job, compz, ilo, ihi, h, wr, wi, z,
            work );
}

//
// Overloaded function for hseqr. Its overload differs for
// * MatrixH&
// * MatrixZ&
// * Default workspace-type (optimal)
//
template< typename MatrixH, typename VectorWR, typename VectorWI,
        typename MatrixZ >
inline typename boost::disable_if< detail::is_workspace< MatrixZ >,
        std::ptrdiff_t >::type
hseqr( const char job, const char compz, const fortran_int_t ilo,
        const fortran_int_t ihi, MatrixH& h, VectorWR& wr, VectorWI& wi,
        MatrixZ& z ) {
    return hseqr_impl< typename bindings::value_type<
            MatrixH >::type >::invoke( job, compz, ilo, ihi, h, wr, wi, z,
            optimal_workspace() );
}

//
// Overloaded function for hseqr. Its overload differs for
// * const MatrixH&
// * MatrixZ&
// * User-defined workspace
//
template< typename MatrixH, typename VectorWR, typename VectorWI,
        typename MatrixZ, typename Workspace >
inline typename boost::enable_if< detail::is_workspace< Workspace >,
        std::ptrdiff_t >::type
hseqr( const char job, const char compz, const fortran_int_t ilo,
        const fortran_int_t ihi, const MatrixH& h, VectorWR& wr,
        VectorWI& wi, MatrixZ& z, Workspace work ) {
    return hseqr_impl< typename bindings::value_type<
            MatrixH >::type >::invoke( job, compz, ilo, ihi, h, wr, wi, z,
            work );
}

//
// Overloaded function for hseqr. Its overload differs for
// * const MatrixH&
// * MatrixZ&
// * Default workspace-type (optimal)
//
template< typename MatrixH, typename VectorWR, typename VectorWI,
        typename MatrixZ >
inline typename boost::disable_if< detail::is_workspace< MatrixZ >,
        std::ptrdiff_t >::type
hseqr( const char job, const char compz, const fortran_int_t ilo,
        const fortran_int_t ihi, const MatrixH& h, VectorWR& wr,
        VectorWI& wi, MatrixZ& z ) {
    return hseqr_impl< typename bindings::value_type<
            MatrixH >::type >::invoke( job, compz, ilo, ihi, h, wr, wi, z,
            optimal_workspace() );
}

//
// Overloaded function for hseqr. Its overload differs for
// * MatrixH&
// * const MatrixZ&
// * User-defined workspace
//
template< typename MatrixH, typename VectorWR, typename VectorWI,
        typename MatrixZ, typename Workspace >
inline typename boost::enable_if< detail::is_workspace< Workspace >,
        std::ptrdiff_t >::type
hseqr( const char job, const char compz, const fortran_int_t ilo,
        const fortran_int_t ihi, MatrixH& h, VectorWR& wr, VectorWI& wi,
        const MatrixZ& z, Workspace work ) {
    return hseqr_impl< typename bindings::value_type<
            MatrixH >::type >::invoke( job, compz, ilo, ihi, h, wr, wi, z,
            work );
}

//
// Overloaded function for hseqr. Its overload differs for
// * MatrixH&
// * const MatrixZ&
// * Default workspace-type (optimal)
//
template< typename MatrixH, typename VectorWR, typename VectorWI,
        typename MatrixZ >
inline typename boost::disable_if< detail::is_workspace< MatrixZ >,
        std::ptrdiff_t >::type
hseqr( const char job, const char compz, const fortran_int_t ilo,
        const fortran_int_t ihi, MatrixH& h, VectorWR& wr, VectorWI& wi,
        const MatrixZ& z ) {
    return hseqr_impl< typename bindings::value_type<
            MatrixH >::type >::invoke( job, compz, ilo, ihi, h, wr, wi, z,
            optimal_workspace() );
}

//
// Overloaded function for hseqr. Its overload differs for
// * const MatrixH&
// * const MatrixZ&
// * User-defined workspace
//
template< typename MatrixH, typename VectorWR, typename VectorWI,
        typename MatrixZ, typename Workspace >
inline typename boost::enable_if< detail::is_workspace< Workspace >,
        std::ptrdiff_t >::type
hseqr( const char job, const char compz, const fortran_int_t ilo,
        const fortran_int_t ihi, const MatrixH& h, VectorWR& wr,
        VectorWI& wi, const MatrixZ& z, Workspace work ) {
    return hseqr_impl< typename bindings::value_type<
            MatrixH >::type >::invoke( job, compz, ilo, ihi, h, wr, wi, z,
            work );
}

//
// Overloaded function for hseqr. Its overload differs for
// * const MatrixH&
// * const MatrixZ&
// * Default workspace-type (optimal)
//
template< typename MatrixH, typename VectorWR, typename VectorWI,
        typename MatrixZ >
inline typename boost::disable_if< detail::is_workspace< MatrixZ >,
        std::ptrdiff_t >::type
hseqr( const char job, const char compz, const fortran_int_t ilo,
        const fortran_int_t ihi, const MatrixH& h, VectorWR& wr,
        VectorWI& wi, const MatrixZ& z ) {
    return hseqr_impl< typename bindings::value_type<
            MatrixH >::type >::invoke( job, compz, ilo, ihi, h, wr, wi, z,
            optimal_workspace() );
}
//
// Overloaded function for hseqr. Its overload differs for
// * MatrixH&
// * MatrixZ&
// * User-defined workspace
//
template< typename MatrixH, typename VectorW, typename MatrixZ,
        typename Workspace >
inline typename boost::enable_if< detail::is_workspace< Workspace >,
        std::ptrdiff_t >::type
hseqr( const char job, const char compz, const fortran_int_t ilo,
        const fortran_int_t ihi, MatrixH& h, VectorW& w, MatrixZ& z,
        Workspace work ) {
    return hseqr_impl< typename bindings::value_type<
            MatrixH >::type >::invoke( job, compz, ilo, ihi, h, w, z, work );
}

//
// Overloaded function for hseqr. Its overload differs for
// * MatrixH&
// * MatrixZ&
// * Default workspace-type (optimal)
//
template< typename MatrixH, typename VectorW, typename MatrixZ >
inline typename boost::disable_if< detail::is_workspace< MatrixZ >,
        std::ptrdiff_t >::type
hseqr( const char job, const char compz, const fortran_int_t ilo,
        const fortran_int_t ihi, MatrixH& h, VectorW& w, MatrixZ& z ) {
    return hseqr_impl< typename bindings::value_type<
            MatrixH >::type >::invoke( job, compz, ilo, ihi, h, w, z,
            optimal_workspace() );
}

//
// Overloaded function for hseqr. Its overload differs for
// * const MatrixH&
// * MatrixZ&
// * User-defined workspace
//
template< typename MatrixH, typename VectorW, typename MatrixZ,
        typename Workspace >
inline typename boost::enable_if< detail::is_workspace< Workspace >,
        std::ptrdiff_t >::type
hseqr( const char job, const char compz, const fortran_int_t ilo,
        const fortran_int_t ihi, const MatrixH& h, VectorW& w, MatrixZ& z,
        Workspace work ) {
    return hseqr_impl< typename bindings::value_type<
            MatrixH >::type >::invoke( job, compz, ilo, ihi, h, w, z, work );
}

//
// Overloaded function for hseqr. Its overload differs for
// * const MatrixH&
// * MatrixZ&
// * Default workspace-type (optimal)
//
template< typename MatrixH, typename VectorW, typename MatrixZ >
inline typename boost::disable_if< detail::is_workspace< MatrixZ >,
        std::ptrdiff_t >::type
hseqr( const char job, const char compz, const fortran_int_t ilo,
        const fortran_int_t ihi, const MatrixH& h, VectorW& w,
        MatrixZ& z ) {
    return hseqr_impl< typename bindings::value_type<
            MatrixH >::type >::invoke( job, compz, ilo, ihi, h, w, z,
            optimal_workspace() );
}

//
// Overloaded function for hseqr. Its overload differs for
// * MatrixH&
// * const MatrixZ&
// * User-defined workspace
//
template< typename MatrixH, typename VectorW, typename MatrixZ,
        typename Workspace >
inline typename boost::enable_if< detail::is_workspace< Workspace >,
        std::ptrdiff_t >::type
hseqr( const char job, const char compz, const fortran_int_t ilo,
        const fortran_int_t ihi, MatrixH& h, VectorW& w, const MatrixZ& z,
        Workspace work ) {
    return hseqr_impl< typename bindings::value_type<
            MatrixH >::type >::invoke( job, compz, ilo, ihi, h, w, z, work );
}

//
// Overloaded function for hseqr. Its overload differs for
// * MatrixH&
// * const MatrixZ&
// * Default workspace-type (optimal)
//
template< typename MatrixH, typename VectorW, typename MatrixZ >
inline typename boost::disable_if< detail::is_workspace< MatrixZ >,
        std::ptrdiff_t >::type
hseqr( const char job, const char compz, const fortran_int_t ilo,
        const fortran_int_t ihi, MatrixH& h, VectorW& w,
        const MatrixZ& z ) {
    return hseqr_impl< typename bindings::value_type<
            MatrixH >::type >::invoke( job, compz, ilo, ihi, h, w, z,
            optimal_workspace() );
}

//
// Overloaded function for hseqr. Its overload differs for
// * const MatrixH&
// * const MatrixZ&
// * User-defined workspace
//
template< typename MatrixH, typename VectorW, typename MatrixZ,
        typename Workspace >
inline typename boost::enable_if< detail::is_workspace< Workspace >,
        std::ptrdiff_t >::type
hseqr( const char job, const char compz, const fortran_int_t ilo,
        const fortran_int_t ihi, const MatrixH& h, VectorW& w,
        const MatrixZ& z, Workspace work ) {
    return hseqr_impl< typename bindings::value_type<
            MatrixH >::type >::invoke( job, compz, ilo, ihi, h, w, z, work );
}

//
// Overloaded function for hseqr. Its overload differs for
// * const MatrixH&
// * const MatrixZ&
// * Default workspace-type (optimal)
//
template< typename MatrixH, typename VectorW, typename MatrixZ >
inline typename boost::disable_if< detail::is_workspace< MatrixZ >,
        std::ptrdiff_t >::type
hseqr( const char job, const char compz, const fortran_int_t ilo,
        const fortran_int_t ihi, const MatrixH& h, VectorW& w,
        const MatrixZ& z ) {
    return hseqr_impl< typename bindings::value_type<
            MatrixH >::type >::invoke( job, compz, ilo, ihi, h, w, z,
            optimal_workspace() );
}

} // namespace lapack
} // namespace bindings
} // namespace numeric
} // namespace boost

#endif
