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

#ifndef BOOST_NUMERIC_BINDINGS_LAPACK_COMPUTATIONAL_TGEVC_HPP
#define BOOST_NUMERIC_BINDINGS_LAPACK_COMPUTATIONAL_TGEVC_HPP

#include <boost/assert.hpp>
#include <boost/numeric/bindings/begin.hpp>
#include <boost/numeric/bindings/detail/array.hpp>
#include <boost/numeric/bindings/is_complex.hpp>
#include <boost/numeric/bindings/is_mutable.hpp>
#include <boost/numeric/bindings/is_real.hpp>
#include <boost/numeric/bindings/lapack/detail/lapack.h>
#include <boost/numeric/bindings/lapack/detail/lapack_option.hpp>
#include <boost/numeric/bindings/lapack/workspace.hpp>
#include <boost/numeric/bindings/remove_imaginary.hpp>
#include <boost/numeric/bindings/size.hpp>
#include <boost/numeric/bindings/stride.hpp>
#include <boost/numeric/bindings/value.hpp>
#include <boost/static_assert.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/type_traits/remove_const.hpp>
#include <boost/utility/enable_if.hpp>

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
// Overloaded function for dispatching to float value-type.
//
inline void tgevc( char side, char howmny, const logical_t* select,
        fortran_int_t n, const float* s, fortran_int_t lds, const float* p,
        fortran_int_t ldp, float* vl, fortran_int_t ldvl, float* vr,
        fortran_int_t ldvr, fortran_int_t mm, fortran_int_t& m, float* work,
        fortran_int_t& info ) {
    LAPACK_STGEVC( &side, &howmny, select, &n, s, &lds, p, &ldp, vl, &ldvl,
            vr, &ldvr, &mm, &m, work, &info );
}

//
// Overloaded function for dispatching to double value-type.
//
inline void tgevc( char side, char howmny, const logical_t* select,
        fortran_int_t n, const double* s, fortran_int_t lds, const double* p,
        fortran_int_t ldp, double* vl, fortran_int_t ldvl, double* vr,
        fortran_int_t ldvr, fortran_int_t mm, fortran_int_t& m, double* work,
        fortran_int_t& info ) {
    LAPACK_DTGEVC( &side, &howmny, select, &n, s, &lds, p, &ldp, vl, &ldvl,
            vr, &ldvr, &mm, &m, work, &info );
}

//
// Overloaded function for dispatching to complex<float> value-type.
//
inline void tgevc( char side, char howmny, const logical_t* select,
        fortran_int_t n, const std::complex<float>* s, fortran_int_t lds,
        const std::complex<float>* p, fortran_int_t ldp,
        std::complex<float>* vl, fortran_int_t ldvl, std::complex<float>* vr,
        fortran_int_t ldvr, fortran_int_t mm, fortran_int_t& m,
        std::complex<float>* work, float* rwork, fortran_int_t& info ) {
    LAPACK_CTGEVC( &side, &howmny, select, &n, s, &lds, p, &ldp, vl, &ldvl,
            vr, &ldvr, &mm, &m, work, rwork, &info );
}

//
// Overloaded function for dispatching to complex<double> value-type.
//
inline void tgevc( char side, char howmny, const logical_t* select,
        fortran_int_t n, const std::complex<double>* s, fortran_int_t lds,
        const std::complex<double>* p, fortran_int_t ldp,
        std::complex<double>* vl, fortran_int_t ldvl,
        std::complex<double>* vr, fortran_int_t ldvr, fortran_int_t mm,
        fortran_int_t& m, std::complex<double>* work, double* rwork,
        fortran_int_t& info ) {
    LAPACK_ZTGEVC( &side, &howmny, select, &n, s, &lds, p, &ldp, vl, &ldvl,
            vr, &ldvr, &mm, &m, work, rwork, &info );
}

} // namespace detail

//
// Value-type based template class. Use this class if you need a type
// for dispatching to tgevc.
//
template< typename Value, typename Enable = void >
struct tgevc_impl {};

//
// This implementation is enabled if Value is a real type.
//
template< typename Value >
struct tgevc_impl< Value, typename boost::enable_if< is_real< Value > >::type > {

    typedef Value value_type;
    typedef typename remove_imaginary< Value >::type real_type;
    typedef tag::column_major order;

    //
    // Static member function for user-defined workspaces, that
    // * Deduces the required arguments for dispatching to LAPACK, and
    // * Asserts that most arguments make sense.
    //
    template< typename VectorSELECT, typename MatrixS, typename MatrixP,
            typename MatrixVL, typename MatrixVR, typename WORK >
    static void invoke( const char side, const char howmny,
            const VectorSELECT& select, const fortran_int_t n,
            const MatrixS& s, const MatrixP& p, MatrixVL& vl, MatrixVR& vr,
            const fortran_int_t mm, fortran_int_t& m,
            fortran_int_t& info, detail::workspace1< WORK > work ) {
        BOOST_STATIC_ASSERT( (boost::is_same< typename remove_const<
                typename value< MatrixS >::type >::type,
                typename remove_const< typename value<
                MatrixP >::type >::type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename remove_const<
                typename value< MatrixS >::type >::type,
                typename remove_const< typename value<
                MatrixVL >::type >::type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename remove_const<
                typename value< MatrixS >::type >::type,
                typename remove_const< typename value<
                MatrixVR >::type >::type >::value) );
        BOOST_STATIC_ASSERT( (is_mutable< MatrixVL >::value) );
        BOOST_STATIC_ASSERT( (is_mutable< MatrixVR >::value) );
        BOOST_ASSERT( howmny == 'A' || howmny == 'B' || howmny == 'S' );
        BOOST_ASSERT( mm >= m );
        BOOST_ASSERT( n >= 0 );
        BOOST_ASSERT( side == 'R' || side == 'L' || side == 'B' );
        BOOST_ASSERT( size(work.select(real_type())) >= min_size_work( n ));
        BOOST_ASSERT( size_minor(p) == 1 || stride_minor(p) == 1 );
        BOOST_ASSERT( size_minor(s) == 1 || stride_minor(s) == 1 );
        BOOST_ASSERT( size_minor(vl) == 1 || stride_minor(vl) == 1 );
        BOOST_ASSERT( size_minor(vr) == 1 || stride_minor(vr) == 1 );
        BOOST_ASSERT( stride_major(p) >= std::max< std::ptrdiff_t >(1,n) );
        BOOST_ASSERT( stride_major(s) >= std::max< std::ptrdiff_t >(1,n) );
        detail::tgevc( side, howmny, begin_value(select), n, begin_value(s),
                stride_major(s), begin_value(p), stride_major(p),
                begin_value(vl), stride_major(vl), begin_value(vr),
                stride_major(vr), mm, m,
                begin_value(work.select(real_type())), info );
    }

    //
    // Static member function that
    // * Figures out the minimal workspace requirements, and passes
    //   the results to the user-defined workspace overload of the 
    //   invoke static member function
    // * Enables the unblocked algorithm (BLAS level 2)
    //
    template< typename VectorSELECT, typename MatrixS, typename MatrixP,
            typename MatrixVL, typename MatrixVR >
    static void invoke( const char side, const char howmny,
            const VectorSELECT& select, const fortran_int_t n,
            const MatrixS& s, const MatrixP& p, MatrixVL& vl, MatrixVR& vr,
            const fortran_int_t mm, fortran_int_t& m,
            fortran_int_t& info, minimal_workspace work ) {
        bindings::detail::array< real_type > tmp_work( min_size_work( n ) );
        invoke( side, howmny, select, n, s, p, vl, vr, mm, m, info,
                workspace( tmp_work ) );
    }

    //
    // Static member function that
    // * Figures out the optimal workspace requirements, and passes
    //   the results to the user-defined workspace overload of the 
    //   invoke static member
    // * Enables the blocked algorithm (BLAS level 3)
    //
    template< typename VectorSELECT, typename MatrixS, typename MatrixP,
            typename MatrixVL, typename MatrixVR >
    static void invoke( const char side, const char howmny,
            const VectorSELECT& select, const fortran_int_t n,
            const MatrixS& s, const MatrixP& p, MatrixVL& vl, MatrixVR& vr,
            const fortran_int_t mm, fortran_int_t& m,
            fortran_int_t& info, optimal_workspace work ) {
        invoke( side, howmny, select, n, s, p, vl, vr, mm, m, info,
                minimal_workspace() );
    }

    //
    // Static member function that returns the minimum size of
    // workspace-array work.
    //
    static std::ptrdiff_t min_size_work( const std::ptrdiff_t n ) {
        return 6*n;
    }
};

//
// This implementation is enabled if Value is a complex type.
//
template< typename Value >
struct tgevc_impl< Value, typename boost::enable_if< is_complex< Value > >::type > {

    typedef Value value_type;
    typedef typename remove_imaginary< Value >::type real_type;
    typedef tag::column_major order;

    //
    // Static member function for user-defined workspaces, that
    // * Deduces the required arguments for dispatching to LAPACK, and
    // * Asserts that most arguments make sense.
    //
    template< typename VectorSELECT, typename MatrixS, typename MatrixP,
            typename MatrixVL, typename MatrixVR, typename WORK,
            typename RWORK >
    static void invoke( const char side, const char howmny,
            const VectorSELECT& select, const fortran_int_t n,
            const MatrixS& s, const MatrixP& p, MatrixVL& vl, MatrixVR& vr,
            const fortran_int_t mm, fortran_int_t& m,
            fortran_int_t& info, detail::workspace2< WORK, RWORK > work ) {
        BOOST_STATIC_ASSERT( (boost::is_same< typename remove_const<
                typename value< MatrixS >::type >::type,
                typename remove_const< typename value<
                MatrixP >::type >::type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename remove_const<
                typename value< MatrixS >::type >::type,
                typename remove_const< typename value<
                MatrixVL >::type >::type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename remove_const<
                typename value< MatrixS >::type >::type,
                typename remove_const< typename value<
                MatrixVR >::type >::type >::value) );
        BOOST_STATIC_ASSERT( (is_mutable< MatrixVL >::value) );
        BOOST_STATIC_ASSERT( (is_mutable< MatrixVR >::value) );
        BOOST_ASSERT( howmny == 'A' || howmny == 'B' || howmny == 'S' );
        BOOST_ASSERT( mm >= m );
        BOOST_ASSERT( n >= 0 );
        BOOST_ASSERT( side == 'R' || side == 'L' || side == 'B' );
        BOOST_ASSERT( size(work.select(real_type())) >= min_size_rwork( n ));
        BOOST_ASSERT( size(work.select(value_type())) >= min_size_work( n ));
        BOOST_ASSERT( size_minor(p) == 1 || stride_minor(p) == 1 );
        BOOST_ASSERT( size_minor(s) == 1 || stride_minor(s) == 1 );
        BOOST_ASSERT( size_minor(vl) == 1 || stride_minor(vl) == 1 );
        BOOST_ASSERT( size_minor(vr) == 1 || stride_minor(vr) == 1 );
        BOOST_ASSERT( stride_major(p) >= std::max< std::ptrdiff_t >(1,n) );
        BOOST_ASSERT( stride_major(s) >= std::max< std::ptrdiff_t >(1,n) );
        detail::tgevc( side, howmny, begin_value(select), n, begin_value(s),
                stride_major(s), begin_value(p), stride_major(p),
                begin_value(vl), stride_major(vl), begin_value(vr),
                stride_major(vr), mm, m,
                begin_value(work.select(value_type())),
                begin_value(work.select(real_type())), info );
    }

    //
    // Static member function that
    // * Figures out the minimal workspace requirements, and passes
    //   the results to the user-defined workspace overload of the 
    //   invoke static member function
    // * Enables the unblocked algorithm (BLAS level 2)
    //
    template< typename VectorSELECT, typename MatrixS, typename MatrixP,
            typename MatrixVL, typename MatrixVR >
    static void invoke( const char side, const char howmny,
            const VectorSELECT& select, const fortran_int_t n,
            const MatrixS& s, const MatrixP& p, MatrixVL& vl, MatrixVR& vr,
            const fortran_int_t mm, fortran_int_t& m,
            fortran_int_t& info, minimal_workspace work ) {
        bindings::detail::array< value_type > tmp_work( min_size_work( n ) );
        bindings::detail::array< real_type > tmp_rwork( min_size_rwork( n ) );
        invoke( side, howmny, select, n, s, p, vl, vr, mm, m, info,
                workspace( tmp_work, tmp_rwork ) );
    }

    //
    // Static member function that
    // * Figures out the optimal workspace requirements, and passes
    //   the results to the user-defined workspace overload of the 
    //   invoke static member
    // * Enables the blocked algorithm (BLAS level 3)
    //
    template< typename VectorSELECT, typename MatrixS, typename MatrixP,
            typename MatrixVL, typename MatrixVR >
    static void invoke( const char side, const char howmny,
            const VectorSELECT& select, const fortran_int_t n,
            const MatrixS& s, const MatrixP& p, MatrixVL& vl, MatrixVR& vr,
            const fortran_int_t mm, fortran_int_t& m,
            fortran_int_t& info, optimal_workspace work ) {
        invoke( side, howmny, select, n, s, p, vl, vr, mm, m, info,
                minimal_workspace() );
    }

    //
    // Static member function that returns the minimum size of
    // workspace-array work.
    //
    static std::ptrdiff_t min_size_work( const std::ptrdiff_t n ) {
        return 2*n;
    }

    //
    // Static member function that returns the minimum size of
    // workspace-array rwork.
    //
    static std::ptrdiff_t min_size_rwork( const std::ptrdiff_t n ) {
        return 2*n;
    }
};


//
// Functions for direct use. These functions are overloaded for temporaries,
// so that wrapped types can still be passed and used for write-access. In
// addition, if applicable, they are overloaded for user-defined workspaces.
// Calls to these functions are passed to the tgevc_impl classes. In the 
// documentation, most overloads are collapsed to avoid a large number of
// prototypes which are very similar.
//

//
// Overloaded function for tgevc. Its overload differs for
// * MatrixVL&
// * MatrixVR&
// * User-defined workspace
//
template< typename VectorSELECT, typename MatrixS, typename MatrixP,
        typename MatrixVL, typename MatrixVR, typename Workspace >
inline std::ptrdiff_t tgevc( const char side, const char howmny,
        const VectorSELECT& select, const fortran_int_t n,
        const MatrixS& s, const MatrixP& p, MatrixVL& vl, MatrixVR& vr,
        const fortran_int_t mm, fortran_int_t& m, Workspace work ) {
    fortran_int_t info(0);
    tgevc_impl< typename value< MatrixS >::type >::invoke( side, howmny,
            select, n, s, p, vl, vr, mm, m, info, work );
    return info;
}

//
// Overloaded function for tgevc. Its overload differs for
// * MatrixVL&
// * MatrixVR&
// * Default workspace-type (optimal)
//
template< typename VectorSELECT, typename MatrixS, typename MatrixP,
        typename MatrixVL, typename MatrixVR >
inline std::ptrdiff_t tgevc( const char side, const char howmny,
        const VectorSELECT& select, const fortran_int_t n,
        const MatrixS& s, const MatrixP& p, MatrixVL& vl, MatrixVR& vr,
        const fortran_int_t mm, fortran_int_t& m ) {
    fortran_int_t info(0);
    tgevc_impl< typename value< MatrixS >::type >::invoke( side, howmny,
            select, n, s, p, vl, vr, mm, m, info, optimal_workspace() );
    return info;
}

//
// Overloaded function for tgevc. Its overload differs for
// * const MatrixVL&
// * MatrixVR&
// * User-defined workspace
//
template< typename VectorSELECT, typename MatrixS, typename MatrixP,
        typename MatrixVL, typename MatrixVR, typename Workspace >
inline std::ptrdiff_t tgevc( const char side, const char howmny,
        const VectorSELECT& select, const fortran_int_t n,
        const MatrixS& s, const MatrixP& p, const MatrixVL& vl, MatrixVR& vr,
        const fortran_int_t mm, fortran_int_t& m, Workspace work ) {
    fortran_int_t info(0);
    tgevc_impl< typename value< MatrixS >::type >::invoke( side, howmny,
            select, n, s, p, vl, vr, mm, m, info, work );
    return info;
}

//
// Overloaded function for tgevc. Its overload differs for
// * const MatrixVL&
// * MatrixVR&
// * Default workspace-type (optimal)
//
template< typename VectorSELECT, typename MatrixS, typename MatrixP,
        typename MatrixVL, typename MatrixVR >
inline std::ptrdiff_t tgevc( const char side, const char howmny,
        const VectorSELECT& select, const fortran_int_t n,
        const MatrixS& s, const MatrixP& p, const MatrixVL& vl, MatrixVR& vr,
        const fortran_int_t mm, fortran_int_t& m ) {
    fortran_int_t info(0);
    tgevc_impl< typename value< MatrixS >::type >::invoke( side, howmny,
            select, n, s, p, vl, vr, mm, m, info, optimal_workspace() );
    return info;
}

//
// Overloaded function for tgevc. Its overload differs for
// * MatrixVL&
// * const MatrixVR&
// * User-defined workspace
//
template< typename VectorSELECT, typename MatrixS, typename MatrixP,
        typename MatrixVL, typename MatrixVR, typename Workspace >
inline std::ptrdiff_t tgevc( const char side, const char howmny,
        const VectorSELECT& select, const fortran_int_t n,
        const MatrixS& s, const MatrixP& p, MatrixVL& vl, const MatrixVR& vr,
        const fortran_int_t mm, fortran_int_t& m, Workspace work ) {
    fortran_int_t info(0);
    tgevc_impl< typename value< MatrixS >::type >::invoke( side, howmny,
            select, n, s, p, vl, vr, mm, m, info, work );
    return info;
}

//
// Overloaded function for tgevc. Its overload differs for
// * MatrixVL&
// * const MatrixVR&
// * Default workspace-type (optimal)
//
template< typename VectorSELECT, typename MatrixS, typename MatrixP,
        typename MatrixVL, typename MatrixVR >
inline std::ptrdiff_t tgevc( const char side, const char howmny,
        const VectorSELECT& select, const fortran_int_t n,
        const MatrixS& s, const MatrixP& p, MatrixVL& vl, const MatrixVR& vr,
        const fortran_int_t mm, fortran_int_t& m ) {
    fortran_int_t info(0);
    tgevc_impl< typename value< MatrixS >::type >::invoke( side, howmny,
            select, n, s, p, vl, vr, mm, m, info, optimal_workspace() );
    return info;
}

//
// Overloaded function for tgevc. Its overload differs for
// * const MatrixVL&
// * const MatrixVR&
// * User-defined workspace
//
template< typename VectorSELECT, typename MatrixS, typename MatrixP,
        typename MatrixVL, typename MatrixVR, typename Workspace >
inline std::ptrdiff_t tgevc( const char side, const char howmny,
        const VectorSELECT& select, const fortran_int_t n,
        const MatrixS& s, const MatrixP& p, const MatrixVL& vl,
        const MatrixVR& vr, const fortran_int_t mm, fortran_int_t& m,
        Workspace work ) {
    fortran_int_t info(0);
    tgevc_impl< typename value< MatrixS >::type >::invoke( side, howmny,
            select, n, s, p, vl, vr, mm, m, info, work );
    return info;
}

//
// Overloaded function for tgevc. Its overload differs for
// * const MatrixVL&
// * const MatrixVR&
// * Default workspace-type (optimal)
//
template< typename VectorSELECT, typename MatrixS, typename MatrixP,
        typename MatrixVL, typename MatrixVR >
inline std::ptrdiff_t tgevc( const char side, const char howmny,
        const VectorSELECT& select, const fortran_int_t n,
        const MatrixS& s, const MatrixP& p, const MatrixVL& vl,
        const MatrixVR& vr, const fortran_int_t mm,
        fortran_int_t& m ) {
    fortran_int_t info(0);
    tgevc_impl< typename value< MatrixS >::type >::invoke( side, howmny,
            select, n, s, p, vl, vr, mm, m, info, optimal_workspace() );
    return info;
}

} // namespace lapack
} // namespace bindings
} // namespace numeric
} // namespace boost

#endif
