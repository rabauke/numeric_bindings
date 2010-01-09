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

#ifndef BOOST_NUMERIC_BINDINGS_LAPACK_COMPUTATIONAL_TRSNA_HPP
#define BOOST_NUMERIC_BINDINGS_LAPACK_COMPUTATIONAL_TRSNA_HPP

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
inline void trsna( char job, char howmny, const logical_t* select,
        fortran_int_t n, const float* t, fortran_int_t ldt, const float* vl,
        fortran_int_t ldvl, const float* vr, fortran_int_t ldvr, float* s,
        float* sep, fortran_int_t mm, fortran_int_t& m, float* work,
        fortran_int_t ldwork, fortran_int_t* iwork, fortran_int_t& info ) {
    LAPACK_STRSNA( &job, &howmny, select, &n, t, &ldt, vl, &ldvl, vr, &ldvr,
            s, sep, &mm, &m, work, &ldwork, iwork, &info );
}

//
// Overloaded function for dispatching to double value-type.
//
inline void trsna( char job, char howmny, const logical_t* select,
        fortran_int_t n, const double* t, fortran_int_t ldt, const double* vl,
        fortran_int_t ldvl, const double* vr, fortran_int_t ldvr, double* s,
        double* sep, fortran_int_t mm, fortran_int_t& m, double* work,
        fortran_int_t ldwork, fortran_int_t* iwork, fortran_int_t& info ) {
    LAPACK_DTRSNA( &job, &howmny, select, &n, t, &ldt, vl, &ldvl, vr, &ldvr,
            s, sep, &mm, &m, work, &ldwork, iwork, &info );
}

//
// Overloaded function for dispatching to complex<float> value-type.
//
inline void trsna( char job, char howmny, const logical_t* select,
        fortran_int_t n, const std::complex<float>* t, fortran_int_t ldt,
        const std::complex<float>* vl, fortran_int_t ldvl,
        const std::complex<float>* vr, fortran_int_t ldvr, float* s,
        float* sep, fortran_int_t mm, fortran_int_t& m,
        std::complex<float>* work, fortran_int_t ldwork, float* rwork,
        fortran_int_t& info ) {
    LAPACK_CTRSNA( &job, &howmny, select, &n, t, &ldt, vl, &ldvl, vr, &ldvr,
            s, sep, &mm, &m, work, &ldwork, rwork, &info );
}

//
// Overloaded function for dispatching to complex<double> value-type.
//
inline void trsna( char job, char howmny, const logical_t* select,
        fortran_int_t n, const std::complex<double>* t, fortran_int_t ldt,
        const std::complex<double>* vl, fortran_int_t ldvl,
        const std::complex<double>* vr, fortran_int_t ldvr, double* s,
        double* sep, fortran_int_t mm, fortran_int_t& m,
        std::complex<double>* work, fortran_int_t ldwork, double* rwork,
        fortran_int_t& info ) {
    LAPACK_ZTRSNA( &job, &howmny, select, &n, t, &ldt, vl, &ldvl, vr, &ldvr,
            s, sep, &mm, &m, work, &ldwork, rwork, &info );
}

} // namespace detail

//
// Value-type based template class. Use this class if you need a type
// for dispatching to trsna.
//
template< typename Value, typename Enable = void >
struct trsna_impl {};

//
// This implementation is enabled if Value is a real type.
//
template< typename Value >
struct trsna_impl< Value, typename boost::enable_if< is_real< Value > >::type > {

    typedef Value value_type;
    typedef typename remove_imaginary< Value >::type real_type;
    typedef tag::column_major order;

    //
    // Static member function for user-defined workspaces, that
    // * Deduces the required arguments for dispatching to LAPACK, and
    // * Asserts that most arguments make sense.
    //
    template< typename VectorSELECT, typename MatrixT, typename MatrixVL,
            typename MatrixVR, typename VectorS, typename VectorSEP,
            typename WORK, typename IWORK >
    static void invoke( const char job, const char howmny,
            const VectorSELECT& select, const MatrixT& t, const MatrixVL& vl,
            const MatrixVR& vr, VectorS& s, VectorSEP& sep,
            const fortran_int_t mm, fortran_int_t& m,
            fortran_int_t& info, detail::workspace2< WORK, IWORK > work ) {
        BOOST_STATIC_ASSERT( (boost::is_same< typename remove_const<
                typename value< MatrixT >::type >::type,
                typename remove_const< typename value<
                MatrixVL >::type >::type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename remove_const<
                typename value< MatrixT >::type >::type,
                typename remove_const< typename value<
                MatrixVR >::type >::type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename remove_const<
                typename value< MatrixT >::type >::type,
                typename remove_const< typename value<
                VectorS >::type >::type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename remove_const<
                typename value< MatrixT >::type >::type,
                typename remove_const< typename value<
                VectorSEP >::type >::type >::value) );
        BOOST_STATIC_ASSERT( (is_mutable< VectorS >::value) );
        BOOST_STATIC_ASSERT( (is_mutable< VectorSEP >::value) );
        BOOST_ASSERT( howmny == 'A' || howmny == 'S' );
        BOOST_ASSERT( job == 'E' || job == 'V' || job == 'B' );
        BOOST_ASSERT( mm >= m );
        BOOST_ASSERT( size(work.select(fortran_int_t())) >=
                min_size_iwork( $CALL_MIN_SIZE ));
        BOOST_ASSERT( size(work.select(real_type())) >= min_size_work(
                $CALL_MIN_SIZE ));
        BOOST_ASSERT( size_column(t) >= 0 );
        BOOST_ASSERT( size_minor(t) == 1 || stride_minor(t) == 1 );
        BOOST_ASSERT( size_minor(vl) == 1 || stride_minor(vl) == 1 );
        BOOST_ASSERT( size_minor(vr) == 1 || stride_minor(vr) == 1 );
        BOOST_ASSERT( stride_major(t) >= std::max< std::ptrdiff_t >(1,
                size_column(t)) );
        detail::trsna( job, howmny, begin_value(select), size_column(t),
                begin_value(t), stride_major(t), begin_value(vl),
                stride_major(vl), begin_value(vr), stride_major(vr),
                begin_value(s), begin_value(sep), mm, m, begin_value(work),
                stride_major(work),
                begin_value(work.select(fortran_int_t())), info );
    }

    //
    // Static member function that
    // * Figures out the minimal workspace requirements, and passes
    //   the results to the user-defined workspace overload of the 
    //   invoke static member function
    // * Enables the unblocked algorithm (BLAS level 2)
    //
    template< typename VectorSELECT, typename MatrixT, typename MatrixVL,
            typename MatrixVR, typename VectorS, typename VectorSEP >
    static void invoke( const char job, const char howmny,
            const VectorSELECT& select, const MatrixT& t, const MatrixVL& vl,
            const MatrixVR& vr, VectorS& s, VectorSEP& sep,
            const fortran_int_t mm, fortran_int_t& m,
            fortran_int_t& info, minimal_workspace work ) {
        bindings::detail::array< real_type > tmp_work( min_size_work(
                $CALL_MIN_SIZE ) );
        bindings::detail::array< fortran_int_t > tmp_iwork(
                min_size_iwork( $CALL_MIN_SIZE ) );
        invoke( job, howmny, select, t, vl, vr, s, sep, mm, m, info,
                workspace( tmp_work, tmp_iwork ) );
    }

    //
    // Static member function that
    // * Figures out the optimal workspace requirements, and passes
    //   the results to the user-defined workspace overload of the 
    //   invoke static member
    // * Enables the blocked algorithm (BLAS level 3)
    //
    template< typename VectorSELECT, typename MatrixT, typename MatrixVL,
            typename MatrixVR, typename VectorS, typename VectorSEP >
    static void invoke( const char job, const char howmny,
            const VectorSELECT& select, const MatrixT& t, const MatrixVL& vl,
            const MatrixVR& vr, VectorS& s, VectorSEP& sep,
            const fortran_int_t mm, fortran_int_t& m,
            fortran_int_t& info, optimal_workspace work ) {
        invoke( job, howmny, select, t, vl, vr, s, sep, mm, m, info,
                minimal_workspace() );
    }

    //
    // Static member function that returns the minimum size of
    // workspace-array work.
    //
    static std::ptrdiff_t min_size_work( $ARGUMENTS ) {
        $MIN_SIZE
    }

    //
    // Static member function that returns the minimum size of
    // workspace-array iwork.
    //
    static std::ptrdiff_t min_size_iwork( $ARGUMENTS ) {
        $MIN_SIZE
    }
};

//
// This implementation is enabled if Value is a complex type.
//
template< typename Value >
struct trsna_impl< Value, typename boost::enable_if< is_complex< Value > >::type > {

    typedef Value value_type;
    typedef typename remove_imaginary< Value >::type real_type;
    typedef tag::column_major order;

    //
    // Static member function for user-defined workspaces, that
    // * Deduces the required arguments for dispatching to LAPACK, and
    // * Asserts that most arguments make sense.
    //
    template< typename VectorSELECT, typename MatrixT, typename MatrixVL,
            typename MatrixVR, typename VectorS, typename VectorSEP,
            typename WORK, typename RWORK >
    static void invoke( const char job, const char howmny,
            const VectorSELECT& select, const MatrixT& t, const MatrixVL& vl,
            const MatrixVR& vr, VectorS& s, VectorSEP& sep,
            const fortran_int_t mm, fortran_int_t& m,
            fortran_int_t& info, detail::workspace2< WORK, RWORK > work ) {
        BOOST_STATIC_ASSERT( (boost::is_same< typename remove_const<
                typename value< VectorS >::type >::type,
                typename remove_const< typename value<
                VectorSEP >::type >::type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename remove_const<
                typename value< MatrixT >::type >::type,
                typename remove_const< typename value<
                MatrixVL >::type >::type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename remove_const<
                typename value< MatrixT >::type >::type,
                typename remove_const< typename value<
                MatrixVR >::type >::type >::value) );
        BOOST_STATIC_ASSERT( (is_mutable< VectorS >::value) );
        BOOST_STATIC_ASSERT( (is_mutable< VectorSEP >::value) );
        BOOST_ASSERT( howmny == 'A' || howmny == 'S' );
        BOOST_ASSERT( job == 'E' || job == 'V' || job == 'B' );
        BOOST_ASSERT( mm >= m );
        BOOST_ASSERT( size(work.select(real_type())) >= min_size_rwork(
                $CALL_MIN_SIZE ));
        BOOST_ASSERT( size(work.select(value_type())) >= min_size_work(
                $CALL_MIN_SIZE ));
        BOOST_ASSERT( size_column(t) >= 0 );
        BOOST_ASSERT( size_minor(t) == 1 || stride_minor(t) == 1 );
        BOOST_ASSERT( size_minor(vl) == 1 || stride_minor(vl) == 1 );
        BOOST_ASSERT( size_minor(vr) == 1 || stride_minor(vr) == 1 );
        BOOST_ASSERT( stride_major(t) >= std::max< std::ptrdiff_t >(1,
                size_column(t)) );
        detail::trsna( job, howmny, begin_value(select), size_column(t),
                begin_value(t), stride_major(t), begin_value(vl),
                stride_major(vl), begin_value(vr), stride_major(vr),
                begin_value(s), begin_value(sep), mm, m, begin_value(work),
                stride_major(work), begin_value(work.select(real_type())),
                info );
    }

    //
    // Static member function that
    // * Figures out the minimal workspace requirements, and passes
    //   the results to the user-defined workspace overload of the 
    //   invoke static member function
    // * Enables the unblocked algorithm (BLAS level 2)
    //
    template< typename VectorSELECT, typename MatrixT, typename MatrixVL,
            typename MatrixVR, typename VectorS, typename VectorSEP >
    static void invoke( const char job, const char howmny,
            const VectorSELECT& select, const MatrixT& t, const MatrixVL& vl,
            const MatrixVR& vr, VectorS& s, VectorSEP& sep,
            const fortran_int_t mm, fortran_int_t& m,
            fortran_int_t& info, minimal_workspace work ) {
        bindings::detail::array< value_type > tmp_work( min_size_work(
                $CALL_MIN_SIZE ) );
        bindings::detail::array< real_type > tmp_rwork( min_size_rwork(
                $CALL_MIN_SIZE ) );
        invoke( job, howmny, select, t, vl, vr, s, sep, mm, m, info,
                workspace( tmp_work, tmp_rwork ) );
    }

    //
    // Static member function that
    // * Figures out the optimal workspace requirements, and passes
    //   the results to the user-defined workspace overload of the 
    //   invoke static member
    // * Enables the blocked algorithm (BLAS level 3)
    //
    template< typename VectorSELECT, typename MatrixT, typename MatrixVL,
            typename MatrixVR, typename VectorS, typename VectorSEP >
    static void invoke( const char job, const char howmny,
            const VectorSELECT& select, const MatrixT& t, const MatrixVL& vl,
            const MatrixVR& vr, VectorS& s, VectorSEP& sep,
            const fortran_int_t mm, fortran_int_t& m,
            fortran_int_t& info, optimal_workspace work ) {
        invoke( job, howmny, select, t, vl, vr, s, sep, mm, m, info,
                minimal_workspace() );
    }

    //
    // Static member function that returns the minimum size of
    // workspace-array work.
    //
    static std::ptrdiff_t min_size_work( $ARGUMENTS ) {
        $MIN_SIZE
    }

    //
    // Static member function that returns the minimum size of
    // workspace-array rwork.
    //
    static std::ptrdiff_t min_size_rwork( $ARGUMENTS ) {
        $MIN_SIZE
    }
};


//
// Functions for direct use. These functions are overloaded for temporaries,
// so that wrapped types can still be passed and used for write-access. In
// addition, if applicable, they are overloaded for user-defined workspaces.
// Calls to these functions are passed to the trsna_impl classes. In the 
// documentation, most overloads are collapsed to avoid a large number of
// prototypes which are very similar.
//

//
// Overloaded function for trsna. Its overload differs for
// * VectorS&
// * VectorSEP&
// * User-defined workspace
//
template< typename VectorSELECT, typename MatrixT, typename MatrixVL,
        typename MatrixVR, typename VectorS, typename VectorSEP,
        typename Workspace >
inline std::ptrdiff_t trsna( const char job, const char howmny,
        const VectorSELECT& select, const MatrixT& t, const MatrixVL& vl,
        const MatrixVR& vr, VectorS& s, VectorSEP& sep,
        const fortran_int_t mm, fortran_int_t& m, Workspace work ) {
    fortran_int_t info(0);
    trsna_impl< typename value< MatrixT >::type >::invoke( job, howmny,
            select, t, vl, vr, s, sep, mm, m, info, work );
    return info;
}

//
// Overloaded function for trsna. Its overload differs for
// * VectorS&
// * VectorSEP&
// * Default workspace-type (optimal)
//
template< typename VectorSELECT, typename MatrixT, typename MatrixVL,
        typename MatrixVR, typename VectorS, typename VectorSEP >
inline std::ptrdiff_t trsna( const char job, const char howmny,
        const VectorSELECT& select, const MatrixT& t, const MatrixVL& vl,
        const MatrixVR& vr, VectorS& s, VectorSEP& sep,
        const fortran_int_t mm, fortran_int_t& m ) {
    fortran_int_t info(0);
    trsna_impl< typename value< MatrixT >::type >::invoke( job, howmny,
            select, t, vl, vr, s, sep, mm, m, info, optimal_workspace() );
    return info;
}

//
// Overloaded function for trsna. Its overload differs for
// * const VectorS&
// * VectorSEP&
// * User-defined workspace
//
template< typename VectorSELECT, typename MatrixT, typename MatrixVL,
        typename MatrixVR, typename VectorS, typename VectorSEP,
        typename Workspace >
inline std::ptrdiff_t trsna( const char job, const char howmny,
        const VectorSELECT& select, const MatrixT& t, const MatrixVL& vl,
        const MatrixVR& vr, const VectorS& s, VectorSEP& sep,
        const fortran_int_t mm, fortran_int_t& m, Workspace work ) {
    fortran_int_t info(0);
    trsna_impl< typename value< MatrixT >::type >::invoke( job, howmny,
            select, t, vl, vr, s, sep, mm, m, info, work );
    return info;
}

//
// Overloaded function for trsna. Its overload differs for
// * const VectorS&
// * VectorSEP&
// * Default workspace-type (optimal)
//
template< typename VectorSELECT, typename MatrixT, typename MatrixVL,
        typename MatrixVR, typename VectorS, typename VectorSEP >
inline std::ptrdiff_t trsna( const char job, const char howmny,
        const VectorSELECT& select, const MatrixT& t, const MatrixVL& vl,
        const MatrixVR& vr, const VectorS& s, VectorSEP& sep,
        const fortran_int_t mm, fortran_int_t& m ) {
    fortran_int_t info(0);
    trsna_impl< typename value< MatrixT >::type >::invoke( job, howmny,
            select, t, vl, vr, s, sep, mm, m, info, optimal_workspace() );
    return info;
}

//
// Overloaded function for trsna. Its overload differs for
// * VectorS&
// * const VectorSEP&
// * User-defined workspace
//
template< typename VectorSELECT, typename MatrixT, typename MatrixVL,
        typename MatrixVR, typename VectorS, typename VectorSEP,
        typename Workspace >
inline std::ptrdiff_t trsna( const char job, const char howmny,
        const VectorSELECT& select, const MatrixT& t, const MatrixVL& vl,
        const MatrixVR& vr, VectorS& s, const VectorSEP& sep,
        const fortran_int_t mm, fortran_int_t& m, Workspace work ) {
    fortran_int_t info(0);
    trsna_impl< typename value< MatrixT >::type >::invoke( job, howmny,
            select, t, vl, vr, s, sep, mm, m, info, work );
    return info;
}

//
// Overloaded function for trsna. Its overload differs for
// * VectorS&
// * const VectorSEP&
// * Default workspace-type (optimal)
//
template< typename VectorSELECT, typename MatrixT, typename MatrixVL,
        typename MatrixVR, typename VectorS, typename VectorSEP >
inline std::ptrdiff_t trsna( const char job, const char howmny,
        const VectorSELECT& select, const MatrixT& t, const MatrixVL& vl,
        const MatrixVR& vr, VectorS& s, const VectorSEP& sep,
        const fortran_int_t mm, fortran_int_t& m ) {
    fortran_int_t info(0);
    trsna_impl< typename value< MatrixT >::type >::invoke( job, howmny,
            select, t, vl, vr, s, sep, mm, m, info, optimal_workspace() );
    return info;
}

//
// Overloaded function for trsna. Its overload differs for
// * const VectorS&
// * const VectorSEP&
// * User-defined workspace
//
template< typename VectorSELECT, typename MatrixT, typename MatrixVL,
        typename MatrixVR, typename VectorS, typename VectorSEP,
        typename Workspace >
inline std::ptrdiff_t trsna( const char job, const char howmny,
        const VectorSELECT& select, const MatrixT& t, const MatrixVL& vl,
        const MatrixVR& vr, const VectorS& s, const VectorSEP& sep,
        const fortran_int_t mm, fortran_int_t& m, Workspace work ) {
    fortran_int_t info(0);
    trsna_impl< typename value< MatrixT >::type >::invoke( job, howmny,
            select, t, vl, vr, s, sep, mm, m, info, work );
    return info;
}

//
// Overloaded function for trsna. Its overload differs for
// * const VectorS&
// * const VectorSEP&
// * Default workspace-type (optimal)
//
template< typename VectorSELECT, typename MatrixT, typename MatrixVL,
        typename MatrixVR, typename VectorS, typename VectorSEP >
inline std::ptrdiff_t trsna( const char job, const char howmny,
        const VectorSELECT& select, const MatrixT& t, const MatrixVL& vl,
        const MatrixVR& vr, const VectorS& s, const VectorSEP& sep,
        const fortran_int_t mm, fortran_int_t& m ) {
    fortran_int_t info(0);
    trsna_impl< typename value< MatrixT >::type >::invoke( job, howmny,
            select, t, vl, vr, s, sep, mm, m, info, optimal_workspace() );
    return info;
}

} // namespace lapack
} // namespace bindings
} // namespace numeric
} // namespace boost

#endif
