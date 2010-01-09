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

#ifndef BOOST_NUMERIC_BINDINGS_LAPACK_COMPUTATIONAL_TREXC_HPP
#define BOOST_NUMERIC_BINDINGS_LAPACK_COMPUTATIONAL_TREXC_HPP

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
inline void trexc( char compq, fortran_int_t n, float* t, fortran_int_t ldt,
        float* q, fortran_int_t ldq, fortran_int_t& ifst, fortran_int_t& ilst,
        float* work, fortran_int_t& info ) {
    LAPACK_STREXC( &compq, &n, t, &ldt, q, &ldq, &ifst, &ilst, work, &info );
}

//
// Overloaded function for dispatching to double value-type.
//
inline void trexc( char compq, fortran_int_t n, double* t, fortran_int_t ldt,
        double* q, fortran_int_t ldq, fortran_int_t& ifst,
        fortran_int_t& ilst, double* work, fortran_int_t& info ) {
    LAPACK_DTREXC( &compq, &n, t, &ldt, q, &ldq, &ifst, &ilst, work, &info );
}

//
// Overloaded function for dispatching to complex<float> value-type.
//
inline void trexc( char compq, fortran_int_t n, std::complex<float>* t,
        fortran_int_t ldt, std::complex<float>* q, fortran_int_t ldq,
        fortran_int_t ifst, fortran_int_t ilst, fortran_int_t& info ) {
    LAPACK_CTREXC( &compq, &n, t, &ldt, q, &ldq, &ifst, &ilst, &info );
}

//
// Overloaded function for dispatching to complex<double> value-type.
//
inline void trexc( char compq, fortran_int_t n, std::complex<double>* t,
        fortran_int_t ldt, std::complex<double>* q, fortran_int_t ldq,
        fortran_int_t ifst, fortran_int_t ilst, fortran_int_t& info ) {
    LAPACK_ZTREXC( &compq, &n, t, &ldt, q, &ldq, &ifst, &ilst, &info );
}

} // namespace detail

//
// Value-type based template class. Use this class if you need a type
// for dispatching to trexc.
//
template< typename Value, typename Enable = void >
struct trexc_impl {};

//
// This implementation is enabled if Value is a real type.
//
template< typename Value >
struct trexc_impl< Value, typename boost::enable_if< is_real< Value > >::type > {

    typedef Value value_type;
    typedef typename remove_imaginary< Value >::type real_type;
    typedef tag::column_major order;

    //
    // Static member function, that
    // * Deduces the required arguments for dispatching to LAPACK, and
    // * Asserts that most arguments make sense.
    //
    template< typename MatrixT, typename MatrixQ >
    static void invoke( const char compq, MatrixT& t, MatrixQ& q,
            fortran_int_t& ifst, fortran_int_t& ilst,
            fortran_int_t& info ) {
        BOOST_STATIC_ASSERT( (boost::is_same< typename remove_const<
                typename value< MatrixT >::type >::type,
                typename remove_const< typename value<
                MatrixQ >::type >::type >::value) );
        BOOST_STATIC_ASSERT( (is_mutable< MatrixT >::value) );
        BOOST_STATIC_ASSERT( (is_mutable< MatrixQ >::value) );
        BOOST_ASSERT( compq == 'V' || compq == 'N' );
        BOOST_ASSERT( size(work.select(real_type())) >= min_size_work(
                size_column(t) ));
        BOOST_ASSERT( size_column(t) >= 0 );
        BOOST_ASSERT( size_minor(q) == 1 || stride_minor(q) == 1 );
        BOOST_ASSERT( size_minor(t) == 1 || stride_minor(t) == 1 );
        BOOST_ASSERT( stride_major(q) >= std::max< std::ptrdiff_t >(1,
                size_column(t)) );
        BOOST_ASSERT( stride_major(t) >= std::max< std::ptrdiff_t >(1,
                size_column(t)) );
        detail::trexc( compq, size_column(t), begin_value(t), stride_major(t),
                begin_value(q), stride_major(q), ifst, ilst,
                begin_value(work.select(real_type())), info );
    }

};

//
// This implementation is enabled if Value is a complex type.
//
template< typename Value >
struct trexc_impl< Value, typename boost::enable_if< is_complex< Value > >::type > {

    typedef Value value_type;
    typedef typename remove_imaginary< Value >::type real_type;
    typedef tag::column_major order;

    //
    // Static member function for user-defined workspaces, that
    // * Deduces the required arguments for dispatching to LAPACK, and
    // * Asserts that most arguments make sense.
    //
    template< typename MatrixT, typename MatrixQ, $WORKSPACE_TYPENAMES >
    static void invoke( const char compq, MatrixT& t, MatrixQ& q,
            const fortran_int_t ifst, const fortran_int_t ilst,
            fortran_int_t& info, detail::workspace$WORKSPACE_SIZE<
            $WORKSPACE_TYPES > work ) {
        BOOST_STATIC_ASSERT( (boost::is_same< typename remove_const<
                typename value< MatrixT >::type >::type,
                typename remove_const< typename value<
                MatrixQ >::type >::type >::value) );
        BOOST_STATIC_ASSERT( (is_mutable< MatrixT >::value) );
        BOOST_STATIC_ASSERT( (is_mutable< MatrixQ >::value) );
        BOOST_ASSERT( compq == 'V' || compq == 'N' );
        BOOST_ASSERT( size_column(t) >= 0 );
        BOOST_ASSERT( size_minor(q) == 1 || stride_minor(q) == 1 );
        BOOST_ASSERT( size_minor(t) == 1 || stride_minor(t) == 1 );
        BOOST_ASSERT( stride_major(q) >= std::max< std::ptrdiff_t >(1,
                size_column(t)) );
        BOOST_ASSERT( stride_major(t) >= std::max< std::ptrdiff_t >(1,
                size_column(t)) );
        detail::trexc( compq, size_column(t), begin_value(t), stride_major(t),
                begin_value(q), stride_major(q), ifst, ilst, info );
    }

    //
    // Static member function that
    // * Figures out the minimal workspace requirements, and passes
    //   the results to the user-defined workspace overload of the 
    //   invoke static member function
    // * Enables the unblocked algorithm (BLAS level 2)
    //
    template< typename MatrixT, typename MatrixQ >
    static void invoke( const char compq, MatrixT& t, MatrixQ& q,
            const fortran_int_t ifst, const fortran_int_t ilst,
            fortran_int_t& info, minimal_workspace work ) {
$SETUP_MIN_WORKARRAYS_POST
        invoke( compq, t, q, ifst, ilst, info, workspace( $TMP_WORKARRAYS ) );
    }

    //
    // Static member function that
    // * Figures out the optimal workspace requirements, and passes
    //   the results to the user-defined workspace overload of the 
    //   invoke static member
    // * Enables the blocked algorithm (BLAS level 3)
    //
    template< typename MatrixT, typename MatrixQ >
    static void invoke( const char compq, MatrixT& t, MatrixQ& q,
            const fortran_int_t ifst, const fortran_int_t ilst,
            fortran_int_t& info, optimal_workspace work ) {
$OPT_WORKSPACE_FUNC
    }

$MIN_SIZE_FUNCS
};


//
// Functions for direct use. These functions are overloaded for temporaries,
// so that wrapped types can still be passed and used for write-access. In
// addition, if applicable, they are overloaded for user-defined workspaces.
// Calls to these functions are passed to the trexc_impl classes. In the 
// documentation, most overloads are collapsed to avoid a large number of
// prototypes which are very similar.
//

//
// Overloaded function for trexc. Its overload differs for
// * MatrixT&
// * MatrixQ&
//
template< typename MatrixT, typename MatrixQ >
inline std::ptrdiff_t trexc( const char compq, MatrixT& t, MatrixQ& q,
        fortran_int_t& ifst, fortran_int_t& ilst ) {
    fortran_int_t info(0);
    trexc_impl< typename value< MatrixT >::type >::invoke( compq, t, q,
            ifst, ilst, info );
    return info;
}

//
// Overloaded function for trexc. Its overload differs for
// * const MatrixT&
// * MatrixQ&
//
template< typename MatrixT, typename MatrixQ >
inline std::ptrdiff_t trexc( const char compq, const MatrixT& t,
        MatrixQ& q, fortran_int_t& ifst, fortran_int_t& ilst ) {
    fortran_int_t info(0);
    trexc_impl< typename value< MatrixT >::type >::invoke( compq, t, q,
            ifst, ilst, info );
    return info;
}

//
// Overloaded function for trexc. Its overload differs for
// * MatrixT&
// * const MatrixQ&
//
template< typename MatrixT, typename MatrixQ >
inline std::ptrdiff_t trexc( const char compq, MatrixT& t,
        const MatrixQ& q, fortran_int_t& ifst, fortran_int_t& ilst ) {
    fortran_int_t info(0);
    trexc_impl< typename value< MatrixT >::type >::invoke( compq, t, q,
            ifst, ilst, info );
    return info;
}

//
// Overloaded function for trexc. Its overload differs for
// * const MatrixT&
// * const MatrixQ&
//
template< typename MatrixT, typename MatrixQ >
inline std::ptrdiff_t trexc( const char compq, const MatrixT& t,
        const MatrixQ& q, fortran_int_t& ifst, fortran_int_t& ilst ) {
    fortran_int_t info(0);
    trexc_impl< typename value< MatrixT >::type >::invoke( compq, t, q,
            ifst, ilst, info );
    return info;
}
//
// Overloaded function for trexc. Its overload differs for
// * MatrixT&
// * MatrixQ&
// * User-defined workspace
//
template< typename MatrixT, typename MatrixQ, typename Workspace >
inline std::ptrdiff_t trexc( const char compq, MatrixT& t, MatrixQ& q,
        const fortran_int_t ifst, const fortran_int_t ilst,
        Workspace work ) {
    fortran_int_t info(0);
    trexc_impl< typename value< MatrixT >::type >::invoke( compq, t, q,
            ifst, ilst, info, work );
    return info;
}

//
// Overloaded function for trexc. Its overload differs for
// * MatrixT&
// * MatrixQ&
// * Default workspace-type (optimal)
//
template< typename MatrixT, typename MatrixQ >
inline std::ptrdiff_t trexc( const char compq, MatrixT& t, MatrixQ& q,
        const fortran_int_t ifst, const fortran_int_t ilst ) {
    fortran_int_t info(0);
    trexc_impl< typename value< MatrixT >::type >::invoke( compq, t, q,
            ifst, ilst, info, optimal_workspace() );
    return info;
}

//
// Overloaded function for trexc. Its overload differs for
// * const MatrixT&
// * MatrixQ&
// * User-defined workspace
//
template< typename MatrixT, typename MatrixQ, typename Workspace >
inline std::ptrdiff_t trexc( const char compq, const MatrixT& t,
        MatrixQ& q, const fortran_int_t ifst,
        const fortran_int_t ilst, Workspace work ) {
    fortran_int_t info(0);
    trexc_impl< typename value< MatrixT >::type >::invoke( compq, t, q,
            ifst, ilst, info, work );
    return info;
}

//
// Overloaded function for trexc. Its overload differs for
// * const MatrixT&
// * MatrixQ&
// * Default workspace-type (optimal)
//
template< typename MatrixT, typename MatrixQ >
inline std::ptrdiff_t trexc( const char compq, const MatrixT& t,
        MatrixQ& q, const fortran_int_t ifst,
        const fortran_int_t ilst ) {
    fortran_int_t info(0);
    trexc_impl< typename value< MatrixT >::type >::invoke( compq, t, q,
            ifst, ilst, info, optimal_workspace() );
    return info;
}

//
// Overloaded function for trexc. Its overload differs for
// * MatrixT&
// * const MatrixQ&
// * User-defined workspace
//
template< typename MatrixT, typename MatrixQ, typename Workspace >
inline std::ptrdiff_t trexc( const char compq, MatrixT& t,
        const MatrixQ& q, const fortran_int_t ifst,
        const fortran_int_t ilst, Workspace work ) {
    fortran_int_t info(0);
    trexc_impl< typename value< MatrixT >::type >::invoke( compq, t, q,
            ifst, ilst, info, work );
    return info;
}

//
// Overloaded function for trexc. Its overload differs for
// * MatrixT&
// * const MatrixQ&
// * Default workspace-type (optimal)
//
template< typename MatrixT, typename MatrixQ >
inline std::ptrdiff_t trexc( const char compq, MatrixT& t,
        const MatrixQ& q, const fortran_int_t ifst,
        const fortran_int_t ilst ) {
    fortran_int_t info(0);
    trexc_impl< typename value< MatrixT >::type >::invoke( compq, t, q,
            ifst, ilst, info, optimal_workspace() );
    return info;
}

//
// Overloaded function for trexc. Its overload differs for
// * const MatrixT&
// * const MatrixQ&
// * User-defined workspace
//
template< typename MatrixT, typename MatrixQ, typename Workspace >
inline std::ptrdiff_t trexc( const char compq, const MatrixT& t,
        const MatrixQ& q, const fortran_int_t ifst,
        const fortran_int_t ilst, Workspace work ) {
    fortran_int_t info(0);
    trexc_impl< typename value< MatrixT >::type >::invoke( compq, t, q,
            ifst, ilst, info, work );
    return info;
}

//
// Overloaded function for trexc. Its overload differs for
// * const MatrixT&
// * const MatrixQ&
// * Default workspace-type (optimal)
//
template< typename MatrixT, typename MatrixQ >
inline std::ptrdiff_t trexc( const char compq, const MatrixT& t,
        const MatrixQ& q, const fortran_int_t ifst,
        const fortran_int_t ilst ) {
    fortran_int_t info(0);
    trexc_impl< typename value< MatrixT >::type >::invoke( compq, t, q,
            ifst, ilst, info, optimal_workspace() );
    return info;
}

} // namespace lapack
} // namespace bindings
} // namespace numeric
} // namespace boost

#endif
