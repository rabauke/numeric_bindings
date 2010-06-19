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

#ifndef BOOST_NUMERIC_BINDINGS_LAPACK_COMPUTATIONAL_HBTRD_HPP
#define BOOST_NUMERIC_BINDINGS_LAPACK_COMPUTATIONAL_HBTRD_HPP

#include <boost/assert.hpp>
#include <boost/numeric/bindings/bandwidth.hpp>
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
// The LAPACK-backend for hbtrd is the netlib-compatible backend.
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
inline std::ptrdiff_t hbtrd( const char vect, const UpLo uplo,
        const fortran_int_t n, const fortran_int_t kd,
        std::complex<float>* ab, const fortran_int_t ldab, float* d, float* e,
        std::complex<float>* q, const fortran_int_t ldq,
        std::complex<float>* work ) {
    fortran_int_t info(0);
    LAPACK_CHBTRD( &vect, &lapack_option< UpLo >::value, &n, &kd, ab, &ldab,
            d, e, q, &ldq, work, &info );
    return info;
}

//
// Overloaded function for dispatching to
// * netlib-compatible LAPACK backend (the default), and
// * complex<double> value-type.
//
template< typename UpLo >
inline std::ptrdiff_t hbtrd( const char vect, const UpLo uplo,
        const fortran_int_t n, const fortran_int_t kd,
        std::complex<double>* ab, const fortran_int_t ldab, double* d,
        double* e, std::complex<double>* q, const fortran_int_t ldq,
        std::complex<double>* work ) {
    fortran_int_t info(0);
    LAPACK_ZHBTRD( &vect, &lapack_option< UpLo >::value, &n, &kd, ab, &ldab,
            d, e, q, &ldq, work, &info );
    return info;
}

} // namespace detail

//
// Value-type based template class. Use this class if you need a type
// for dispatching to hbtrd.
//
template< typename Value >
struct hbtrd_impl {

    typedef Value value_type;
    typedef typename remove_imaginary< Value >::type real_type;

    //
    // Static member function for user-defined workspaces, that
    // * Deduces the required arguments for dispatching to LAPACK, and
    // * Asserts that most arguments make sense.
    //
    template< typename MatrixAB, typename VectorD, typename VectorE,
            typename MatrixQ, typename WORK >
    static std::ptrdiff_t invoke( const char vect, MatrixAB& ab, VectorD& d,
            VectorE& e, MatrixQ& q, detail::workspace1< WORK > work ) {
        namespace bindings = ::boost::numeric::bindings;
        typedef typename result_of::uplo_tag< MatrixAB >::type uplo;
        BOOST_STATIC_ASSERT( (bindings::is_column_major< MatrixAB >::value) );
        BOOST_STATIC_ASSERT( (bindings::is_column_major< MatrixQ >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename remove_const<
                typename bindings::value_type< VectorD >::type >::type,
                typename remove_const< typename bindings::value_type<
                VectorE >::type >::type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename remove_const<
                typename bindings::value_type< MatrixAB >::type >::type,
                typename remove_const< typename bindings::value_type<
                MatrixQ >::type >::type >::value) );
        BOOST_STATIC_ASSERT( (bindings::is_mutable< MatrixAB >::value) );
        BOOST_STATIC_ASSERT( (bindings::is_mutable< VectorD >::value) );
        BOOST_STATIC_ASSERT( (bindings::is_mutable< VectorE >::value) );
        BOOST_STATIC_ASSERT( (bindings::is_mutable< MatrixQ >::value) );
        BOOST_ASSERT( bindings::bandwidth(ab, uplo()) >= 0 );
        BOOST_ASSERT( bindings::size(d) >= bindings::size_column(ab) );
        BOOST_ASSERT( bindings::size(work.select(value_type())) >=
                min_size_work( bindings::size_column(ab) ));
        BOOST_ASSERT( bindings::size_column(ab) >= 0 );
        BOOST_ASSERT( bindings::size_minor(ab) == 1 ||
                bindings::stride_minor(ab) == 1 );
        BOOST_ASSERT( bindings::size_minor(q) == 1 ||
                bindings::stride_minor(q) == 1 );
        BOOST_ASSERT( bindings::stride_major(ab) >= bindings::bandwidth(ab,
                uplo())+1 );
        BOOST_ASSERT( vect == 'N' || vect == 'V' || vect == 'U' );
        return detail::hbtrd( vect, uplo(), bindings::size_column(ab),
                bindings::bandwidth(ab, uplo()), bindings::begin_value(ab),
                bindings::stride_major(ab), bindings::begin_value(d),
                bindings::begin_value(e), bindings::begin_value(q),
                bindings::stride_major(q),
                bindings::begin_value(work.select(value_type())) );
    }

    //
    // Static member function that
    // * Figures out the minimal workspace requirements, and passes
    //   the results to the user-defined workspace overload of the 
    //   invoke static member function
    // * Enables the unblocked algorithm (BLAS level 2)
    //
    template< typename MatrixAB, typename VectorD, typename VectorE,
            typename MatrixQ >
    static std::ptrdiff_t invoke( const char vect, MatrixAB& ab, VectorD& d,
            VectorE& e, MatrixQ& q, minimal_workspace work ) {
        namespace bindings = ::boost::numeric::bindings;
        typedef typename result_of::uplo_tag< MatrixAB >::type uplo;
        bindings::detail::array< value_type > tmp_work( min_size_work(
                bindings::size_column(ab) ) );
        return invoke( vect, ab, d, e, q, workspace( tmp_work ) );
    }

    //
    // Static member function that
    // * Figures out the optimal workspace requirements, and passes
    //   the results to the user-defined workspace overload of the 
    //   invoke static member
    // * Enables the blocked algorithm (BLAS level 3)
    //
    template< typename MatrixAB, typename VectorD, typename VectorE,
            typename MatrixQ >
    static std::ptrdiff_t invoke( const char vect, MatrixAB& ab, VectorD& d,
            VectorE& e, MatrixQ& q, optimal_workspace work ) {
        namespace bindings = ::boost::numeric::bindings;
        typedef typename result_of::uplo_tag< MatrixAB >::type uplo;
        return invoke( vect, ab, d, e, q, minimal_workspace() );
    }

    //
    // Static member function that returns the minimum size of
    // workspace-array work.
    //
    static std::ptrdiff_t min_size_work( const std::ptrdiff_t n ) {
        return n;
    }
};


//
// Functions for direct use. These functions are overloaded for temporaries,
// so that wrapped types can still be passed and used for write-access. In
// addition, if applicable, they are overloaded for user-defined workspaces.
// Calls to these functions are passed to the hbtrd_impl classes. In the 
// documentation, most overloads are collapsed to avoid a large number of
// prototypes which are very similar.
//

//
// Overloaded function for hbtrd. Its overload differs for
// * MatrixAB&
// * MatrixQ&
// * User-defined workspace
//
template< typename MatrixAB, typename VectorD, typename VectorE,
        typename MatrixQ, typename Workspace >
inline typename boost::enable_if< detail::is_workspace< Workspace >,
        std::ptrdiff_t >::type
hbtrd( const char vect, MatrixAB& ab, VectorD& d, VectorE& e, MatrixQ& q,
        Workspace work ) {
    return hbtrd_impl< typename bindings::value_type<
            MatrixAB >::type >::invoke( vect, ab, d, e, q, work );
}

//
// Overloaded function for hbtrd. Its overload differs for
// * MatrixAB&
// * MatrixQ&
// * Default workspace-type (optimal)
//
template< typename MatrixAB, typename VectorD, typename VectorE,
        typename MatrixQ >
inline typename boost::disable_if< detail::is_workspace< MatrixQ >,
        std::ptrdiff_t >::type
hbtrd( const char vect, MatrixAB& ab, VectorD& d, VectorE& e,
        MatrixQ& q ) {
    return hbtrd_impl< typename bindings::value_type<
            MatrixAB >::type >::invoke( vect, ab, d, e, q,
            optimal_workspace() );
}

//
// Overloaded function for hbtrd. Its overload differs for
// * const MatrixAB&
// * MatrixQ&
// * User-defined workspace
//
template< typename MatrixAB, typename VectorD, typename VectorE,
        typename MatrixQ, typename Workspace >
inline typename boost::enable_if< detail::is_workspace< Workspace >,
        std::ptrdiff_t >::type
hbtrd( const char vect, const MatrixAB& ab, VectorD& d, VectorE& e,
        MatrixQ& q, Workspace work ) {
    return hbtrd_impl< typename bindings::value_type<
            MatrixAB >::type >::invoke( vect, ab, d, e, q, work );
}

//
// Overloaded function for hbtrd. Its overload differs for
// * const MatrixAB&
// * MatrixQ&
// * Default workspace-type (optimal)
//
template< typename MatrixAB, typename VectorD, typename VectorE,
        typename MatrixQ >
inline typename boost::disable_if< detail::is_workspace< MatrixQ >,
        std::ptrdiff_t >::type
hbtrd( const char vect, const MatrixAB& ab, VectorD& d, VectorE& e,
        MatrixQ& q ) {
    return hbtrd_impl< typename bindings::value_type<
            MatrixAB >::type >::invoke( vect, ab, d, e, q,
            optimal_workspace() );
}

//
// Overloaded function for hbtrd. Its overload differs for
// * MatrixAB&
// * const MatrixQ&
// * User-defined workspace
//
template< typename MatrixAB, typename VectorD, typename VectorE,
        typename MatrixQ, typename Workspace >
inline typename boost::enable_if< detail::is_workspace< Workspace >,
        std::ptrdiff_t >::type
hbtrd( const char vect, MatrixAB& ab, VectorD& d, VectorE& e,
        const MatrixQ& q, Workspace work ) {
    return hbtrd_impl< typename bindings::value_type<
            MatrixAB >::type >::invoke( vect, ab, d, e, q, work );
}

//
// Overloaded function for hbtrd. Its overload differs for
// * MatrixAB&
// * const MatrixQ&
// * Default workspace-type (optimal)
//
template< typename MatrixAB, typename VectorD, typename VectorE,
        typename MatrixQ >
inline typename boost::disable_if< detail::is_workspace< MatrixQ >,
        std::ptrdiff_t >::type
hbtrd( const char vect, MatrixAB& ab, VectorD& d, VectorE& e,
        const MatrixQ& q ) {
    return hbtrd_impl< typename bindings::value_type<
            MatrixAB >::type >::invoke( vect, ab, d, e, q,
            optimal_workspace() );
}

//
// Overloaded function for hbtrd. Its overload differs for
// * const MatrixAB&
// * const MatrixQ&
// * User-defined workspace
//
template< typename MatrixAB, typename VectorD, typename VectorE,
        typename MatrixQ, typename Workspace >
inline typename boost::enable_if< detail::is_workspace< Workspace >,
        std::ptrdiff_t >::type
hbtrd( const char vect, const MatrixAB& ab, VectorD& d, VectorE& e,
        const MatrixQ& q, Workspace work ) {
    return hbtrd_impl< typename bindings::value_type<
            MatrixAB >::type >::invoke( vect, ab, d, e, q, work );
}

//
// Overloaded function for hbtrd. Its overload differs for
// * const MatrixAB&
// * const MatrixQ&
// * Default workspace-type (optimal)
//
template< typename MatrixAB, typename VectorD, typename VectorE,
        typename MatrixQ >
inline typename boost::disable_if< detail::is_workspace< MatrixQ >,
        std::ptrdiff_t >::type
hbtrd( const char vect, const MatrixAB& ab, VectorD& d, VectorE& e,
        const MatrixQ& q ) {
    return hbtrd_impl< typename bindings::value_type<
            MatrixAB >::type >::invoke( vect, ab, d, e, q,
            optimal_workspace() );
}

} // namespace lapack
} // namespace bindings
} // namespace numeric
} // namespace boost

#endif
