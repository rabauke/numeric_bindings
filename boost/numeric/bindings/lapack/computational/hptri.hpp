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

#ifndef BOOST_NUMERIC_BINDINGS_LAPACK_COMPUTATIONAL_HPTRI_HPP
#define BOOST_NUMERIC_BINDINGS_LAPACK_COMPUTATIONAL_HPTRI_HPP

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
#include <boost/numeric/bindings/uplo_tag.hpp>
#include <boost/numeric/bindings/value_type.hpp>
#include <boost/static_assert.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/type_traits/remove_const.hpp>
#include <boost/utility/enable_if.hpp>

//
// The LAPACK-backend for hptri is the netlib-compatible backend.
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
inline std::ptrdiff_t hptri( const UpLo uplo, const fortran_int_t n, float* ap,
        const fortran_int_t* ipiv, float* work ) {
    fortran_int_t info(0);
    LAPACK_SSPTRI( &lapack_option< UpLo >::value, &n, ap, ipiv, work, &info );
    return info;
}

//
// Overloaded function for dispatching to
// * netlib-compatible LAPACK backend (the default), and
// * double value-type.
//
template< typename UpLo >
inline std::ptrdiff_t hptri( const UpLo uplo, const fortran_int_t n,
        double* ap, const fortran_int_t* ipiv, double* work ) {
    fortran_int_t info(0);
    LAPACK_DSPTRI( &lapack_option< UpLo >::value, &n, ap, ipiv, work, &info );
    return info;
}

//
// Overloaded function for dispatching to
// * netlib-compatible LAPACK backend (the default), and
// * complex<float> value-type.
//
template< typename UpLo >
inline std::ptrdiff_t hptri( const UpLo uplo, const fortran_int_t n,
        std::complex<float>* ap, const fortran_int_t* ipiv,
        std::complex<float>* work ) {
    fortran_int_t info(0);
    LAPACK_CHPTRI( &lapack_option< UpLo >::value, &n, ap, ipiv, work, &info );
    return info;
}

//
// Overloaded function for dispatching to
// * netlib-compatible LAPACK backend (the default), and
// * complex<double> value-type.
//
template< typename UpLo >
inline std::ptrdiff_t hptri( const UpLo uplo, const fortran_int_t n,
        std::complex<double>* ap, const fortran_int_t* ipiv,
        std::complex<double>* work ) {
    fortran_int_t info(0);
    LAPACK_ZHPTRI( &lapack_option< UpLo >::value, &n, ap, ipiv, work, &info );
    return info;
}

} // namespace detail

//
// Value-type based template class. Use this class if you need a type
// for dispatching to hptri.
//
template< typename Value, typename Enable = void >
struct hptri_impl {};

//
// This implementation is enabled if Value is a real type.
//
template< typename Value >
struct hptri_impl< Value, typename boost::enable_if< is_real< Value > >::type > {

    typedef Value value_type;
    typedef typename remove_imaginary< Value >::type real_type;

    //
    // Static member function for user-defined workspaces, that
    // * Deduces the required arguments for dispatching to LAPACK, and
    // * Asserts that most arguments make sense.
    //
    template< typename MatrixAP, typename VectorIPIV, typename WORK >
    static std::ptrdiff_t invoke( MatrixAP& ap, const VectorIPIV& ipiv,
            detail::workspace1< WORK > work ) {
        namespace bindings = ::boost::numeric::bindings;
        typedef typename result_of::uplo_tag< MatrixAP >::type uplo;
        BOOST_STATIC_ASSERT( (bindings::is_mutable< MatrixAP >::value) );
        BOOST_ASSERT( bindings::size(ipiv) >= bindings::size_column(ap) );
        BOOST_ASSERT( bindings::size(work.select(real_type())) >=
                min_size_work( bindings::size_column(ap) ));
        BOOST_ASSERT( bindings::size_column(ap) >= 0 );
        return detail::hptri( uplo(), bindings::size_column(ap),
                bindings::begin_value(ap), bindings::begin_value(ipiv),
                bindings::begin_value(work.select(real_type())) );
    }

    //
    // Static member function that
    // * Figures out the minimal workspace requirements, and passes
    //   the results to the user-defined workspace overload of the 
    //   invoke static member function
    // * Enables the unblocked algorithm (BLAS level 2)
    //
    template< typename MatrixAP, typename VectorIPIV >
    static std::ptrdiff_t invoke( MatrixAP& ap, const VectorIPIV& ipiv,
            minimal_workspace ) {
        namespace bindings = ::boost::numeric::bindings;
        typedef typename result_of::uplo_tag< MatrixAP >::type uplo;
        bindings::detail::array< real_type > tmp_work( min_size_work(
                bindings::size_column(ap) ) );
        return invoke( ap, ipiv, workspace( tmp_work ) );
    }

    //
    // Static member function that
    // * Figures out the optimal workspace requirements, and passes
    //   the results to the user-defined workspace overload of the 
    //   invoke static member
    // * Enables the blocked algorithm (BLAS level 3)
    //
    template< typename MatrixAP, typename VectorIPIV >
    static std::ptrdiff_t invoke( MatrixAP& ap, const VectorIPIV& ipiv,
            optimal_workspace ) {
        namespace bindings = ::boost::numeric::bindings;
        typedef typename result_of::uplo_tag< MatrixAP >::type uplo;
        return invoke( ap, ipiv, minimal_workspace() );
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
// This implementation is enabled if Value is a complex type.
//
template< typename Value >
struct hptri_impl< Value, typename boost::enable_if< is_complex< Value > >::type > {

    typedef Value value_type;
    typedef typename remove_imaginary< Value >::type real_type;

    //
    // Static member function for user-defined workspaces, that
    // * Deduces the required arguments for dispatching to LAPACK, and
    // * Asserts that most arguments make sense.
    //
    template< typename MatrixAP, typename VectorIPIV, typename WORK >
    static std::ptrdiff_t invoke( MatrixAP& ap, const VectorIPIV& ipiv,
            detail::workspace1< WORK > work ) {
        namespace bindings = ::boost::numeric::bindings;
        typedef typename result_of::uplo_tag< MatrixAP >::type uplo;
        BOOST_STATIC_ASSERT( (bindings::is_mutable< MatrixAP >::value) );
        BOOST_ASSERT( bindings::size(ipiv) >= bindings::size_column(ap) );
        BOOST_ASSERT( bindings::size(work.select(value_type())) >=
                min_size_work( bindings::size_column(ap) ));
        BOOST_ASSERT( bindings::size_column(ap) >= 0 );
        return detail::hptri( uplo(), bindings::size_column(ap),
                bindings::begin_value(ap), bindings::begin_value(ipiv),
                bindings::begin_value(work.select(value_type())) );
    }

    //
    // Static member function that
    // * Figures out the minimal workspace requirements, and passes
    //   the results to the user-defined workspace overload of the 
    //   invoke static member function
    // * Enables the unblocked algorithm (BLAS level 2)
    //
    template< typename MatrixAP, typename VectorIPIV >
    static std::ptrdiff_t invoke( MatrixAP& ap, const VectorIPIV& ipiv,
            minimal_workspace ) {
        namespace bindings = ::boost::numeric::bindings;
        typedef typename result_of::uplo_tag< MatrixAP >::type uplo;
        bindings::detail::array< value_type > tmp_work( min_size_work(
                bindings::size_column(ap) ) );
        return invoke( ap, ipiv, workspace( tmp_work ) );
    }

    //
    // Static member function that
    // * Figures out the optimal workspace requirements, and passes
    //   the results to the user-defined workspace overload of the 
    //   invoke static member
    // * Enables the blocked algorithm (BLAS level 3)
    //
    template< typename MatrixAP, typename VectorIPIV >
    static std::ptrdiff_t invoke( MatrixAP& ap, const VectorIPIV& ipiv,
            optimal_workspace ) {
        namespace bindings = ::boost::numeric::bindings;
        typedef typename result_of::uplo_tag< MatrixAP >::type uplo;
        return invoke( ap, ipiv, minimal_workspace() );
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
// Calls to these functions are passed to the hptri_impl classes. In the 
// documentation, most overloads are collapsed to avoid a large number of
// prototypes which are very similar.
//

//
// Overloaded function for hptri. Its overload differs for
// * User-defined workspace
//
template< typename MatrixAP, typename VectorIPIV, typename Workspace >
inline typename boost::enable_if< detail::is_workspace< Workspace >,
        std::ptrdiff_t >::type
hptri( MatrixAP& ap, const VectorIPIV& ipiv, Workspace work ) {
    return hptri_impl< typename bindings::value_type<
            MatrixAP >::type >::invoke( ap, ipiv, work );
}

//
// Overloaded function for hptri. Its overload differs for
// * Default workspace-type (optimal)
//
template< typename MatrixAP, typename VectorIPIV >
inline typename boost::disable_if< detail::is_workspace< VectorIPIV >,
        std::ptrdiff_t >::type
hptri( MatrixAP& ap, const VectorIPIV& ipiv ) {
    return hptri_impl< typename bindings::value_type<
            MatrixAP >::type >::invoke( ap, ipiv, optimal_workspace() );
}

} // namespace lapack
} // namespace bindings
} // namespace numeric
} // namespace boost

#endif
