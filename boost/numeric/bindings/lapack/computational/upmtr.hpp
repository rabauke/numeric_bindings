//
// Copyright (c) 2003--2009
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

#ifndef BOOST_NUMERIC_BINDINGS_LAPACK_COMPUTATIONAL_UPMTR_HPP
#define BOOST_NUMERIC_BINDINGS_LAPACK_COMPUTATIONAL_UPMTR_HPP

#include <boost/assert.hpp>
#include <boost/numeric/bindings/lapack/detail/lapack.h>
#include <boost/numeric/bindings/lapack/workspace.hpp>
#include <boost/numeric/bindings/traits/detail/array.hpp>
#include <boost/numeric/bindings/traits/traits.hpp>
#include <boost/numeric/bindings/traits/type_traits.hpp>
#include <boost/static_assert.hpp>
#include <boost/type_traits/is_same.hpp>

namespace boost {
namespace numeric {
namespace bindings {
namespace lapack {

//$DESCRIPTION

// overloaded functions to call lapack
namespace detail {
    inline void upmtr( char const side, char const uplo, char const trans,
            integer_t const m, integer_t const n, traits::complex_f* ap,
            traits::complex_f* tau, traits::complex_f* c, integer_t const ldc,
            traits::complex_f* work, integer_t& info ) {
        LAPACK_CUPMTR( &side, &uplo, &trans, &m, &n, traits::complex_ptr(ap),
                traits::complex_ptr(tau), traits::complex_ptr(c), &ldc,
                traits::complex_ptr(work), &info );
    }
    inline void upmtr( char const side, char const uplo, char const trans,
            integer_t const m, integer_t const n, traits::complex_d* ap,
            traits::complex_d* tau, traits::complex_d* c, integer_t const ldc,
            traits::complex_d* work, integer_t& info ) {
        LAPACK_ZUPMTR( &side, &uplo, &trans, &m, &n, traits::complex_ptr(ap),
                traits::complex_ptr(tau), traits::complex_ptr(c), &ldc,
                traits::complex_ptr(work), &info );
    }
}

// value-type based template
template< typename ValueType >
struct upmtr_impl {

    typedef ValueType value_type;
    typedef typename traits::type_traits<ValueType>::real_type real_type;

    // user-defined workspace specialization
    template< typename VectorAP, typename VectorTAU, typename MatrixC,
            typename WORK >
    static void compute( char const side, char const uplo, char const trans,
            VectorAP& ap, VectorTAU& tau, MatrixC& c, integer_t& info,
            detail::workspace1< WORK > work ) {
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::vector_traits<
                VectorAP >::value_type, typename traits::vector_traits<
                VectorTAU >::value_type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::vector_traits<
                VectorAP >::value_type, typename traits::matrix_traits<
                MatrixC >::value_type >::value) );
        BOOST_ASSERT( side == 'L' || side == 'R' );
        BOOST_ASSERT( uplo == 'U' || uplo == 'L' );
        BOOST_ASSERT( trans == 'N' || trans == 'C' );
        BOOST_ASSERT( traits::matrix_num_rows(c) >= 0 );
        BOOST_ASSERT( traits::matrix_num_columns(c) >= 0 );
        BOOST_ASSERT( traits::leading_dimension(c) >= std::max(1,
                traits::matrix_num_rows(c)) );
        BOOST_ASSERT( traits::vector_size(work.select(value_type())) >=
                min_size_work( $CALL_MIN_SIZE ));
        detail::upmtr( side, uplo, trans, traits::matrix_num_rows(c),
                traits::matrix_num_columns(c), traits::vector_storage(ap),
                traits::vector_storage(tau), traits::matrix_storage(c),
                traits::leading_dimension(c),
                traits::vector_storage(work.select(value_type())), info );
    }

    // minimal workspace specialization
    template< typename VectorAP, typename VectorTAU, typename MatrixC >
    static void compute( char const side, char const uplo, char const trans,
            VectorAP& ap, VectorTAU& tau, MatrixC& c, integer_t& info,
            minimal_workspace work ) {
        traits::detail::array< value_type > tmp_work( min_size_work(
                $CALL_MIN_SIZE ) );
        compute( side, uplo, trans, ap, tau, c, info, workspace( tmp_work ) );
    }

    // optimal workspace specialization
    template< typename VectorAP, typename VectorTAU, typename MatrixC >
    static void compute( char const side, char const uplo, char const trans,
            VectorAP& ap, VectorTAU& tau, MatrixC& c, integer_t& info,
            optimal_workspace work ) {
        compute( side, uplo, trans, ap, tau, c, info, minimal_workspace() );
    }

    static integer_t min_size_work( $ARGUMENTS ) {
        $MIN_SIZE
    }
};


// template function to call upmtr
template< typename VectorAP, typename VectorTAU, typename MatrixC,
        typename Workspace >
inline integer_t upmtr( char const side, char const uplo,
        char const trans, VectorAP& ap, VectorTAU& tau, MatrixC& c,
        Workspace work = optimal_workspace() ) {
    typedef typename traits::vector_traits< VectorAP >::value_type value_type;
    integer_t info(0);
    upmtr_impl< value_type >::compute( side, uplo, trans, ap, tau, c,
            info, work );
    return info;
}

}}}} // namespace boost::numeric::bindings::lapack

#endif
