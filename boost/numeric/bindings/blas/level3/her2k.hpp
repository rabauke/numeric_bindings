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

#ifndef BOOST_NUMERIC_BINDINGS_BLAS_HER2K_HPP
#define BOOST_NUMERIC_BINDINGS_BLAS_HER2K_HPP

#include <boost/numeric/bindings/blas/blas.h>
#include <boost/numeric/bindings/traits/traits.hpp>
#include <boost/numeric/bindings/traits/type_traits.hpp>
#include <boost/static_assert.hpp>
#include <boost/type_traits/is_same.hpp>
#include <cassert>

namespace boost {
namespace numeric {
namespace bindings {
namespace blas {

//$DESCRIPTION

// overloaded functions to call blas
namespace detail {
    inline void her2k( char const uplo, char const trans, integer_t const n,
            integer_t const k, traits::complex_f const alpha,
            traits::complex_f* a, integer_t const lda, traits::complex_f* b,
            integer_t const ldb, float const beta, traits::complex_f* c,
            integer_t const ldc ) {
        BLAS_CHER2K( &uplo, &trans, &n, &k, traits::complex_ptr(&alpha),
                traits::complex_ptr(a), &lda, traits::complex_ptr(b), &ldb,
                &beta, traits::complex_ptr(c), &ldc );
    }
    inline void her2k( char const uplo, char const trans, integer_t const n,
            integer_t const k, traits::complex_d const alpha,
            traits::complex_d* a, integer_t const lda, traits::complex_d* b,
            integer_t const ldb, double const beta, traits::complex_d* c,
            integer_t const ldc ) {
        BLAS_ZHER2K( &uplo, &trans, &n, &k, traits::complex_ptr(&alpha),
                traits::complex_ptr(a), &lda, traits::complex_ptr(b), &ldb,
                &beta, traits::complex_ptr(c), &ldc );
    }
}

// value-type based template
template< typename ValueType >
struct her2k_impl {

    typedef ValueType value_type;
    typedef void return_type;

    // templated specialization
    template< typename MatrixA, typename MatrixB, typename MatrixC >
    static return_type compute( char const trans, integer_t const k,
            value_type const alpha, MatrixA& a, MatrixB& b,
            real_type const beta, MatrixC& c ) {
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixA >::value_type, typename traits::matrix_traits<
                MatrixB >::value_type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixA >::value_type, typename traits::matrix_traits<
                MatrixC >::value_type >::value) );
        detail::her2k( traits::matrix_uplo_tag(c), trans,
                traits::matrix_num_columns(c), k, alpha,
                traits::matrix_storage(a), traits::leading_dimension(a),
                traits::matrix_storage(b), traits::leading_dimension(b), beta,
                traits::matrix_storage(c), traits::leading_dimension(c) );
    }
};

// template function to call her2k
template< typename MatrixA, typename MatrixB, typename MatrixC >
inline typename her2k_impl< typename traits::matrix_traits<
        MatrixA >::value_type >::return_type
her2k( char const trans, integer_t const k,
        typename traits::matrix_traits< MatrixA >::value_type const alpha,
        MatrixA& a, MatrixB& b, typename traits::matrix_traits<
        MatrixA >::value_type const beta, MatrixC& c ) {
    typedef typename traits::matrix_traits< MatrixA >::value_type value_type;
    her2k_impl< value_type >::compute( trans, k, alpha, a, b, beta, c );
}

}}}} // namespace boost::numeric::bindings::blas

#endif
