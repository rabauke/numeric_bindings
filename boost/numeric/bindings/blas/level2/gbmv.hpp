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

#ifndef BOOST_NUMERIC_BINDINGS_BLAS_GBMV_HPP
#define BOOST_NUMERIC_BINDINGS_BLAS_GBMV_HPP

#include <boost/numeric/bindings/blas/blas.h>
#include <boost/numeric/bindings/traits/traits.hpp>
#include <boost/numeric/bindings/traits/type_traits.hpp>
#include <cassert>

namespace boost {
namespace numeric {
namespace bindings {
namespace blas {

//$DESCRIPTION

// overloaded functions to call blas
namespace detail {
    inline void gbmv( char const trans, integer_t const m, integer_t const n,
            integer_t const kl, integer_t const ku, float const alpha,
            float* a, integer_t const lda, float* x, integer_t const incx,
            float const beta, float* y, integer_t const incy ) {
        BLAS_SGBMV( &trans, &m, &n, &kl, &ku, &alpha, a, &lda, x, &incx,
                &beta, y, &incy );
    }
    inline void gbmv( char const trans, integer_t const m, integer_t const n,
            integer_t const kl, integer_t const ku, double const alpha,
            double* a, integer_t const lda, double* x, integer_t const incx,
            double const beta, double* y, integer_t const incy ) {
        BLAS_DGBMV( &trans, &m, &n, &kl, &ku, &alpha, a, &lda, x, &incx,
                &beta, y, &incy );
    }
    inline void gbmv( char const trans, integer_t const m, integer_t const n,
            integer_t const kl, integer_t const ku,
            traits::complex_f const alpha, traits::complex_f* a,
            integer_t const lda, traits::complex_f* x, integer_t const incx,
            traits::complex_f const beta, traits::complex_f* y,
            integer_t const incy ) {
        BLAS_CGBMV( &trans, &m, &n, &kl, &ku, traits::complex_ptr(&alpha),
                traits::complex_ptr(a), &lda, traits::complex_ptr(x), &incx,
                traits::complex_ptr(&beta), traits::complex_ptr(y), &incy );
    }
    inline void gbmv( char const trans, integer_t const m, integer_t const n,
            integer_t const kl, integer_t const ku,
            traits::complex_d const alpha, traits::complex_d* a,
            integer_t const lda, traits::complex_d* x, integer_t const incx,
            traits::complex_d const beta, traits::complex_d* y,
            integer_t const incy ) {
        BLAS_ZGBMV( &trans, &m, &n, &kl, &ku, traits::complex_ptr(&alpha),
                traits::complex_ptr(a), &lda, traits::complex_ptr(x), &incx,
                traits::complex_ptr(&beta), traits::complex_ptr(y), &incy );
    }
}

// value-type based template
template< typename ValueType >
struct gbmv_impl {

    typedef ValueType value_type;
    typedef void return_type;

    // templated specialization
    template< typename MatrixA, typename VectorX, typename VectorY >
    static return_type compute( char const trans, integer_t const kl,
            integer_t const ku, traits::complex_d const alpha, MatrixA& a,
            VectorX& x, traits::complex_d const beta, VectorY& y ) {
        detail::gbmv( trans, traits::matrix_size1(a),
                traits::matrix_size2(a), kl, ku, alpha,
                traits::matrix_storage(a), traits::leading_dimension(a),
                traits::vector_storage(x), traits::vector_stride(x), beta,
                traits::vector_storage(y), traits::vector_stride(y) );
    }
};

// template function to call gbmv
template< typename MatrixA, typename VectorX, typename VectorY >
inline integer_t gbmv( char const trans, integer_t const kl,
        integer_t const ku, traits::complex_d const alpha, MatrixA& a,
        VectorX& x, traits::complex_d const beta, VectorY& y ) {
    typedef typename traits::matrix_traits< MatrixA >::value_type value_type;
    gbmv_impl< value_type >::compute( trans, kl, ku, alpha, a, x, beta,
            y );
}


}}}} // namespace boost::numeric::bindings::blas

#endif
