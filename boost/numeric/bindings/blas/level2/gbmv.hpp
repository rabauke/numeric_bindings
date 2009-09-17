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

#ifndef BOOST_NUMERIC_BINDINGS_BLAS_LEVEL2_GBMV_HPP
#define BOOST_NUMERIC_BINDINGS_BLAS_LEVEL2_GBMV_HPP

#include <boost/mpl/bool.hpp>
#include <boost/numeric/bindings/blas/detail/blas.h>
#include <boost/numeric/bindings/traits/traits.hpp>
#include <boost/numeric/bindings/traits/type_traits.hpp>
#include <boost/static_assert.hpp>
#include <boost/type_traits/is_same.hpp>

namespace boost {
namespace numeric {
namespace bindings {
namespace blas {
namespace level2 {

// overloaded functions to call blas
namespace detail {
    inline void gbmv( const char trans, const integer_t m, const integer_t n,
            const integer_t kl, const integer_t ku, const float alpha,
            const float* a, const integer_t lda, const float* x,
            const integer_t incx, const float beta, float* y,
            const integer_t incy ) {
        BLAS_SGBMV( &trans, &m, &n, &kl, &ku, &alpha, a, &lda, x, &incx,
                &beta, y, &incy );
    }
    inline void gbmv( const char trans, const integer_t m, const integer_t n,
            const integer_t kl, const integer_t ku, const double alpha,
            const double* a, const integer_t lda, const double* x,
            const integer_t incx, const double beta, double* y,
            const integer_t incy ) {
        BLAS_DGBMV( &trans, &m, &n, &kl, &ku, &alpha, a, &lda, x, &incx,
                &beta, y, &incy );
    }
    inline void gbmv( const char trans, const integer_t m, const integer_t n,
            const integer_t kl, const integer_t ku,
            const traits::complex_f alpha, const traits::complex_f* a,
            const integer_t lda, const traits::complex_f* x,
            const integer_t incx, const traits::complex_f beta,
            traits::complex_f* y, const integer_t incy ) {
        BLAS_CGBMV( &trans, &m, &n, &kl, &ku, traits::complex_ptr(&alpha),
                traits::complex_ptr(a), &lda, traits::complex_ptr(x), &incx,
                traits::complex_ptr(&beta), traits::complex_ptr(y), &incy );
    }
    inline void gbmv( const char trans, const integer_t m, const integer_t n,
            const integer_t kl, const integer_t ku,
            const traits::complex_d alpha, const traits::complex_d* a,
            const integer_t lda, const traits::complex_d* x,
            const integer_t incx, const traits::complex_d beta,
            traits::complex_d* y, const integer_t incy ) {
        BLAS_ZGBMV( &trans, &m, &n, &kl, &ku, traits::complex_ptr(&alpha),
                traits::complex_ptr(a), &lda, traits::complex_ptr(x), &incx,
                traits::complex_ptr(&beta), traits::complex_ptr(y), &incy );
    }
}

// value-type based template
template< typename ValueType >
struct gbmv_impl {

    typedef ValueType value_type;
    typedef typename traits::type_traits<ValueType>::real_type real_type;
    typedef void return_type;

    // templated specialization
    template< typename MatrixA, typename VectorX, typename VectorY >
    static return_type invoke( const char trans, const integer_t kl,
            const integer_t ku, const value_type alpha, const MatrixA& a,
            const VectorX& x, const value_type beta, VectorY& y ) {
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixA >::value_type, typename traits::vector_traits<
                VectorX >::value_type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixA >::value_type, typename traits::vector_traits<
                VectorY >::value_type >::value) );
        detail::gbmv( trans, traits::matrix_num_rows(a),
                traits::matrix_num_columns(a), kl, ku, alpha,
                traits::matrix_storage(a), traits::leading_dimension(a),
                traits::vector_storage(x), traits::vector_stride(x), beta,
                traits::vector_storage(y), traits::vector_stride(y) );
    }
};

// low-level template function for direct calls to level2::gbmv
template< typename MatrixA, typename VectorX, typename VectorY >
inline typename gbmv_impl< typename traits::matrix_traits<
        MatrixA >::value_type >::return_type
gbmv( const char trans, const integer_t kl, const integer_t ku,
        const typename traits::matrix_traits< MatrixA >::value_type alpha,
        const MatrixA& a, const VectorX& x,
        const typename traits::matrix_traits< MatrixA >::value_type beta,
        VectorY& y ) {
    typedef typename traits::matrix_traits< MatrixA >::value_type value_type;
    gbmv_impl< value_type >::invoke( trans, kl, ku, alpha, a, x, beta,
            y );
}

}}}}} // namespace boost::numeric::bindings::blas::level2

#endif
