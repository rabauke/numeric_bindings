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

#ifndef BOOST_NUMERIC_BINDINGS_BLAS_LEVEL2_HPR2_HPP
#define BOOST_NUMERIC_BINDINGS_BLAS_LEVEL2_HPR2_HPP

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
    inline void hpr2( char const uplo, integer_t const n,
            traits::complex_f const alpha, traits::complex_f* x,
            integer_t const incx, traits::complex_f* y, integer_t const incy,
            traits::complex_f* ap ) {
        BLAS_CHPR2( &uplo, &n, traits::complex_ptr(&alpha),
                traits::complex_ptr(x), &incx, traits::complex_ptr(y), &incy,
                traits::complex_ptr(ap) );
    }
    inline void hpr2( char const uplo, integer_t const n,
            traits::complex_d const alpha, traits::complex_d* x,
            integer_t const incx, traits::complex_d* y, integer_t const incy,
            traits::complex_d* ap ) {
        BLAS_ZHPR2( &uplo, &n, traits::complex_ptr(&alpha),
                traits::complex_ptr(x), &incx, traits::complex_ptr(y), &incy,
                traits::complex_ptr(ap) );
    }
}

// value-type based template
template< typename ValueType >
struct hpr2_impl {

    typedef ValueType value_type;
    typedef typename traits::type_traits<ValueType>::real_type real_type;
    typedef void return_type;

    // templated specialization
    template< typename VectorX, typename VectorY, typename MatrixAP >
    static return_type invoke( value_type const alpha, VectorX& x, VectorY& y,
            MatrixAP& ap ) {
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::vector_traits<
                VectorX >::value_type, typename traits::vector_traits<
                VectorY >::value_type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::vector_traits<
                VectorX >::value_type, typename traits::matrix_traits<
                MatrixAP >::value_type >::value) );
        detail::hpr2( traits::matrix_uplo_tag(ap),
                traits::matrix_num_columns(ap), alpha,
                traits::vector_storage(x), traits::vector_stride(x),
                traits::vector_storage(y), traits::vector_stride(y),
                traits::matrix_storage(ap) );
    }
};

// low-level template function for direct calls to level2::hpr2
template< typename VectorX, typename VectorY, typename MatrixAP >
inline typename hpr2_impl< typename traits::vector_traits<
        VectorX >::value_type >::return_type
hpr2( typename traits::vector_traits< VectorX >::value_type const alpha,
        VectorX& x, VectorY& y, MatrixAP& ap ) {
    typedef typename traits::vector_traits< VectorX >::value_type value_type;
    hpr2_impl< value_type >::invoke( alpha, x, y, ap );
}

}}}}} // namespace boost::numeric::bindings::blas::level2

#endif
