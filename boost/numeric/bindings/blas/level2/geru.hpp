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

#ifndef BOOST_NUMERIC_BINDINGS_BLAS_LEVEL2_GERU_HPP
#define BOOST_NUMERIC_BINDINGS_BLAS_LEVEL2_GERU_HPP

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
    inline void geru( integer_t const m, integer_t const n,
            traits::complex_f const alpha, traits::complex_f* x,
            integer_t const incx, traits::complex_f* y, integer_t const incy,
            traits::complex_f* a, integer_t const lda ) {
        BLAS_CGERU( &m, &n, traits::complex_ptr(&alpha),
                traits::complex_ptr(x), &incx, traits::complex_ptr(y), &incy,
                traits::complex_ptr(a), &lda );
    }
    inline void geru( integer_t const m, integer_t const n,
            traits::complex_d const alpha, traits::complex_d* x,
            integer_t const incx, traits::complex_d* y, integer_t const incy,
            traits::complex_d* a, integer_t const lda ) {
        BLAS_ZGERU( &m, &n, traits::complex_ptr(&alpha),
                traits::complex_ptr(x), &incx, traits::complex_ptr(y), &incy,
                traits::complex_ptr(a), &lda );
    }
}

// value-type based template
template< typename ValueType >
struct geru_impl {

    typedef ValueType value_type;
    typedef typename traits::type_traits<ValueType>::real_type real_type;
    typedef void return_type;

    // templated specialization
    template< typename VectorX, typename VectorY, typename MatrixA >
    static return_type invoke( value_type const alpha, VectorX& x, VectorY& y,
            MatrixA& a ) {
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::vector_traits<
                VectorX >::value_type, typename traits::vector_traits<
                VectorY >::value_type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::vector_traits<
                VectorX >::value_type, typename traits::matrix_traits<
                MatrixA >::value_type >::value) );
        detail::geru( traits::matrix_num_rows(a),
                traits::matrix_num_columns(a), alpha,
                traits::vector_storage(x), traits::vector_stride(x),
                traits::vector_storage(y), traits::vector_stride(y),
                traits::matrix_storage(a), traits::leading_dimension(a) );
    }
};

// low-level template function for direct calls to level2::geru
template< typename VectorX, typename VectorY, typename MatrixA >
inline typename geru_impl< typename traits::vector_traits<
        VectorX >::value_type >::return_type
geru( typename traits::vector_traits< VectorX >::value_type const alpha,
        VectorX& x, VectorY& y, MatrixA& a ) {
    typedef typename traits::vector_traits< VectorX >::value_type value_type;
    geru_impl< value_type >::invoke( alpha, x, y, a );
}

}}}}} // namespace boost::numeric::bindings::blas::level2

#endif
