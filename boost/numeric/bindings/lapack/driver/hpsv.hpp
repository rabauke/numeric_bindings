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

#ifndef BOOST_NUMERIC_BINDINGS_LAPACK_HPSV_HPP
#define BOOST_NUMERIC_BINDINGS_LAPACK_HPSV_HPP

#include <boost/numeric/bindings/lapack/lapack.h>
#include <boost/numeric/bindings/traits/traits.hpp>
#include <boost/numeric/bindings/traits/type_traits.hpp>
#include <boost/static_assert.hpp>
#include <boost/type_traits/is_same.hpp>
#include <cassert>

namespace boost {
namespace numeric {
namespace bindings {
namespace lapack {

//$DESCRIPTION

// overloaded functions to call lapack
namespace detail {
    inline void hpsv( char const uplo, integer_t const n,
            integer_t const nrhs, traits::complex_f* ap, integer_t* ipiv,
            traits::complex_f* b, integer_t const ldb, integer_t& info ) {
        LAPACK_CHPSV( &uplo, &n, &nrhs, traits::complex_ptr(ap), ipiv,
                traits::complex_ptr(b), &ldb, &info );
    }
    inline void hpsv( char const uplo, integer_t const n,
            integer_t const nrhs, traits::complex_d* ap, integer_t* ipiv,
            traits::complex_d* b, integer_t const ldb, integer_t& info ) {
        LAPACK_ZHPSV( &uplo, &n, &nrhs, traits::complex_ptr(ap), ipiv,
                traits::complex_ptr(b), &ldb, &info );
    }
}

// value-type based template
template< typename ValueType >
struct hpsv_impl {

    typedef ValueType value_type;
    typedef typename traits::type_traits<ValueType>::real_type real_type;

    // templated specialization
    template< typename MatrixAP, typename VectorIPIV, typename MatrixB >
    static void compute( MatrixAP& ap, VectorIPIV& ipiv, MatrixB& b,
            integer_t& info ) {
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixAP >::value_type, typename traits::matrix_traits<
                MatrixB >::value_type >::value) );
#ifndef NDEBUG
        assert( traits::matrix_uplo_tag(ap) == 'U' ||
                traits::matrix_uplo_tag(ap) == 'L' );
        assert( traits::matrix_num_columns(ap) >= 0 );
        assert( traits::matrix_num_columns(b) >= 0 );
        assert( traits::leading_dimension(b) >= std::max(1,
                traits::matrix_num_columns(ap)) );
#endif
        detail::hpsv( traits::matrix_uplo_tag(ap),
                traits::matrix_num_columns(ap), traits::matrix_num_columns(b),
                traits::matrix_storage(ap), traits::vector_storage(ipiv),
                traits::matrix_storage(b), traits::leading_dimension(b),
                info );
    }
};


// template function to call hpsv
template< typename MatrixAP, typename VectorIPIV, typename MatrixB >
inline integer_t hpsv( MatrixAP& ap, VectorIPIV& ipiv, MatrixB& b ) {
    typedef typename traits::matrix_traits< MatrixAP >::value_type value_type;
    integer_t info(0);
    hpsv_impl< value_type >::compute( ap, ipiv, b, info );
    return info;
}

}}}} // namespace boost::numeric::bindings::lapack

#endif
