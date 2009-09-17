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

#ifndef BOOST_NUMERIC_BINDINGS_LAPACK_COMPUTATIONAL_GETRF_HPP
#define BOOST_NUMERIC_BINDINGS_LAPACK_COMPUTATIONAL_GETRF_HPP

#include <boost/assert.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/numeric/bindings/lapack/detail/lapack.h>
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
    inline void getrf( const integer_t m, const integer_t n, float* a,
            const integer_t lda, integer_t* ipiv, integer_t& info ) {
        LAPACK_SGETRF( &m, &n, a, &lda, ipiv, &info );
    }
    inline void getrf( const integer_t m, const integer_t n, double* a,
            const integer_t lda, integer_t* ipiv, integer_t& info ) {
        LAPACK_DGETRF( &m, &n, a, &lda, ipiv, &info );
    }
    inline void getrf( const integer_t m, const integer_t n,
            traits::complex_f* a, const integer_t lda, integer_t* ipiv,
            integer_t& info ) {
        LAPACK_CGETRF( &m, &n, traits::complex_ptr(a), &lda, ipiv, &info );
    }
    inline void getrf( const integer_t m, const integer_t n,
            traits::complex_d* a, const integer_t lda, integer_t* ipiv,
            integer_t& info ) {
        LAPACK_ZGETRF( &m, &n, traits::complex_ptr(a), &lda, ipiv, &info );
    }
}

// value-type based template
template< typename ValueType >
struct getrf_impl {

    typedef ValueType value_type;
    typedef typename traits::type_traits<ValueType>::real_type real_type;

    // templated specialization
    template< typename MatrixA, typename VectorIPIV >
    static void invoke( MatrixA& a, VectorIPIV& ipiv, integer_t& info ) {
        BOOST_ASSERT( traits::matrix_num_rows(a) >= 0 );
        BOOST_ASSERT( traits::matrix_num_columns(a) >= 0 );
        BOOST_ASSERT( traits::leading_dimension(a) >= std::max(1,
                traits::matrix_num_rows(a)) );
        BOOST_ASSERT( traits::vector_size(ipiv) >=
                std::min(traits::matrix_num_rows(a),
                traits::matrix_num_columns(a)) );
        detail::getrf( traits::matrix_num_rows(a),
                traits::matrix_num_columns(a), traits::matrix_storage(a),
                traits::leading_dimension(a), traits::vector_storage(ipiv),
                info );
    }
};


// template function to call getrf
template< typename MatrixA, typename VectorIPIV >
inline integer_t getrf( MatrixA& a, VectorIPIV& ipiv ) {
    typedef typename traits::matrix_traits< MatrixA >::value_type value_type;
    integer_t info(0);
    getrf_impl< value_type >::invoke( a, ipiv, info );
    return info;
}

}}}} // namespace boost::numeric::bindings::lapack

#endif
