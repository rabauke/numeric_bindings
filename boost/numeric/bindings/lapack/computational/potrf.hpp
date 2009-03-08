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

#ifndef BOOST_NUMERIC_BINDINGS_LAPACK_POTRF_HPP
#define BOOST_NUMERIC_BINDINGS_LAPACK_POTRF_HPP

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
    inline void potrf( char const uplo, integer_t const n, float* a,
            integer_t const lda, integer_t& info ) {
        LAPACK_SPOTRF( &uplo, &n, a, &lda, &info );
    }
    inline void potrf( char const uplo, integer_t const n, double* a,
            integer_t const lda, integer_t& info ) {
        LAPACK_DPOTRF( &uplo, &n, a, &lda, &info );
    }
    inline void potrf( char const uplo, integer_t const n,
            traits::complex_f* a, integer_t const lda, integer_t& info ) {
        LAPACK_CPOTRF( &uplo, &n, traits::complex_ptr(a), &lda, &info );
    }
    inline void potrf( char const uplo, integer_t const n,
            traits::complex_d* a, integer_t const lda, integer_t& info ) {
        LAPACK_ZPOTRF( &uplo, &n, traits::complex_ptr(a), &lda, &info );
    }
}

// value-type based template
template< typename ValueType >
struct potrf_impl {

    typedef ValueType value_type;
    typedef typename traits::type_traits<ValueType>::real_type real_type;

    // templated specialization
    template< typename MatrixA >
    static void compute( MatrixA& a, integer_t& info ) {
        
#ifndef NDEBUG
        assert( traits::matrix_uplo_tag(a) == 'U' ||
                traits::matrix_uplo_tag(a) == 'L' );
        assert( traits::matrix_num_columns(a) >= 0 );
        assert( traits::leading_dimension(a) >= std::max(1,
                traits::matrix_num_columns(a)) );
#endif
        detail::potrf( traits::matrix_uplo_tag(a),
                traits::matrix_num_columns(a), traits::matrix_storage(a),
                traits::leading_dimension(a), info );
    }
};


// template function to call potrf
template< typename MatrixA >
inline integer_t potrf( MatrixA& a ) {
    typedef typename traits::matrix_traits< MatrixA >::value_type value_type;
    integer_t info(0);
    potrf_impl< value_type >::compute( a, info );
    return info;
}

}}}} // namespace boost::numeric::bindings::lapack

#endif
