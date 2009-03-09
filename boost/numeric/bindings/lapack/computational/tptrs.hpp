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

#ifndef BOOST_NUMERIC_BINDINGS_LAPACK_TPTRS_HPP
#define BOOST_NUMERIC_BINDINGS_LAPACK_TPTRS_HPP

#include <boost/assert.hpp>
#include <boost/numeric/bindings/lapack/lapack.h>
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
    inline void tptrs( char const uplo, char const trans, char const diag,
            integer_t const n, integer_t const nrhs, float* ap, float* b,
            integer_t const ldb, integer_t& info ) {
        LAPACK_STPTRS( &uplo, &trans, &diag, &n, &nrhs, ap, b, &ldb, &info );
    }
    inline void tptrs( char const uplo, char const trans, char const diag,
            integer_t const n, integer_t const nrhs, double* ap, double* b,
            integer_t const ldb, integer_t& info ) {
        LAPACK_DTPTRS( &uplo, &trans, &diag, &n, &nrhs, ap, b, &ldb, &info );
    }
    inline void tptrs( char const uplo, char const trans, char const diag,
            integer_t const n, integer_t const nrhs, traits::complex_f* ap,
            traits::complex_f* b, integer_t const ldb, integer_t& info ) {
        LAPACK_CTPTRS( &uplo, &trans, &diag, &n, &nrhs,
                traits::complex_ptr(ap), traits::complex_ptr(b), &ldb, &info );
    }
    inline void tptrs( char const uplo, char const trans, char const diag,
            integer_t const n, integer_t const nrhs, traits::complex_d* ap,
            traits::complex_d* b, integer_t const ldb, integer_t& info ) {
        LAPACK_ZTPTRS( &uplo, &trans, &diag, &n, &nrhs,
                traits::complex_ptr(ap), traits::complex_ptr(b), &ldb, &info );
    }
}

// value-type based template
template< typename ValueType >
struct tptrs_impl {

    typedef ValueType value_type;
    typedef typename traits::type_traits<ValueType>::real_type real_type;

    // templated specialization
    template< typename MatrixAP, typename MatrixB >
    static void compute( char const uplo, char const trans, char const diag,
            integer_t const n, MatrixAP& ap, MatrixB& b, integer_t& info ) {
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixAP >::value_type, typename traits::matrix_traits<
                MatrixB >::value_type >::value) );
        BOOST_ASSERT( uplo == 'U' || uplo == 'L' );
        BOOST_ASSERT( trans == 'N' || trans == 'T' || trans == 'C' );
        BOOST_ASSERT( diag == 'N' || diag == 'U' );
        BOOST_ASSERT( n >= 0 );
        BOOST_ASSERT( traits::matrix_num_columns(b) >= 0 );
        BOOST_ASSERT( traits::leading_dimension(b) >= std::max(1,n) );
        detail::tptrs( uplo, trans, diag, n, traits::matrix_num_columns(b),
                traits::matrix_storage(ap), traits::matrix_storage(b),
                traits::leading_dimension(b), info );
    }
};


// template function to call tptrs
template< typename MatrixAP, typename MatrixB >
inline integer_t tptrs( char const uplo, char const trans,
        char const diag, integer_t const n, MatrixAP& ap, MatrixB& b ) {
    typedef typename traits::matrix_traits< MatrixAP >::value_type value_type;
    integer_t info(0);
    tptrs_impl< value_type >::compute( uplo, trans, diag, n, ap, b,
            info );
    return info;
}

}}}} // namespace boost::numeric::bindings::lapack

#endif
