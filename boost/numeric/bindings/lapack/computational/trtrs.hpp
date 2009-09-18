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

#ifndef BOOST_NUMERIC_BINDINGS_LAPACK_COMPUTATIONAL_TRTRS_HPP
#define BOOST_NUMERIC_BINDINGS_LAPACK_COMPUTATIONAL_TRTRS_HPP

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

inline void trtrs( const char uplo, const char trans, const char diag,
        const integer_t n, const integer_t nrhs, const float* a,
        const integer_t lda, float* b, const integer_t ldb, integer_t& info ) {
    LAPACK_STRTRS( &uplo, &trans, &diag, &n, &nrhs, a, &lda, b, &ldb, &info );
}

inline void trtrs( const char uplo, const char trans, const char diag,
        const integer_t n, const integer_t nrhs, const double* a,
        const integer_t lda, double* b, const integer_t ldb,
        integer_t& info ) {
    LAPACK_DTRTRS( &uplo, &trans, &diag, &n, &nrhs, a, &lda, b, &ldb, &info );
}

inline void trtrs( const char uplo, const char trans, const char diag,
        const integer_t n, const integer_t nrhs, const traits::complex_f* a,
        const integer_t lda, traits::complex_f* b, const integer_t ldb,
        integer_t& info ) {
    LAPACK_CTRTRS( &uplo, &trans, &diag, &n, &nrhs, traits::complex_ptr(a),
            &lda, traits::complex_ptr(b), &ldb, &info );
}

inline void trtrs( const char uplo, const char trans, const char diag,
        const integer_t n, const integer_t nrhs, const traits::complex_d* a,
        const integer_t lda, traits::complex_d* b, const integer_t ldb,
        integer_t& info ) {
    LAPACK_ZTRTRS( &uplo, &trans, &diag, &n, &nrhs, traits::complex_ptr(a),
            &lda, traits::complex_ptr(b), &ldb, &info );
}

} // namespace detail

// value-type based template
template< typename ValueType >
struct trtrs_impl {

    typedef ValueType value_type;
    typedef typename traits::type_traits<ValueType>::real_type real_type;

    // templated specialization
    template< typename MatrixA, typename MatrixB >
    static void invoke( const char uplo, const char trans, const char diag,
            const MatrixA& a, MatrixB& b, integer_t& info ) {
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixA >::value_type, typename traits::matrix_traits<
                MatrixB >::value_type >::value) );
        BOOST_ASSERT( uplo == 'U' || uplo == 'L' );
        BOOST_ASSERT( trans == 'N' || trans == 'T' || trans == 'C' );
        BOOST_ASSERT( diag == 'N' || diag == 'U' );
        BOOST_ASSERT( traits::matrix_num_columns(a) >= 0 );
        BOOST_ASSERT( traits::matrix_num_columns(b) >= 0 );
        BOOST_ASSERT( traits::leading_dimension(a) >= std::max(1,
                traits::matrix_num_columns(a)) );
        BOOST_ASSERT( traits::leading_dimension(b) >= std::max(1,
                traits::matrix_num_columns(a)) );
        detail::trtrs( uplo, trans, diag, traits::matrix_num_columns(a),
                traits::matrix_num_columns(b), traits::matrix_storage(a),
                traits::leading_dimension(a), traits::matrix_storage(b),
                traits::leading_dimension(b), info );
    }
};


// template function to call trtrs
template< typename MatrixA, typename MatrixB >
inline integer_t trtrs( const char uplo, const char trans,
        const char diag, const MatrixA& a, MatrixB& b ) {
    typedef typename traits::matrix_traits< MatrixA >::value_type value_type;
    integer_t info(0);
    trtrs_impl< value_type >::invoke( uplo, trans, diag, a, b, info );
    return info;
}

} // namespace lapack
} // namespace bindings
} // namespace numeric
} // namespace boost

#endif
