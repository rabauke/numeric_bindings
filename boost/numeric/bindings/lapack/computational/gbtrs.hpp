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

#ifndef BOOST_NUMERIC_BINDINGS_LAPACK_COMPUTATIONAL_GBTRS_HPP
#define BOOST_NUMERIC_BINDINGS_LAPACK_COMPUTATIONAL_GBTRS_HPP

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

inline void gbtrs( const char trans, const integer_t n, const integer_t kl,
        const integer_t ku, const integer_t nrhs, const float* ab,
        const integer_t ldab, const integer_t* ipiv, float* b,
        const integer_t ldb, integer_t& info ) {
    LAPACK_SGBTRS( &trans, &n, &kl, &ku, &nrhs, ab, &ldab, ipiv, b, &ldb,
            &info );
}

inline void gbtrs( const char trans, const integer_t n, const integer_t kl,
        const integer_t ku, const integer_t nrhs, const double* ab,
        const integer_t ldab, const integer_t* ipiv, double* b,
        const integer_t ldb, integer_t& info ) {
    LAPACK_DGBTRS( &trans, &n, &kl, &ku, &nrhs, ab, &ldab, ipiv, b, &ldb,
            &info );
}

inline void gbtrs( const char trans, const integer_t n, const integer_t kl,
        const integer_t ku, const integer_t nrhs, const traits::complex_f* ab,
        const integer_t ldab, const integer_t* ipiv, traits::complex_f* b,
        const integer_t ldb, integer_t& info ) {
    LAPACK_CGBTRS( &trans, &n, &kl, &ku, &nrhs, traits::complex_ptr(ab),
            &ldab, ipiv, traits::complex_ptr(b), &ldb, &info );
}

inline void gbtrs( const char trans, const integer_t n, const integer_t kl,
        const integer_t ku, const integer_t nrhs, const traits::complex_d* ab,
        const integer_t ldab, const integer_t* ipiv, traits::complex_d* b,
        const integer_t ldb, integer_t& info ) {
    LAPACK_ZGBTRS( &trans, &n, &kl, &ku, &nrhs, traits::complex_ptr(ab),
            &ldab, ipiv, traits::complex_ptr(b), &ldb, &info );
}

} // namespace detail

// value-type based template
template< typename ValueType >
struct gbtrs_impl {

    typedef ValueType value_type;
    typedef typename traits::type_traits<ValueType>::real_type real_type;

    // templated specialization
    template< typename MatrixAB, typename VectorIPIV, typename MatrixB >
    static void invoke( const char trans, const integer_t n,
            const integer_t kl, const integer_t ku, const MatrixAB& ab,
            const VectorIPIV& ipiv, MatrixB& b, integer_t& info ) {
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixAB >::value_type, typename traits::matrix_traits<
                MatrixB >::value_type >::value) );
        BOOST_ASSERT( trans == 'N' || trans == 'T' || trans == 'C' );
        BOOST_ASSERT( n >= 0 );
        BOOST_ASSERT( kl >= 0 );
        BOOST_ASSERT( ku >= 0 );
        BOOST_ASSERT( traits::matrix_num_columns(b) >= 0 );
        BOOST_ASSERT( traits::leading_dimension(ab) >= 2 );
        BOOST_ASSERT( traits::vector_size(ipiv) >= n );
        BOOST_ASSERT( traits::leading_dimension(b) >= std::max(1,n) );
        detail::gbtrs( trans, n, kl, ku, traits::matrix_num_columns(b),
                traits::matrix_storage(ab), traits::leading_dimension(ab),
                traits::vector_storage(ipiv), traits::matrix_storage(b),
                traits::leading_dimension(b), info );
    }
};


// template function to call gbtrs
template< typename MatrixAB, typename VectorIPIV, typename MatrixB >
inline integer_t gbtrs( const char trans, const integer_t n,
        const integer_t kl, const integer_t ku, const MatrixAB& ab,
        const VectorIPIV& ipiv, MatrixB& b ) {
    typedef typename traits::matrix_traits< MatrixAB >::value_type value_type;
    integer_t info(0);
    gbtrs_impl< value_type >::invoke( trans, n, kl, ku, ab, ipiv, b,
            info );
    return info;
}

} // namespace lapack
} // namespace bindings
} // namespace numeric
} // namespace boost

#endif
