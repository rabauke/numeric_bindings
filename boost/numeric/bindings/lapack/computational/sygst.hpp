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

#ifndef BOOST_NUMERIC_BINDINGS_LAPACK_COMPUTATIONAL_SYGST_HPP
#define BOOST_NUMERIC_BINDINGS_LAPACK_COMPUTATIONAL_SYGST_HPP

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

inline void sygst( const integer_t itype, const char uplo, const integer_t n,
        float* a, const integer_t lda, const float* b, const integer_t ldb,
        integer_t& info ) {
    LAPACK_SSYGST( &itype, &uplo, &n, a, &lda, b, &ldb, &info );
}
inline void sygst( const integer_t itype, const char uplo, const integer_t n,
        double* a, const integer_t lda, const double* b, const integer_t ldb,
        integer_t& info ) {
    LAPACK_DSYGST( &itype, &uplo, &n, a, &lda, b, &ldb, &info );
}
} // namespace detail

// value-type based template
template< typename ValueType >
struct sygst_impl {

    typedef ValueType value_type;
    typedef typename traits::type_traits<ValueType>::real_type real_type;

    // templated specialization
    template< typename MatrixA, typename MatrixB >
    static void invoke( const integer_t itype, const integer_t n, MatrixA& a,
            const MatrixB& b, integer_t& info ) {
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixA >::value_type, typename traits::matrix_traits<
                MatrixB >::value_type >::value) );
        BOOST_ASSERT( traits::matrix_uplo_tag(a) == 'U' ||
                traits::matrix_uplo_tag(a) == 'L' );
        BOOST_ASSERT( n >= 0 );
        BOOST_ASSERT( traits::leading_dimension(a) >= std::max<
                std::ptrdiff_t >(1,n) );
        BOOST_ASSERT( traits::leading_dimension(b) >= std::max<
                std::ptrdiff_t >(1,n) );
        detail::sygst( itype, traits::matrix_uplo_tag(a), n,
                traits::matrix_storage(a), traits::leading_dimension(a),
                traits::matrix_storage(b), traits::leading_dimension(b),
                info );
    }
};


// template function to call sygst
template< typename MatrixA, typename MatrixB >
inline integer_t sygst( const integer_t itype, const integer_t n,
        MatrixA& a, const MatrixB& b ) {
    typedef typename traits::matrix_traits< MatrixA >::value_type value_type;
    integer_t info(0);
    sygst_impl< value_type >::invoke( itype, n, a, b, info );
    return info;
}

} // namespace lapack
} // namespace bindings
} // namespace numeric
} // namespace boost

#endif
