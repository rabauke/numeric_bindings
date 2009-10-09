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

#ifndef BOOST_NUMERIC_BINDINGS_LAPACK_COMPUTATIONAL_GGHRD_HPP
#define BOOST_NUMERIC_BINDINGS_LAPACK_COMPUTATIONAL_GGHRD_HPP

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

inline void gghrd( const char compq, const char compz, const integer_t n,
        const integer_t ilo, const integer_t ihi, float* a,
        const integer_t lda, float* b, const integer_t ldb, float* q,
        const integer_t ldq, float* z, const integer_t ldz, integer_t& info ) {
    LAPACK_SGGHRD( &compq, &compz, &n, &ilo, &ihi, a, &lda, b, &ldb, q, &ldq,
            z, &ldz, &info );
}
inline void gghrd( const char compq, const char compz, const integer_t n,
        const integer_t ilo, const integer_t ihi, double* a,
        const integer_t lda, double* b, const integer_t ldb, double* q,
        const integer_t ldq, double* z, const integer_t ldz,
        integer_t& info ) {
    LAPACK_DGGHRD( &compq, &compz, &n, &ilo, &ihi, a, &lda, b, &ldb, q, &ldq,
            z, &ldz, &info );
}
inline void gghrd( const char compq, const char compz, const integer_t n,
        const integer_t ilo, const integer_t ihi, traits::complex_f* a,
        const integer_t lda, traits::complex_f* b, const integer_t ldb,
        traits::complex_f* q, const integer_t ldq, traits::complex_f* z,
        const integer_t ldz, integer_t& info ) {
    LAPACK_CGGHRD( &compq, &compz, &n, &ilo, &ihi, traits::complex_ptr(a),
            &lda, traits::complex_ptr(b), &ldb, traits::complex_ptr(q), &ldq,
            traits::complex_ptr(z), &ldz, &info );
}
inline void gghrd( const char compq, const char compz, const integer_t n,
        const integer_t ilo, const integer_t ihi, traits::complex_d* a,
        const integer_t lda, traits::complex_d* b, const integer_t ldb,
        traits::complex_d* q, const integer_t ldq, traits::complex_d* z,
        const integer_t ldz, integer_t& info ) {
    LAPACK_ZGGHRD( &compq, &compz, &n, &ilo, &ihi, traits::complex_ptr(a),
            &lda, traits::complex_ptr(b), &ldb, traits::complex_ptr(q), &ldq,
            traits::complex_ptr(z), &ldz, &info );
}
} // namespace detail

// value-type based template
template< typename ValueType >
struct gghrd_impl {

    typedef ValueType value_type;
    typedef typename traits::type_traits<ValueType>::real_type real_type;

    // templated specialization
    template< typename MatrixA, typename MatrixB, typename MatrixQ,
            typename MatrixZ >
    static void invoke( const char compq, const char compz, const integer_t n,
            const integer_t ilo, MatrixA& a, MatrixB& b, MatrixQ& q,
            MatrixZ& z, integer_t& info ) {
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixA >::value_type, typename traits::matrix_traits<
                MatrixB >::value_type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixA >::value_type, typename traits::matrix_traits<
                MatrixQ >::value_type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixA >::value_type, typename traits::matrix_traits<
                MatrixZ >::value_type >::value) );
        BOOST_ASSERT( compq == 'N' || compq == 'I' || compq == 'V' );
        BOOST_ASSERT( compz == 'N' || compz == 'I' || compz == 'V' );
        BOOST_ASSERT( n >= 0 );
        BOOST_ASSERT( traits::leading_dimension(a) >= std::max<
                std::ptrdiff_t >(1,n) );
        BOOST_ASSERT( traits::leading_dimension(b) >= std::max<
                std::ptrdiff_t >(1,n) );
        detail::gghrd( compq, compz, n, ilo, traits::matrix_num_columns(a),
                traits::matrix_storage(a), traits::leading_dimension(a),
                traits::matrix_storage(b), traits::leading_dimension(b),
                traits::matrix_storage(q), traits::leading_dimension(q),
                traits::matrix_storage(z), traits::leading_dimension(z),
                info );
    }
};


// template function to call gghrd
template< typename MatrixA, typename MatrixB, typename MatrixQ,
        typename MatrixZ >
inline integer_t gghrd( const char compq, const char compz,
        const integer_t n, const integer_t ilo, MatrixA& a, MatrixB& b,
        MatrixQ& q, MatrixZ& z ) {
    typedef typename traits::matrix_traits< MatrixA >::value_type value_type;
    integer_t info(0);
    gghrd_impl< value_type >::invoke( compq, compz, n, ilo, a, b, q, z,
            info );
    return info;
}

} // namespace lapack
} // namespace bindings
} // namespace numeric
} // namespace boost

#endif
