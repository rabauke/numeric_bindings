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

#ifndef BOOST_NUMERIC_BINDINGS_LAPACK_COMPUTATIONAL_TRSYL_HPP
#define BOOST_NUMERIC_BINDINGS_LAPACK_COMPUTATIONAL_TRSYL_HPP

#include <boost/assert.hpp>
#include <boost/numeric/bindings/lapack/detail/lapack.h>
#include <boost/numeric/bindings/traits/is_complex.hpp>
#include <boost/numeric/bindings/traits/is_real.hpp>
#include <boost/numeric/bindings/traits/traits.hpp>
#include <boost/numeric/bindings/traits/type_traits.hpp>
#include <boost/static_assert.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/utility/enable_if.hpp>

namespace boost {
namespace numeric {
namespace bindings {
namespace lapack {

//$DESCRIPTION

// overloaded functions to call lapack
namespace detail {
    inline void trsyl( char const trana, char const tranb,
            integer_t const isgn, integer_t const m, integer_t const n,
            float* a, integer_t const lda, float* b, integer_t const ldb,
            float* c, integer_t const ldc, float& scale, integer_t& info ) {
        LAPACK_STRSYL( &trana, &tranb, &isgn, &m, &n, a, &lda, b, &ldb, c,
                &ldc, &scale, &info );
    }
    inline void trsyl( char const trana, char const tranb,
            integer_t const isgn, integer_t const m, integer_t const n,
            double* a, integer_t const lda, double* b, integer_t const ldb,
            double* c, integer_t const ldc, double& scale, integer_t& info ) {
        LAPACK_DTRSYL( &trana, &tranb, &isgn, &m, &n, a, &lda, b, &ldb, c,
                &ldc, &scale, &info );
    }
    inline void trsyl( char const trana, char const tranb,
            integer_t const isgn, integer_t const m, integer_t const n,
            traits::complex_f* a, integer_t const lda, traits::complex_f* b,
            integer_t const ldb, traits::complex_f* c, integer_t const ldc,
            float& scale, integer_t& info ) {
        LAPACK_CTRSYL( &trana, &tranb, &isgn, &m, &n, traits::complex_ptr(a),
                &lda, traits::complex_ptr(b), &ldb, traits::complex_ptr(c),
                &ldc, &scale, &info );
    }
    inline void trsyl( char const trana, char const tranb,
            integer_t const isgn, integer_t const m, integer_t const n,
            traits::complex_d* a, integer_t const lda, traits::complex_d* b,
            integer_t const ldb, traits::complex_d* c, integer_t const ldc,
            double& scale, integer_t& info ) {
        LAPACK_ZTRSYL( &trana, &tranb, &isgn, &m, &n, traits::complex_ptr(a),
                &lda, traits::complex_ptr(b), &ldb, traits::complex_ptr(c),
                &ldc, &scale, &info );
    }
}

// value-type based template
template< typename ValueType, typename Enable = void >
struct trsyl_impl{};

// real specialization
template< typename ValueType >
struct trsyl_impl< ValueType, typename boost::enable_if< traits::is_real<ValueType> >::type > {

    typedef ValueType value_type;
    typedef typename traits::type_traits<ValueType>::real_type real_type;

    // templated specialization
    template< typename MatrixA, typename MatrixB, typename MatrixC >
    static void compute( char const trana, char const tranb,
            integer_t const isgn, integer_t const m, integer_t const n,
            MatrixA& a, MatrixB& b, MatrixC& c, real_type& scale,
            integer_t& info ) {
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixA >::value_type, typename traits::matrix_traits<
                MatrixB >::value_type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixA >::value_type, typename traits::matrix_traits<
                MatrixC >::value_type >::value) );
        BOOST_ASSERT( trana == 'N' || trana == 'T' || trana == 'C' );
        BOOST_ASSERT( tranb == 'N' || tranb == 'T' || tranb == 'C' );
        BOOST_ASSERT( m >= 0 );
        BOOST_ASSERT( n >= 0 );
        BOOST_ASSERT( traits::leading_dimension(a) >= std::max(1,m) );
        BOOST_ASSERT( traits::leading_dimension(b) >= std::max(1,n) );
        BOOST_ASSERT( traits::leading_dimension(c) >= std::max(1,m) );
        detail::trsyl( trana, tranb, isgn, m, n, traits::matrix_storage(a),
                traits::leading_dimension(a), traits::matrix_storage(b),
                traits::leading_dimension(b), traits::matrix_storage(c),
                traits::leading_dimension(c), scale, info );
    }
};

// complex specialization
template< typename ValueType >
struct trsyl_impl< ValueType, typename boost::enable_if< traits::is_complex<ValueType> >::type > {

    typedef ValueType value_type;
    typedef typename traits::type_traits<ValueType>::real_type real_type;

    // templated specialization
    template< typename MatrixA, typename MatrixB, typename MatrixC >
    static void compute( char const trana, char const tranb,
            integer_t const isgn, integer_t const m, integer_t const n,
            MatrixA& a, MatrixB& b, MatrixC& c, real_type& scale,
            integer_t& info ) {
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixA >::value_type, typename traits::matrix_traits<
                MatrixB >::value_type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixA >::value_type, typename traits::matrix_traits<
                MatrixC >::value_type >::value) );
        BOOST_ASSERT( trana == 'N' || trana == 'C' );
        BOOST_ASSERT( tranb == 'N' || tranb == 'C' );
        BOOST_ASSERT( m >= 0 );
        BOOST_ASSERT( n >= 0 );
        BOOST_ASSERT( traits::leading_dimension(a) >= std::max(1,m) );
        BOOST_ASSERT( traits::leading_dimension(b) >= std::max(1,n) );
        BOOST_ASSERT( traits::leading_dimension(c) >= std::max(1,m) );
        detail::trsyl( trana, tranb, isgn, m, n, traits::matrix_storage(a),
                traits::leading_dimension(a), traits::matrix_storage(b),
                traits::leading_dimension(b), traits::matrix_storage(c),
                traits::leading_dimension(c), scale, info );
    }
};


// template function to call trsyl
template< typename MatrixA, typename MatrixB, typename MatrixC >
inline integer_t trsyl( char const trana, char const tranb,
        integer_t const isgn, integer_t const m, integer_t const n,
        MatrixA& a, MatrixB& b, MatrixC& c, typename traits::matrix_traits<
        MatrixA >::value_type& scale ) {
    typedef typename traits::matrix_traits< MatrixA >::value_type value_type;
    integer_t info(0);
    trsyl_impl< value_type >::compute( trana, tranb, isgn, m, n, a, b, c,
            scale, info );
    return info;
}

}}}} // namespace boost::numeric::bindings::lapack

#endif
