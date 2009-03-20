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

#ifndef BOOST_NUMERIC_BINDINGS_LAPACK_COMPUTATIONAL_LATRD_HPP
#define BOOST_NUMERIC_BINDINGS_LAPACK_COMPUTATIONAL_LATRD_HPP

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
    inline void latrd( char const uplo, integer_t const n, integer_t const nb,
            float* a, integer_t const lda, float* e, float* tau, float* w,
            integer_t const ldw ) {
        LAPACK_SLATRD( &uplo, &n, &nb, a, &lda, e, tau, w, &ldw );
    }
    inline void latrd( char const uplo, integer_t const n, integer_t const nb,
            double* a, integer_t const lda, double* e, double* tau, double* w,
            integer_t const ldw ) {
        LAPACK_DLATRD( &uplo, &n, &nb, a, &lda, e, tau, w, &ldw );
    }
    inline void latrd( char const uplo, integer_t const n, integer_t const nb,
            traits::complex_f* a, integer_t const lda, float* e,
            traits::complex_f* tau, traits::complex_f* w,
            integer_t const ldw ) {
        LAPACK_CLATRD( &uplo, &n, &nb, traits::complex_ptr(a), &lda, e,
                traits::complex_ptr(tau), traits::complex_ptr(w), &ldw );
    }
    inline void latrd( char const uplo, integer_t const n, integer_t const nb,
            traits::complex_d* a, integer_t const lda, double* e,
            traits::complex_d* tau, traits::complex_d* w,
            integer_t const ldw ) {
        LAPACK_ZLATRD( &uplo, &n, &nb, traits::complex_ptr(a), &lda, e,
                traits::complex_ptr(tau), traits::complex_ptr(w), &ldw );
    }
}

// value-type based template
template< typename ValueType, typename Enable = void >
struct latrd_impl{};

// real specialization
template< typename ValueType >
struct latrd_impl< ValueType, typename boost::enable_if< traits::is_real<ValueType> >::type > {

    typedef ValueType value_type;
    typedef typename traits::type_traits<ValueType>::real_type real_type;

    // templated specialization
    template< typename MatrixA, typename VectorE, typename VectorTAU,
            typename MatrixW >
    static void invoke( integer_t const nb, MatrixA& a, VectorE& e,
            VectorTAU& tau, MatrixW& w ) {
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixA >::value_type, typename traits::vector_traits<
                VectorE >::value_type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixA >::value_type, typename traits::vector_traits<
                VectorTAU >::value_type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixA >::value_type, typename traits::matrix_traits<
                MatrixW >::value_type >::value) );
        BOOST_ASSERT( traits::matrix_uplo_tag(a) == 'U' ||
                traits::matrix_uplo_tag(a) == 'L' );
        BOOST_ASSERT( traits::leading_dimension(a) >= (ERROR) );
        BOOST_ASSERT( traits::leading_dimension(w) >= std::max(1,
                traits::matrix_num_columns(a)) );
        detail::latrd( traits::matrix_uplo_tag(a),
                traits::matrix_num_columns(a), nb, traits::matrix_storage(a),
                traits::leading_dimension(a), traits::vector_storage(e),
                traits::vector_storage(tau), traits::matrix_storage(w),
                traits::leading_dimension(w) );
    }
};

// complex specialization
template< typename ValueType >
struct latrd_impl< ValueType, typename boost::enable_if< traits::is_complex<ValueType> >::type > {

    typedef ValueType value_type;
    typedef typename traits::type_traits<ValueType>::real_type real_type;

    // templated specialization
    template< typename MatrixA, typename VectorE, typename VectorTAU,
            typename MatrixW >
    static void invoke( integer_t const nb, MatrixA& a, VectorE& e,
            VectorTAU& tau, MatrixW& w ) {
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixA >::value_type, typename traits::vector_traits<
                VectorTAU >::value_type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixA >::value_type, typename traits::matrix_traits<
                MatrixW >::value_type >::value) );
        BOOST_ASSERT( traits::matrix_uplo_tag(h) == 'U' ||
                traits::matrix_uplo_tag(h) == 'L' );
        BOOST_ASSERT( traits::leading_dimension(a) >= std::max(1,
                traits::matrix_num_columns(a)) );
        BOOST_ASSERT( traits::leading_dimension(w) >= std::max(1,
                traits::matrix_num_columns(a)) );
        detail::latrd( traits::matrix_uplo_tag(h),
                traits::matrix_num_columns(a), nb, traits::matrix_storage(a),
                traits::leading_dimension(a), traits::vector_storage(e),
                traits::vector_storage(tau), traits::matrix_storage(w),
                traits::leading_dimension(w) );
    }
};


// template function to call latrd
template< typename MatrixA, typename VectorE, typename VectorTAU,
        typename MatrixW >
inline integer_t latrd( integer_t const nb, MatrixA& a, VectorE& e,
        VectorTAU& tau, MatrixW& w ) {
    typedef typename traits::matrix_traits< MatrixA >::value_type value_type;
    integer_t info(0);
    latrd_impl< value_type >::invoke( nb, a, e, tau, w );
    return info;
}

}}}} // namespace boost::numeric::bindings::lapack

#endif
