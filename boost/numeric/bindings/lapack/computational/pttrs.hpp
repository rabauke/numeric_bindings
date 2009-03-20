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

#ifndef BOOST_NUMERIC_BINDINGS_LAPACK_COMPUTATIONAL_PTTRS_HPP
#define BOOST_NUMERIC_BINDINGS_LAPACK_COMPUTATIONAL_PTTRS_HPP

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
    inline void pttrs( integer_t const n, integer_t const nrhs, float* d,
            float* e, float* b, integer_t const ldb, integer_t& info ) {
        LAPACK_SPTTRS( &n, &nrhs, d, e, b, &ldb, &info );
    }
    inline void pttrs( integer_t const n, integer_t const nrhs, double* d,
            double* e, double* b, integer_t const ldb, integer_t& info ) {
        LAPACK_DPTTRS( &n, &nrhs, d, e, b, &ldb, &info );
    }
    inline void pttrs( char const uplo, integer_t const n,
            integer_t const nrhs, float* d, traits::complex_f* e,
            traits::complex_f* b, integer_t const ldb, integer_t& info ) {
        LAPACK_CPTTRS( &uplo, &n, &nrhs, d, traits::complex_ptr(e),
                traits::complex_ptr(b), &ldb, &info );
    }
    inline void pttrs( char const uplo, integer_t const n,
            integer_t const nrhs, double* d, traits::complex_d* e,
            traits::complex_d* b, integer_t const ldb, integer_t& info ) {
        LAPACK_ZPTTRS( &uplo, &n, &nrhs, d, traits::complex_ptr(e),
                traits::complex_ptr(b), &ldb, &info );
    }
}

// value-type based template
template< typename ValueType, typename Enable = void >
struct pttrs_impl{};

// real specialization
template< typename ValueType >
struct pttrs_impl< ValueType, typename boost::enable_if< traits::is_real<ValueType> >::type > {

    typedef ValueType value_type;
    typedef typename traits::type_traits<ValueType>::real_type real_type;

$INCLUDE_TEMPLATES
    // templated specialization
    template< typename VectorD, typename VectorE, typename MatrixB >
    static void invoke( integer_t const n, VectorD& d, VectorE& e, MatrixB& b,
            integer_t& info ) {
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::vector_traits<
                VectorD >::value_type, typename traits::vector_traits<
                VectorE >::value_type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::vector_traits<
                VectorD >::value_type, typename traits::matrix_traits<
                MatrixB >::value_type >::value) );
        BOOST_ASSERT( n >= 0 );
        BOOST_ASSERT( traits::matrix_num_columns(b) >= 0 );
        BOOST_ASSERT( traits::vector_size(d) >= n );
        BOOST_ASSERT( traits::vector_size(e) >= n-1 );
        BOOST_ASSERT( traits::leading_dimension(b) >= std::max(1,n) );
        detail::pttrs( n, traits::matrix_num_columns(b),
                traits::vector_storage(d), traits::vector_storage(e),
                traits::matrix_storage(b), traits::leading_dimension(b),
                info );
    }
};

// complex specialization
template< typename ValueType >
struct pttrs_impl< ValueType, typename boost::enable_if< traits::is_complex<ValueType> >::type > {

    typedef ValueType value_type;
    typedef typename traits::type_traits<ValueType>::real_type real_type;

$INCLUDE_TEMPLATES
    // templated specialization
    template< typename VectorD, typename VectorE, typename MatrixB >
    static void invoke( char const uplo, integer_t const n, VectorD& d,
            VectorE& e, MatrixB& b, integer_t& info ) {
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::vector_traits<
                VectorE >::value_type, typename traits::matrix_traits<
                MatrixB >::value_type >::value) );
        BOOST_ASSERT( uplo == 'U' || uplo == 'L' );
        BOOST_ASSERT( n >= 0 );
        BOOST_ASSERT( traits::matrix_num_columns(b) >= 0 );
        BOOST_ASSERT( traits::vector_size(d) >= n );
        BOOST_ASSERT( traits::leading_dimension(b) >= std::max(1,n) );
        detail::pttrs( uplo, n, traits::matrix_num_columns(b),
                traits::vector_storage(d), traits::vector_storage(e),
                traits::matrix_storage(b), traits::leading_dimension(b),
                info );
    }
};


// template function to call pttrs
template< typename VectorD, typename VectorE, typename MatrixB >
inline integer_t pttrs( integer_t const n, VectorD& d, VectorE& e,
        MatrixB& b ) {
    typedef typename traits::vector_traits< VectorD >::value_type value_type;
    integer_t info(0);
    pttrs_impl< value_type >::invoke( n, d, e, b, info );
    return info;
}
// template function to call pttrs
template< typename VectorD, typename VectorE, typename MatrixB >
inline integer_t pttrs( char const uplo, integer_t const n, VectorD& d,
        VectorE& e, MatrixB& b ) {
    typedef typename traits::vector_traits< VectorD >::value_type value_type;
    integer_t info(0);
    pttrs_impl< value_type >::invoke( uplo, n, d, e, b, info );
    return info;
}

}}}} // namespace boost::numeric::bindings::lapack

#endif
