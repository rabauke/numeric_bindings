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

#ifndef BOOST_NUMERIC_BINDINGS_LAPACK_DRIVER_PBSV_HPP
#define BOOST_NUMERIC_BINDINGS_LAPACK_DRIVER_PBSV_HPP

#include <boost/assert.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/numeric/bindings/lapack/detail/lapack.h>
#include <boost/numeric/bindings/lapack/keywords.hpp>
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
    inline void pbsv( char const uplo, integer_t const n, integer_t const kd,
            integer_t const nrhs, float* ab, integer_t const ldab, float* b,
            integer_t const ldb, integer_t& info ) {
        LAPACK_SPBSV( &uplo, &n, &kd, &nrhs, ab, &ldab, b, &ldb, &info );
    }
    inline void pbsv( char const uplo, integer_t const n, integer_t const kd,
            integer_t const nrhs, double* ab, integer_t const ldab, double* b,
            integer_t const ldb, integer_t& info ) {
        LAPACK_DPBSV( &uplo, &n, &kd, &nrhs, ab, &ldab, b, &ldb, &info );
    }
    inline void pbsv( char const uplo, integer_t const n, integer_t const kd,
            integer_t const nrhs, traits::complex_f* ab, integer_t const ldab,
            traits::complex_f* b, integer_t const ldb, integer_t& info ) {
        LAPACK_CPBSV( &uplo, &n, &kd, &nrhs, traits::complex_ptr(ab), &ldab,
                traits::complex_ptr(b), &ldb, &info );
    }
    inline void pbsv( char const uplo, integer_t const n, integer_t const kd,
            integer_t const nrhs, traits::complex_d* ab, integer_t const ldab,
            traits::complex_d* b, integer_t const ldb, integer_t& info ) {
        LAPACK_ZPBSV( &uplo, &n, &kd, &nrhs, traits::complex_ptr(ab), &ldab,
                traits::complex_ptr(b), &ldb, &info );
    }
}

// value-type based template
template< typename ValueType >
struct pbsv_impl {

    typedef ValueType value_type;
    typedef typename traits::type_traits<ValueType>::real_type real_type;
    typedef typename mpl::vector< keywords::tag::A,
            keywords::tag::B > valid_keywords;

    // templated specialization
    template< typename MatrixAB, typename MatrixB >
    static void compute( integer_t const kd, MatrixAB& ab, MatrixB& b,
            integer_t& info ) {
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixAB >::value_type, typename traits::matrix_traits<
                MatrixB >::value_type >::value) );
        BOOST_ASSERT( traits::matrix_uplo_tag(ab) == 'U' ||
                traits::matrix_uplo_tag(ab) == 'L' );
        BOOST_ASSERT( traits::matrix_num_columns(ab) >= 0 );
        BOOST_ASSERT( kd >= 0 );
        BOOST_ASSERT( traits::matrix_num_columns(b) >= 0 );
        BOOST_ASSERT( traits::leading_dimension(ab) >= kd+1 );
        BOOST_ASSERT( traits::leading_dimension(b) >= std::max(1,
                traits::matrix_num_columns(ab)) );
        detail::pbsv( traits::matrix_uplo_tag(ab),
                traits::matrix_num_columns(ab), kd,
                traits::matrix_num_columns(b), traits::matrix_storage(ab),
                traits::leading_dimension(ab), traits::matrix_storage(b),
                traits::leading_dimension(b), info );
    }
};


// template function to call pbsv
template< typename MatrixAB, typename MatrixB >
inline integer_t pbsv( integer_t const kd, MatrixAB& ab, MatrixB& b ) {
    typedef typename traits::matrix_traits< MatrixAB >::value_type value_type;
    integer_t info(0);
    pbsv_impl< value_type >::compute( kd, ab, b, info );
    return info;
}

}}}} // namespace boost::numeric::bindings::lapack

#endif
