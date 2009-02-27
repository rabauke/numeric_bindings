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

#ifndef BOOST_NUMERIC_BINDINGS_LAPACK_GEBAK_HPP
#define BOOST_NUMERIC_BINDINGS_LAPACK_GEBAK_HPP

#include <boost/numeric/bindings/lapack/lapack.h>
#include <boost/numeric/bindings/traits/is_complex.hpp>
#include <boost/numeric/bindings/traits/is_real.hpp>
#include <boost/numeric/bindings/traits/traits.hpp>
#include <boost/numeric/bindings/traits/type_traits.hpp>
#include <boost/static_assert.hpp
#include <boost/type_traits/is_same.hpp>
#include <boost/utility/enable_if.hpp>
#include <cassert>

namespace boost {
namespace numeric {
namespace bindings {
namespace lapack {

//$DESCRIPTION

// overloaded functions to call lapack
namespace detail {
    inline void gebak( char const job, char const side, integer_t const n,
            integer_t const ilo, integer_t const ihi, float* scale,
            integer_t const m, float* v, integer_t const ldv,
            integer_t& info ) {
        LAPACK_SGEBAK( &job, &side, &n, &ilo, &ihi, scale, &m, v, &ldv,
                &info );
    }
    inline void gebak( char const job, char const side, integer_t const n,
            integer_t const ilo, integer_t const ihi, double* scale,
            integer_t const m, double* v, integer_t const ldv,
            integer_t& info ) {
        LAPACK_DGEBAK( &job, &side, &n, &ilo, &ihi, scale, &m, v, &ldv,
                &info );
    }
    inline void gebak( char const job, char const side, integer_t const n,
            integer_t const ilo, integer_t const ihi, float* scale,
            integer_t const m, traits::complex_f* v, integer_t const ldv,
            integer_t& info ) {
        LAPACK_CGEBAK( &job, &side, &n, &ilo, &ihi, scale, &m,
                traits::complex_ptr(v), &ldv, &info );
    }
    inline void gebak( char const job, char const side, integer_t const n,
            integer_t const ilo, integer_t const ihi, double* scale,
            integer_t const m, traits::complex_d* v, integer_t const ldv,
            integer_t& info ) {
        LAPACK_ZGEBAK( &job, &side, &n, &ilo, &ihi, scale, &m,
                traits::complex_ptr(v), &ldv, &info );
    }
}

// value-type based template
template< typename ValueType, typename Enable = void >
struct gebak_impl{};

// real specialization
template< typename ValueType >
struct gebak_impl< ValueType, typename boost::enable_if< traits::is_real<ValueType> >::type > {

    typedef ValueType value_type;
    typedef typename traits::type_traits<ValueType>::real_type real_type;

    // templated specialization
    template< typename VectorSCALE, typename MatrixV >
    static void compute( char const job, char const side, integer_t const ilo,
            integer_t const ihi, VectorSCALE& scale, MatrixV& v,
            integer_t& info ) {
        BOOST_STATIC_ASSERT( boost::is_same< typename traits::vector_traits<
                VectorSCALE >::value_type, typename traits::matrix_traits<
                MatrixV >::value_type > );
#ifndef NDEBUG
        assert( job == 'N' || job == 'P' || job == 'S' || job == 'B' );
        assert( side == 'R' || side == 'L' );
        assert( traits::matrix_size1(v) >= 0 );
        assert( traits::vector_size(scale) >= traits::matrix_size1(v) );
        assert( traits::matrix_size2(v) >= 0 );
        assert( traits::leading_dimension(v) >= std::max(1,
                traits::matrix_size1(v)) );
#endif
        detail::gebak( job, side, traits::matrix_size1(v), ilo, ihi,
                traits::vector_storage(scale), traits::matrix_size2(v),
                traits::matrix_storage(v), traits::leading_dimension(v),
                info );
    }
};

// complex specialization
template< typename ValueType >
struct gebak_impl< ValueType, typename boost::enable_if< traits::is_complex<ValueType> >::type > {

    typedef ValueType value_type;
    typedef typename traits::type_traits<ValueType>::real_type real_type;

    // templated specialization
    template< typename VectorSCALE, typename MatrixV >
    static void compute( char const job, char const side, integer_t const ilo,
            integer_t const ihi, VectorSCALE& scale, MatrixV& v,
            integer_t& info ) {
        
#ifndef NDEBUG
        assert( job == 'N' || job == 'P' || job == 'S' || job == 'B' );
        assert( side == 'R' || side == 'L' );
        assert( traits::matrix_size1(v) >= 0 );
        assert( traits::vector_size(scale) >= traits::matrix_size1(v) );
        assert( traits::matrix_size2(v) >= 0 );
        assert( traits::leading_dimension(v) >= std::max(1,
                traits::matrix_size1(v)) );
#endif
        detail::gebak( job, side, traits::matrix_size1(v), ilo, ihi,
                traits::vector_storage(scale), traits::matrix_size2(v),
                traits::matrix_storage(v), traits::leading_dimension(v),
                info );
    }
};


// template function to call gebak
template< typename VectorSCALE, typename MatrixV >
inline integer_t gebak( char const job, char const side,
        integer_t const ilo, integer_t const ihi, VectorSCALE& scale,
        MatrixV& v ) {
    typedef typename traits::vector_traits<
            VectorSCALE >::value_type value_type;
    integer_t info(0);
    gebak_impl< value_type >::compute( job, side, ilo, ihi, scale, v,
            info );
    return info;
}

}}}} // namespace boost::numeric::bindings::lapack

#endif
