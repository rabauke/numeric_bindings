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

#ifndef BOOST_NUMERIC_BINDINGS_LAPACK_COMPUTATIONAL_LATRZ_HPP
#define BOOST_NUMERIC_BINDINGS_LAPACK_COMPUTATIONAL_LATRZ_HPP

#include <boost/assert.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/numeric/bindings/lapack/detail/lapack.h>
#include <boost/numeric/bindings/lapack/workspace.hpp>
#include <boost/numeric/bindings/traits/detail/array.hpp>
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
    inline void latrz( const integer_t m, const integer_t n,
            const integer_t l, float* a, const integer_t lda, float* tau,
            float* work ) {
        LAPACK_SLATRZ( &m, &n, &l, a, &lda, tau, work );
    }
    inline void latrz( const integer_t m, const integer_t n,
            const integer_t l, double* a, const integer_t lda, double* tau,
            double* work ) {
        LAPACK_DLATRZ( &m, &n, &l, a, &lda, tau, work );
    }
    inline void latrz( const integer_t m, const integer_t n,
            const integer_t l, traits::complex_f* a, const integer_t lda,
            traits::complex_f* tau, traits::complex_f* work ) {
        LAPACK_CLATRZ( &m, &n, &l, traits::complex_ptr(a), &lda,
                traits::complex_ptr(tau), traits::complex_ptr(work) );
    }
    inline void latrz( const integer_t m, const integer_t n,
            const integer_t l, traits::complex_d* a, const integer_t lda,
            traits::complex_d* tau, traits::complex_d* work ) {
        LAPACK_ZLATRZ( &m, &n, &l, traits::complex_ptr(a), &lda,
                traits::complex_ptr(tau), traits::complex_ptr(work) );
    }
}

// value-type based template
template< typename ValueType, typename Enable = void >
struct latrz_impl{};

// real specialization
template< typename ValueType >
struct latrz_impl< ValueType, typename boost::enable_if< traits::is_real<ValueType> >::type > {

    typedef ValueType value_type;
    typedef typename traits::type_traits<ValueType>::real_type real_type;

    // user-defined workspace specialization
    template< typename MatrixA, typename VectorTAU, typename WORK >
    static void invoke( MatrixA& a, VectorTAU& tau, detail::workspace1<
            WORK > work ) {
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixA >::value_type, typename traits::vector_traits<
                VectorTAU >::value_type >::value) );
        BOOST_ASSERT( traits::matrix_num_rows(a) >= 0 );
        BOOST_ASSERT( traits::matrix_num_columns(a) >= 0 );
        BOOST_ASSERT( traits::matrix_num_columns(a) >= 0 );
        BOOST_ASSERT( traits::leading_dimension(a) >= std::max(1,
                traits::matrix_num_rows(a)) );
        BOOST_ASSERT( traits::vector_size(tau) >= traits::matrix_num_rows(a) );
        BOOST_ASSERT( traits::vector_size(work.select(real_type())) >=
                min_size_work( traits::matrix_num_rows(a) ));
        detail::latrz( traits::matrix_num_rows(a),
                traits::matrix_num_columns(a), traits::matrix_num_columns(a),
                traits::matrix_storage(a), traits::leading_dimension(a),
                traits::vector_storage(tau),
                traits::vector_storage(work.select(real_type())) );
    }

    // minimal workspace specialization
    template< typename MatrixA, typename VectorTAU >
    static void invoke( MatrixA& a, VectorTAU& tau, minimal_workspace work ) {
        traits::detail::array< real_type > tmp_work( min_size_work(
                traits::matrix_num_rows(a) ) );
        invoke( a, tau, workspace( tmp_work ) );
    }

    // optimal workspace specialization
    template< typename MatrixA, typename VectorTAU >
    static void invoke( MatrixA& a, VectorTAU& tau, optimal_workspace work ) {
        invoke( a, tau, minimal_workspace() );
    }

    static integer_t min_size_work( const integer_t m ) {
        return m;
    }
};

// complex specialization
template< typename ValueType >
struct latrz_impl< ValueType, typename boost::enable_if< traits::is_complex<ValueType> >::type > {

    typedef ValueType value_type;
    typedef typename traits::type_traits<ValueType>::real_type real_type;

    // user-defined workspace specialization
    template< typename MatrixA, typename VectorTAU, typename WORK >
    static void invoke( MatrixA& a, VectorTAU& tau, detail::workspace1<
            WORK > work ) {
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixA >::value_type, typename traits::vector_traits<
                VectorTAU >::value_type >::value) );
        BOOST_ASSERT( traits::matrix_num_rows(a) >= 0 );
        BOOST_ASSERT( traits::matrix_num_columns(a) >= 0 );
        BOOST_ASSERT( traits::matrix_num_columns(a) >= 0 );
        BOOST_ASSERT( traits::leading_dimension(a) >= std::max(1,
                traits::matrix_num_rows(a)) );
        BOOST_ASSERT( traits::vector_size(tau) >= traits::matrix_num_rows(a) );
        BOOST_ASSERT( traits::vector_size(work.select(value_type())) >=
                min_size_work( traits::matrix_num_rows(a) ));
        detail::latrz( traits::matrix_num_rows(a),
                traits::matrix_num_columns(a), traits::matrix_num_columns(a),
                traits::matrix_storage(a), traits::leading_dimension(a),
                traits::vector_storage(tau),
                traits::vector_storage(work.select(value_type())) );
    }

    // minimal workspace specialization
    template< typename MatrixA, typename VectorTAU >
    static void invoke( MatrixA& a, VectorTAU& tau, minimal_workspace work ) {
        traits::detail::array< value_type > tmp_work( min_size_work(
                traits::matrix_num_rows(a) ) );
        invoke( a, tau, workspace( tmp_work ) );
    }

    // optimal workspace specialization
    template< typename MatrixA, typename VectorTAU >
    static void invoke( MatrixA& a, VectorTAU& tau, optimal_workspace work ) {
        invoke( a, tau, minimal_workspace() );
    }

    static integer_t min_size_work( const integer_t m ) {
        return m;
    }
};


// template function to call latrz
template< typename MatrixA, typename VectorTAU, typename Workspace >
inline integer_t latrz( MatrixA& a, VectorTAU& tau, Workspace work ) {
    typedef typename traits::matrix_traits< MatrixA >::value_type value_type;
    integer_t info(0);
    latrz_impl< value_type >::invoke( a, tau, work );
    return info;
}

// template function to call latrz, default workspace type
template< typename MatrixA, typename VectorTAU >
inline integer_t latrz( MatrixA& a, VectorTAU& tau ) {
    typedef typename traits::matrix_traits< MatrixA >::value_type value_type;
    integer_t info(0);
    latrz_impl< value_type >::invoke( a, tau, optimal_workspace() );
    return info;
}

}}}} // namespace boost::numeric::bindings::lapack

#endif
