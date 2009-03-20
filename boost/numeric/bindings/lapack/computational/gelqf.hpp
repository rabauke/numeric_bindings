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

#ifndef BOOST_NUMERIC_BINDINGS_LAPACK_COMPUTATIONAL_GELQF_HPP
#define BOOST_NUMERIC_BINDINGS_LAPACK_COMPUTATIONAL_GELQF_HPP

#include <boost/assert.hpp>
#include <boost/numeric/bindings/lapack/detail/lapack.h>
#include <boost/numeric/bindings/lapack/workspace.hpp>
#include <boost/numeric/bindings/traits/detail/array.hpp>
#include <boost/numeric/bindings/traits/detail/utils.hpp>
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
    inline void gelqf( integer_t const m, integer_t const n, float* a,
            integer_t const lda, float* tau, float* work,
            integer_t const lwork, integer_t& info ) {
        LAPACK_SGELQF( &m, &n, a, &lda, tau, work, &lwork, &info );
    }
    inline void gelqf( integer_t const m, integer_t const n, double* a,
            integer_t const lda, double* tau, double* work,
            integer_t const lwork, integer_t& info ) {
        LAPACK_DGELQF( &m, &n, a, &lda, tau, work, &lwork, &info );
    }
    inline void gelqf( integer_t const m, integer_t const n,
            traits::complex_f* a, integer_t const lda, traits::complex_f* tau,
            traits::complex_f* work, integer_t const lwork, integer_t& info ) {
        LAPACK_CGELQF( &m, &n, traits::complex_ptr(a), &lda,
                traits::complex_ptr(tau), traits::complex_ptr(work), &lwork,
                &info );
    }
    inline void gelqf( integer_t const m, integer_t const n,
            traits::complex_d* a, integer_t const lda, traits::complex_d* tau,
            traits::complex_d* work, integer_t const lwork, integer_t& info ) {
        LAPACK_ZGELQF( &m, &n, traits::complex_ptr(a), &lda,
                traits::complex_ptr(tau), traits::complex_ptr(work), &lwork,
                &info );
    }
}

// value-type based template
template< typename ValueType, typename Enable = void >
struct gelqf_impl{};

// real specialization
template< typename ValueType >
struct gelqf_impl< ValueType, typename boost::enable_if< traits::is_real<ValueType> >::type > {

    typedef ValueType value_type;
    typedef typename traits::type_traits<ValueType>::real_type real_type;

    // user-defined workspace specialization
    template< typename MatrixA, typename VectorTAU, typename WORK >
    static void invoke( MatrixA& a, VectorTAU& tau, integer_t& info,
            detail::workspace1< WORK > work ) {
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixA >::value_type, typename traits::vector_traits<
                VectorTAU >::value_type >::value) );
        BOOST_ASSERT( traits::matrix_num_rows(a) >= 0 );
        BOOST_ASSERT( traits::matrix_num_columns(a) >= 0 );
        BOOST_ASSERT( traits::leading_dimension(a) >= std::max(1,
                traits::matrix_num_rows(a)) );
        BOOST_ASSERT( traits::vector_size(tau) >=
                std::min(traits::matrix_num_rows(a),
                traits::matrix_num_columns(a)) );
        BOOST_ASSERT( traits::vector_size(work.select(real_type())) >=
                min_size_work( traits::matrix_num_rows(a) ));
        detail::gelqf( traits::matrix_num_rows(a),
                traits::matrix_num_columns(a), traits::matrix_storage(a),
                traits::leading_dimension(a), traits::vector_storage(tau),
                traits::vector_storage(work.select(real_type())),
                traits::vector_size(work.select(real_type())), info );
    }

    // minimal workspace specialization
    template< typename MatrixA, typename VectorTAU >
    static void invoke( MatrixA& a, VectorTAU& tau, integer_t& info,
            minimal_workspace work ) {
        traits::detail::array< real_type > tmp_work( min_size_work(
                traits::matrix_num_rows(a) ) );
        invoke( a, tau, info, workspace( tmp_work ) );
    }

    // optimal workspace specialization
    template< typename MatrixA, typename VectorTAU >
    static void invoke( MatrixA& a, VectorTAU& tau, integer_t& info,
            optimal_workspace work ) {
        real_type opt_size_work;
        detail::gelqf( traits::matrix_num_rows(a),
                traits::matrix_num_columns(a), traits::matrix_storage(a),
                traits::leading_dimension(a), traits::vector_storage(tau),
                &opt_size_work, -1, info );
        traits::detail::array< real_type > tmp_work(
                traits::detail::to_int( opt_size_work ) );
        invoke( a, tau, info, workspace( tmp_work ) );
    }

    static integer_t min_size_work( integer_t const m ) {
        return std::max( 1, m );
    }
};

// complex specialization
template< typename ValueType >
struct gelqf_impl< ValueType, typename boost::enable_if< traits::is_complex<ValueType> >::type > {

    typedef ValueType value_type;
    typedef typename traits::type_traits<ValueType>::real_type real_type;

    // user-defined workspace specialization
    template< typename MatrixA, typename VectorTAU, typename WORK >
    static void invoke( MatrixA& a, VectorTAU& tau, integer_t& info,
            detail::workspace1< WORK > work ) {
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixA >::value_type, typename traits::vector_traits<
                VectorTAU >::value_type >::value) );
        BOOST_ASSERT( traits::matrix_num_rows(a) >= 0 );
        BOOST_ASSERT( traits::matrix_num_columns(a) >= 0 );
        BOOST_ASSERT( traits::leading_dimension(a) >= std::max(1,
                traits::matrix_num_rows(a)) );
        BOOST_ASSERT( traits::vector_size(tau) >=
                std::min(traits::matrix_num_rows(a),
                traits::matrix_num_columns(a)) );
        BOOST_ASSERT( traits::vector_size(work.select(value_type())) >=
                min_size_work( traits::matrix_num_rows(a) ));
        detail::gelqf( traits::matrix_num_rows(a),
                traits::matrix_num_columns(a), traits::matrix_storage(a),
                traits::leading_dimension(a), traits::vector_storage(tau),
                traits::vector_storage(work.select(value_type())),
                traits::vector_size(work.select(value_type())), info );
    }

    // minimal workspace specialization
    template< typename MatrixA, typename VectorTAU >
    static void invoke( MatrixA& a, VectorTAU& tau, integer_t& info,
            minimal_workspace work ) {
        traits::detail::array< value_type > tmp_work( min_size_work(
                traits::matrix_num_rows(a) ) );
        invoke( a, tau, info, workspace( tmp_work ) );
    }

    // optimal workspace specialization
    template< typename MatrixA, typename VectorTAU >
    static void invoke( MatrixA& a, VectorTAU& tau, integer_t& info,
            optimal_workspace work ) {
        value_type opt_size_work;
        detail::gelqf( traits::matrix_num_rows(a),
                traits::matrix_num_columns(a), traits::matrix_storage(a),
                traits::leading_dimension(a), traits::vector_storage(tau),
                &opt_size_work, -1, info );
        traits::detail::array< value_type > tmp_work(
                traits::detail::to_int( opt_size_work ) );
        invoke( a, tau, info, workspace( tmp_work ) );
    }

    static integer_t min_size_work( integer_t const m ) {
        return std::max( 1, m );
    }
};


// template function to call gelqf
template< typename MatrixA, typename VectorTAU, typename Workspace >
inline integer_t gelqf( MatrixA& a, VectorTAU& tau, Workspace work ) {
    typedef typename traits::matrix_traits< MatrixA >::value_type value_type;
    integer_t info(0);
    gelqf_impl< value_type >::invoke( a, tau, info, work );
    return info;
}

// template function to call gelqf, default workspace type
template< typename MatrixA, typename VectorTAU >
inline integer_t gelqf( MatrixA& a, VectorTAU& tau ) {
    typedef typename traits::matrix_traits< MatrixA >::value_type value_type;
    integer_t info(0);
    gelqf_impl< value_type >::invoke( a, tau, info, optimal_workspace() );
    return info;
}

}}}} // namespace boost::numeric::bindings::lapack

#endif
