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

#ifndef BOOST_NUMERIC_BINDINGS_LAPACK_DRIVER_GELSY_HPP
#define BOOST_NUMERIC_BINDINGS_LAPACK_DRIVER_GELSY_HPP

#include <boost/assert.hpp>
#include <boost/mpl/bool.hpp>
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
    inline void gelsy( integer_t const m, integer_t const n,
            integer_t const nrhs, float* a, integer_t const lda, float* b,
            integer_t const ldb, integer_t* jpvt, float const rcond,
            integer_t& rank, float* work, integer_t const lwork,
            integer_t& info ) {
        LAPACK_SGELSY( &m, &n, &nrhs, a, &lda, b, &ldb, jpvt, &rcond, &rank,
                work, &lwork, &info );
    }
    inline void gelsy( integer_t const m, integer_t const n,
            integer_t const nrhs, double* a, integer_t const lda, double* b,
            integer_t const ldb, integer_t* jpvt, double const rcond,
            integer_t& rank, double* work, integer_t const lwork,
            integer_t& info ) {
        LAPACK_DGELSY( &m, &n, &nrhs, a, &lda, b, &ldb, jpvt, &rcond, &rank,
                work, &lwork, &info );
    }
    inline void gelsy( integer_t const m, integer_t const n,
            integer_t const nrhs, traits::complex_f* a, integer_t const lda,
            traits::complex_f* b, integer_t const ldb, integer_t* jpvt,
            float const rcond, integer_t& rank, traits::complex_f* work,
            integer_t const lwork, float* rwork, integer_t& info ) {
        LAPACK_CGELSY( &m, &n, &nrhs, traits::complex_ptr(a), &lda,
                traits::complex_ptr(b), &ldb, jpvt, &rcond, &rank,
                traits::complex_ptr(work), &lwork, rwork, &info );
    }
    inline void gelsy( integer_t const m, integer_t const n,
            integer_t const nrhs, traits::complex_d* a, integer_t const lda,
            traits::complex_d* b, integer_t const ldb, integer_t* jpvt,
            double const rcond, integer_t& rank, traits::complex_d* work,
            integer_t const lwork, double* rwork, integer_t& info ) {
        LAPACK_ZGELSY( &m, &n, &nrhs, traits::complex_ptr(a), &lda,
                traits::complex_ptr(b), &ldb, jpvt, &rcond, &rank,
                traits::complex_ptr(work), &lwork, rwork, &info );
    }
}

// value-type based template
template< typename ValueType, typename Enable = void >
struct gelsy_impl{};

// real specialization
template< typename ValueType >
struct gelsy_impl< ValueType, typename boost::enable_if< traits::is_real<ValueType> >::type > {

    typedef ValueType value_type;
    typedef typename traits::type_traits<ValueType>::real_type real_type;

    // user-defined workspace specialization
    template< typename MatrixA, typename MatrixB, typename VectorJPVT,
            typename WORK >
    static void invoke( MatrixA& a, MatrixB& b, VectorJPVT& jpvt,
            real_type const rcond, integer_t& rank, integer_t& info,
            detail::workspace1< WORK > work ) {
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixA >::value_type, typename traits::matrix_traits<
                MatrixB >::value_type >::value) );
        BOOST_ASSERT( traits::matrix_num_rows(a) >= 0 );
        BOOST_ASSERT( traits::matrix_num_columns(a) >= 0 );
        BOOST_ASSERT( traits::matrix_num_columns(b) >= 0 );
        BOOST_ASSERT( traits::leading_dimension(a) >= std::max(1,
                traits::matrix_num_rows(a)) );
        BOOST_ASSERT( traits::leading_dimension(b) >= std::max(1,
                std::max(traits::matrix_num_rows(a),
                traits::matrix_num_columns(a))) );
        BOOST_ASSERT( traits::vector_size(work.select(real_type())) >=
                min_size_work( traits::matrix_num_rows(a),
                traits::matrix_num_columns(a),
                traits::matrix_num_columns(b) ));
        detail::gelsy( traits::matrix_num_rows(a),
                traits::matrix_num_columns(a), traits::matrix_num_columns(b),
                traits::matrix_storage(a), traits::leading_dimension(a),
                traits::matrix_storage(b), traits::leading_dimension(b),
                traits::vector_storage(jpvt), rcond, rank,
                traits::vector_storage(work.select(real_type())),
                traits::vector_size(work.select(real_type())), info );
    }

    // minimal workspace specialization
    template< typename MatrixA, typename MatrixB, typename VectorJPVT >
    static void invoke( MatrixA& a, MatrixB& b, VectorJPVT& jpvt,
            real_type const rcond, integer_t& rank, integer_t& info,
            minimal_workspace work ) {
        traits::detail::array< real_type > tmp_work( min_size_work(
                traits::matrix_num_rows(a), traits::matrix_num_columns(a),
                traits::matrix_num_columns(b) ) );
        invoke( a, b, jpvt, rcond, rank, info, workspace( tmp_work ) );
    }

    // optimal workspace specialization
    template< typename MatrixA, typename MatrixB, typename VectorJPVT >
    static void invoke( MatrixA& a, MatrixB& b, VectorJPVT& jpvt,
            real_type const rcond, integer_t& rank, integer_t& info,
            optimal_workspace work ) {
        real_type opt_size_work;
        detail::gelsy( traits::matrix_num_rows(a),
                traits::matrix_num_columns(a), traits::matrix_num_columns(b),
                traits::matrix_storage(a), traits::leading_dimension(a),
                traits::matrix_storage(b), traits::leading_dimension(b),
                traits::vector_storage(jpvt), rcond, rank, &opt_size_work, -1,
                info );
        traits::detail::array< real_type > tmp_work(
                traits::detail::to_int( opt_size_work ) );
        invoke( a, b, jpvt, rcond, rank, info, workspace( tmp_work ) );
    }

    static integer_t min_size_work( integer_t const m, integer_t const n,
            integer_t const nrhs ) {
        integer_t minmn = std::min( m, n );
        return std::max( 1, std::max( minmn+3*n+1, 2*minmn+nrhs ));
    }
};

// complex specialization
template< typename ValueType >
struct gelsy_impl< ValueType, typename boost::enable_if< traits::is_complex<ValueType> >::type > {

    typedef ValueType value_type;
    typedef typename traits::type_traits<ValueType>::real_type real_type;

    // user-defined workspace specialization
    template< typename MatrixA, typename MatrixB, typename VectorJPVT,
            typename WORK, typename RWORK >
    static void invoke( MatrixA& a, MatrixB& b, VectorJPVT& jpvt,
            real_type const rcond, integer_t& rank, integer_t& info,
            detail::workspace2< WORK, RWORK > work ) {
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixA >::value_type, typename traits::matrix_traits<
                MatrixB >::value_type >::value) );
        BOOST_ASSERT( traits::matrix_num_rows(a) >= 0 );
        BOOST_ASSERT( traits::matrix_num_columns(a) >= 0 );
        BOOST_ASSERT( traits::matrix_num_columns(b) >= 0 );
        BOOST_ASSERT( traits::leading_dimension(a) >= std::max(1,
                traits::matrix_num_rows(a)) );
        BOOST_ASSERT( traits::leading_dimension(b) >= std::max(1,
                std::max(traits::matrix_num_rows(a),
                traits::matrix_num_columns(a))) );
        BOOST_ASSERT( traits::vector_size(work.select(value_type())) >=
                min_size_work( traits::matrix_num_rows(a),
                traits::matrix_num_columns(a),
                traits::matrix_num_columns(b) ));
        BOOST_ASSERT( traits::vector_size(work.select(real_type())) >=
                min_size_rwork( traits::matrix_num_columns(a) ));
        detail::gelsy( traits::matrix_num_rows(a),
                traits::matrix_num_columns(a), traits::matrix_num_columns(b),
                traits::matrix_storage(a), traits::leading_dimension(a),
                traits::matrix_storage(b), traits::leading_dimension(b),
                traits::vector_storage(jpvt), rcond, rank,
                traits::vector_storage(work.select(value_type())),
                traits::vector_size(work.select(value_type())),
                traits::vector_storage(work.select(real_type())), info );
    }

    // minimal workspace specialization
    template< typename MatrixA, typename MatrixB, typename VectorJPVT >
    static void invoke( MatrixA& a, MatrixB& b, VectorJPVT& jpvt,
            real_type const rcond, integer_t& rank, integer_t& info,
            minimal_workspace work ) {
        traits::detail::array< value_type > tmp_work( min_size_work(
                traits::matrix_num_rows(a), traits::matrix_num_columns(a),
                traits::matrix_num_columns(b) ) );
        traits::detail::array< real_type > tmp_rwork( min_size_rwork(
                traits::matrix_num_columns(a) ) );
        invoke( a, b, jpvt, rcond, rank, info, workspace( tmp_work,
                tmp_rwork ) );
    }

    // optimal workspace specialization
    template< typename MatrixA, typename MatrixB, typename VectorJPVT >
    static void invoke( MatrixA& a, MatrixB& b, VectorJPVT& jpvt,
            real_type const rcond, integer_t& rank, integer_t& info,
            optimal_workspace work ) {
        value_type opt_size_work;
        traits::detail::array< real_type > tmp_rwork( min_size_rwork(
                traits::matrix_num_columns(a) ) );
        detail::gelsy( traits::matrix_num_rows(a),
                traits::matrix_num_columns(a), traits::matrix_num_columns(b),
                traits::matrix_storage(a), traits::leading_dimension(a),
                traits::matrix_storage(b), traits::leading_dimension(b),
                traits::vector_storage(jpvt), rcond, rank, &opt_size_work, -1,
                traits::vector_storage(tmp_rwork), info );
        traits::detail::array< value_type > tmp_work(
                traits::detail::to_int( opt_size_work ) );
        invoke( a, b, jpvt, rcond, rank, info, workspace( tmp_work,
                tmp_rwork ) );
    }

    static integer_t min_size_work( integer_t const m, integer_t const n,
            integer_t const nrhs ) {
        integer_t minmn = std::min( m, n );
        return std::max( 1, std::max( std::max( 2*minmn, n+1 ), minmn+nrhs ) );
    }

    static integer_t min_size_rwork( integer_t const n ) {
        return 2*n;
    }
};


// template function to call gelsy
template< typename MatrixA, typename MatrixB, typename VectorJPVT,
        typename Workspace >
inline integer_t gelsy( MatrixA& a, MatrixB& b, VectorJPVT& jpvt,
        typename traits::type_traits< typename traits::matrix_traits<
        MatrixA >::value_type >::real_type const rcond, integer_t& rank,
        Workspace work ) {
    typedef typename traits::matrix_traits< MatrixA >::value_type value_type;
    integer_t info(0);
    gelsy_impl< value_type >::invoke( a, b, jpvt, rcond, rank, info,
            work );
    return info;
}

// template function to call gelsy, default workspace type
template< typename MatrixA, typename MatrixB, typename VectorJPVT >
inline integer_t gelsy( MatrixA& a, MatrixB& b, VectorJPVT& jpvt,
        typename traits::type_traits< typename traits::matrix_traits<
        MatrixA >::value_type >::real_type const rcond, integer_t& rank ) {
    typedef typename traits::matrix_traits< MatrixA >::value_type value_type;
    integer_t info(0);
    gelsy_impl< value_type >::invoke( a, b, jpvt, rcond, rank, info,
            optimal_workspace() );
    return info;
}

}}}} // namespace boost::numeric::bindings::lapack

#endif
