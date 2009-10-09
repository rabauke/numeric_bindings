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

#ifndef BOOST_NUMERIC_BINDINGS_LAPACK_COMPUTATIONAL_GEQP3_HPP
#define BOOST_NUMERIC_BINDINGS_LAPACK_COMPUTATIONAL_GEQP3_HPP

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

inline void geqp3( const integer_t m, const integer_t n, float* a,
        const integer_t lda, integer_t* jpvt, float* tau, float* work,
        const integer_t lwork, integer_t& info ) {
    LAPACK_SGEQP3( &m, &n, a, &lda, jpvt, tau, work, &lwork, &info );
}
inline void geqp3( const integer_t m, const integer_t n, double* a,
        const integer_t lda, integer_t* jpvt, double* tau, double* work,
        const integer_t lwork, integer_t& info ) {
    LAPACK_DGEQP3( &m, &n, a, &lda, jpvt, tau, work, &lwork, &info );
}
inline void geqp3( const integer_t m, const integer_t n, traits::complex_f* a,
        const integer_t lda, integer_t* jpvt, traits::complex_f* tau,
        traits::complex_f* work, const integer_t lwork, float* rwork,
        integer_t& info ) {
    LAPACK_CGEQP3( &m, &n, traits::complex_ptr(a), &lda, jpvt,
            traits::complex_ptr(tau), traits::complex_ptr(work), &lwork,
            rwork, &info );
}
inline void geqp3( const integer_t m, const integer_t n, traits::complex_d* a,
        const integer_t lda, integer_t* jpvt, traits::complex_d* tau,
        traits::complex_d* work, const integer_t lwork, double* rwork,
        integer_t& info ) {
    LAPACK_ZGEQP3( &m, &n, traits::complex_ptr(a), &lda, jpvt,
            traits::complex_ptr(tau), traits::complex_ptr(work), &lwork,
            rwork, &info );
}
} // namespace detail

// value-type based template
template< typename ValueType, typename Enable = void >
struct geqp3_impl{};

// real specialization
template< typename ValueType >
struct geqp3_impl< ValueType, typename boost::enable_if< traits::is_real<ValueType> >::type > {

    typedef ValueType value_type;
    typedef typename traits::type_traits<ValueType>::real_type real_type;

    // user-defined workspace specialization
    template< typename MatrixA, typename VectorJPVT, typename VectorTAU,
            typename WORK >
    static void invoke( MatrixA& a, VectorJPVT& jpvt, VectorTAU& tau,
            integer_t& info, detail::workspace1< WORK > work ) {
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixA >::value_type, typename traits::vector_traits<
                VectorTAU >::value_type >::value) );
        BOOST_ASSERT( traits::matrix_num_rows(a) >= 0 );
        BOOST_ASSERT( traits::matrix_num_columns(a) >= 0 );
        BOOST_ASSERT( traits::leading_dimension(a) >= std::max<
                std::ptrdiff_t >(1,traits::matrix_num_rows(a)) );
        BOOST_ASSERT( traits::vector_size(tau) >= std::min<
                std::ptrdiff_t >(traits::matrix_num_rows(a),
                traits::matrix_num_columns(a)) );
        BOOST_ASSERT( traits::vector_size(work.select(real_type())) >=
                min_size_work( traits::matrix_num_columns(a) ));
        detail::geqp3( traits::matrix_num_rows(a),
                traits::matrix_num_columns(a), traits::matrix_storage(a),
                traits::leading_dimension(a), traits::vector_storage(jpvt),
                traits::vector_storage(tau),
                traits::vector_storage(work.select(real_type())),
                traits::vector_size(work.select(real_type())), info );
    }

    // minimal workspace specialization
    template< typename MatrixA, typename VectorJPVT, typename VectorTAU >
    static void invoke( MatrixA& a, VectorJPVT& jpvt, VectorTAU& tau,
            integer_t& info, minimal_workspace work ) {
        traits::detail::array< real_type > tmp_work( min_size_work(
                traits::matrix_num_columns(a) ) );
        invoke( a, jpvt, tau, info, workspace( tmp_work ) );
    }

    // optimal workspace specialization
    template< typename MatrixA, typename VectorJPVT, typename VectorTAU >
    static void invoke( MatrixA& a, VectorJPVT& jpvt, VectorTAU& tau,
            integer_t& info, optimal_workspace work ) {
        real_type opt_size_work;
        detail::geqp3( traits::matrix_num_rows(a),
                traits::matrix_num_columns(a), traits::matrix_storage(a),
                traits::leading_dimension(a), traits::vector_storage(jpvt),
                traits::vector_storage(tau), &opt_size_work, -1, info );
        traits::detail::array< real_type > tmp_work(
                traits::detail::to_int( opt_size_work ) );
        invoke( a, jpvt, tau, info, workspace( tmp_work ) );
    }

    static integer_t min_size_work( const integer_t n ) {
        return 3*n+1;
    }
};

// complex specialization
template< typename ValueType >
struct geqp3_impl< ValueType, typename boost::enable_if< traits::is_complex<ValueType> >::type > {

    typedef ValueType value_type;
    typedef typename traits::type_traits<ValueType>::real_type real_type;

    // user-defined workspace specialization
    template< typename MatrixA, typename VectorJPVT, typename VectorTAU,
            typename WORK, typename RWORK >
    static void invoke( MatrixA& a, VectorJPVT& jpvt, VectorTAU& tau,
            integer_t& info, detail::workspace2< WORK, RWORK > work ) {
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixA >::value_type, typename traits::vector_traits<
                VectorTAU >::value_type >::value) );
        BOOST_ASSERT( traits::matrix_num_rows(a) >= 0 );
        BOOST_ASSERT( traits::matrix_num_columns(a) >= 0 );
        BOOST_ASSERT( traits::leading_dimension(a) >= std::max<
                std::ptrdiff_t >(1,traits::matrix_num_rows(a)) );
        BOOST_ASSERT( traits::vector_size(tau) >= std::min<
                std::ptrdiff_t >(traits::matrix_num_rows(a),
                traits::matrix_num_columns(a)) );
        BOOST_ASSERT( traits::vector_size(work.select(value_type())) >=
                min_size_work( traits::matrix_num_columns(a) ));
        BOOST_ASSERT( traits::vector_size(work.select(real_type())) >=
                min_size_rwork( traits::matrix_num_columns(a) ));
        detail::geqp3( traits::matrix_num_rows(a),
                traits::matrix_num_columns(a), traits::matrix_storage(a),
                traits::leading_dimension(a), traits::vector_storage(jpvt),
                traits::vector_storage(tau),
                traits::vector_storage(work.select(value_type())),
                traits::vector_size(work.select(value_type())),
                traits::vector_storage(work.select(real_type())), info );
    }

    // minimal workspace specialization
    template< typename MatrixA, typename VectorJPVT, typename VectorTAU >
    static void invoke( MatrixA& a, VectorJPVT& jpvt, VectorTAU& tau,
            integer_t& info, minimal_workspace work ) {
        traits::detail::array< value_type > tmp_work( min_size_work(
                traits::matrix_num_columns(a) ) );
        traits::detail::array< real_type > tmp_rwork( min_size_rwork(
                traits::matrix_num_columns(a) ) );
        invoke( a, jpvt, tau, info, workspace( tmp_work, tmp_rwork ) );
    }

    // optimal workspace specialization
    template< typename MatrixA, typename VectorJPVT, typename VectorTAU >
    static void invoke( MatrixA& a, VectorJPVT& jpvt, VectorTAU& tau,
            integer_t& info, optimal_workspace work ) {
        value_type opt_size_work;
        traits::detail::array< real_type > tmp_rwork( min_size_rwork(
                traits::matrix_num_columns(a) ) );
        detail::geqp3( traits::matrix_num_rows(a),
                traits::matrix_num_columns(a), traits::matrix_storage(a),
                traits::leading_dimension(a), traits::vector_storage(jpvt),
                traits::vector_storage(tau), &opt_size_work, -1,
                traits::vector_storage(tmp_rwork), info );
        traits::detail::array< value_type > tmp_work(
                traits::detail::to_int( opt_size_work ) );
        invoke( a, jpvt, tau, info, workspace( tmp_work, tmp_rwork ) );
    }

    static integer_t min_size_work( const integer_t n ) {
        return n+1;
    }

    static integer_t min_size_rwork( const integer_t n ) {
        return 2*n;
    }
};


// template function to call geqp3
template< typename MatrixA, typename VectorJPVT, typename VectorTAU,
        typename Workspace >
inline integer_t geqp3( MatrixA& a, VectorJPVT& jpvt, VectorTAU& tau,
        Workspace work ) {
    typedef typename traits::matrix_traits< MatrixA >::value_type value_type;
    integer_t info(0);
    geqp3_impl< value_type >::invoke( a, jpvt, tau, info, work );
    return info;
}

// template function to call geqp3, default workspace type
template< typename MatrixA, typename VectorJPVT, typename VectorTAU >
inline integer_t geqp3( MatrixA& a, VectorJPVT& jpvt, VectorTAU& tau ) {
    typedef typename traits::matrix_traits< MatrixA >::value_type value_type;
    integer_t info(0);
    geqp3_impl< value_type >::invoke( a, jpvt, tau, info,
            optimal_workspace() );
    return info;
}

} // namespace lapack
} // namespace bindings
} // namespace numeric
} // namespace boost

#endif
