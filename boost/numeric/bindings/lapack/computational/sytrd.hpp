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

#ifndef BOOST_NUMERIC_BINDINGS_LAPACK_COMPUTATIONAL_SYTRD_HPP
#define BOOST_NUMERIC_BINDINGS_LAPACK_COMPUTATIONAL_SYTRD_HPP

#include <boost/assert.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/numeric/bindings/lapack/detail/lapack.h>
#include <boost/numeric/bindings/lapack/workspace.hpp>
#include <boost/numeric/bindings/traits/detail/array.hpp>
#include <boost/numeric/bindings/traits/detail/utils.hpp>
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

inline void sytrd( const char uplo, const integer_t n, float* a,
        const integer_t lda, float* d, float* e, float* tau, float* work,
        const integer_t lwork, integer_t& info ) {
    LAPACK_SSYTRD( &uplo, &n, a, &lda, d, e, tau, work, &lwork, &info );
}
inline void sytrd( const char uplo, const integer_t n, double* a,
        const integer_t lda, double* d, double* e, double* tau, double* work,
        const integer_t lwork, integer_t& info ) {
    LAPACK_DSYTRD( &uplo, &n, a, &lda, d, e, tau, work, &lwork, &info );
}
} // namespace detail

// value-type based template
template< typename ValueType >
struct sytrd_impl {

    typedef ValueType value_type;
    typedef typename traits::type_traits<ValueType>::real_type real_type;

    // user-defined workspace specialization
    template< typename MatrixA, typename VectorD, typename VectorE,
            typename VectorTAU, typename WORK >
    static void invoke( MatrixA& a, VectorD& d, VectorE& e, VectorTAU& tau,
            integer_t& info, detail::workspace1< WORK > work ) {
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixA >::value_type, typename traits::vector_traits<
                VectorD >::value_type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixA >::value_type, typename traits::vector_traits<
                VectorE >::value_type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixA >::value_type, typename traits::vector_traits<
                VectorTAU >::value_type >::value) );
        BOOST_ASSERT( traits::matrix_uplo_tag(a) == 'U' ||
                traits::matrix_uplo_tag(a) == 'L' );
        BOOST_ASSERT( traits::matrix_num_columns(a) >= 0 );
        BOOST_ASSERT( traits::leading_dimension(a) >= std::max<
                std::ptrdiff_t >(1,traits::matrix_num_columns(a)) );
        BOOST_ASSERT( traits::vector_size(d) >=
                traits::matrix_num_columns(a) );
        BOOST_ASSERT( traits::vector_size(tau) >=
                traits::matrix_num_columns(a)-1 );
        BOOST_ASSERT( traits::vector_size(work.select(real_type())) >=
                min_size_work( $CALL_MIN_SIZE ));
        detail::sytrd( traits::matrix_uplo_tag(a),
                traits::matrix_num_columns(a), traits::matrix_storage(a),
                traits::leading_dimension(a), traits::vector_storage(d),
                traits::vector_storage(e), traits::vector_storage(tau),
                traits::vector_storage(work.select(real_type())),
                traits::vector_size(work.select(real_type())), info );
    }

    // minimal workspace specialization
    template< typename MatrixA, typename VectorD, typename VectorE,
            typename VectorTAU >
    static void invoke( MatrixA& a, VectorD& d, VectorE& e, VectorTAU& tau,
            integer_t& info, minimal_workspace work ) {
        traits::detail::array< real_type > tmp_work( min_size_work(
                $CALL_MIN_SIZE ) );
        invoke( a, d, e, tau, info, workspace( tmp_work ) );
    }

    // optimal workspace specialization
    template< typename MatrixA, typename VectorD, typename VectorE,
            typename VectorTAU >
    static void invoke( MatrixA& a, VectorD& d, VectorE& e, VectorTAU& tau,
            integer_t& info, optimal_workspace work ) {
        real_type opt_size_work;
        detail::sytrd( traits::matrix_uplo_tag(a),
                traits::matrix_num_columns(a), traits::matrix_storage(a),
                traits::leading_dimension(a), traits::vector_storage(d),
                traits::vector_storage(e), traits::vector_storage(tau),
                &opt_size_work, -1, info );
        traits::detail::array< real_type > tmp_work(
                traits::detail::to_int( opt_size_work ) );
        invoke( a, d, e, tau, info, workspace( tmp_work ) );
    }

    static integer_t min_size_work( $ARGUMENTS ) {
        $MIN_SIZE
    }
};


// template function to call sytrd
template< typename MatrixA, typename VectorD, typename VectorE,
        typename VectorTAU, typename Workspace >
inline integer_t sytrd( MatrixA& a, VectorD& d, VectorE& e,
        VectorTAU& tau, Workspace work ) {
    typedef typename traits::matrix_traits< MatrixA >::value_type value_type;
    integer_t info(0);
    sytrd_impl< value_type >::invoke( a, d, e, tau, info, work );
    return info;
}

// template function to call sytrd, default workspace type
template< typename MatrixA, typename VectorD, typename VectorE,
        typename VectorTAU >
inline integer_t sytrd( MatrixA& a, VectorD& d, VectorE& e,
        VectorTAU& tau ) {
    typedef typename traits::matrix_traits< MatrixA >::value_type value_type;
    integer_t info(0);
    sytrd_impl< value_type >::invoke( a, d, e, tau, info,
            optimal_workspace() );
    return info;
}

} // namespace lapack
} // namespace bindings
} // namespace numeric
} // namespace boost

#endif
