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

#ifndef BOOST_NUMERIC_BINDINGS_LAPACK_COMPUTATIONAL_UNGTR_HPP
#define BOOST_NUMERIC_BINDINGS_LAPACK_COMPUTATIONAL_UNGTR_HPP

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

inline void ungtr( const char uplo, const integer_t n, traits::complex_f* a,
        const integer_t lda, const traits::complex_f* tau,
        traits::complex_f* work, const integer_t lwork, integer_t& info ) {
    LAPACK_CUNGTR( &uplo, &n, traits::complex_ptr(a), &lda,
            traits::complex_ptr(tau), traits::complex_ptr(work), &lwork,
            &info );
}
inline void ungtr( const char uplo, const integer_t n, traits::complex_d* a,
        const integer_t lda, const traits::complex_d* tau,
        traits::complex_d* work, const integer_t lwork, integer_t& info ) {
    LAPACK_ZUNGTR( &uplo, &n, traits::complex_ptr(a), &lda,
            traits::complex_ptr(tau), traits::complex_ptr(work), &lwork,
            &info );
}
} // namespace detail

// value-type based template
template< typename ValueType >
struct ungtr_impl {

    typedef ValueType value_type;
    typedef typename traits::type_traits<ValueType>::real_type real_type;

    // user-defined workspace specialization
    template< typename MatrixA, typename VectorTAU, typename WORK >
    static void invoke( const integer_t n, MatrixA& a, const VectorTAU& tau,
            integer_t& info, detail::workspace1< WORK > work ) {
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixA >::value_type, typename traits::vector_traits<
                VectorTAU >::value_type >::value) );
        BOOST_ASSERT( traits::matrix_uplo_tag(a) == 'U' ||
                traits::matrix_uplo_tag(a) == 'L' );
        BOOST_ASSERT( n >= 0 );
        BOOST_ASSERT( traits::leading_dimension(a) >= n );
        BOOST_ASSERT( traits::vector_size(tau) >= n-1 );
        BOOST_ASSERT( traits::vector_size(work.select(value_type())) >=
                min_size_work( side, m, n ));
        detail::ungtr( traits::matrix_uplo_tag(a), n,
                traits::matrix_storage(a), traits::leading_dimension(a),
                traits::vector_storage(tau),
                traits::vector_storage(work.select(value_type())),
                traits::vector_size(work.select(value_type())), info );
    }

    // minimal workspace specialization
    template< typename MatrixA, typename VectorTAU >
    static void invoke( const integer_t n, MatrixA& a, const VectorTAU& tau,
            integer_t& info, minimal_workspace work ) {
        traits::detail::array< value_type > tmp_work( min_size_work( side, m,
                n ) );
        invoke( n, a, tau, info, workspace( tmp_work ) );
    }

    // optimal workspace specialization
    template< typename MatrixA, typename VectorTAU >
    static void invoke( const integer_t n, MatrixA& a, const VectorTAU& tau,
            integer_t& info, optimal_workspace work ) {
        value_type opt_size_work;
        detail::ungtr( traits::matrix_uplo_tag(a), n,
                traits::matrix_storage(a), traits::leading_dimension(a),
                traits::vector_storage(tau), &opt_size_work, -1, info );
        traits::detail::array< value_type > tmp_work(
                traits::detail::to_int( opt_size_work ) );
        invoke( n, a, tau, info, workspace( tmp_work ) );
    }

    static integer_t min_size_work( const integer_t side, const integer_t m,
            const integer_t n ) {
        if ( side == 'L' )
            return std::max( 1, n );
        else
            return std::max( 1, m );
    }
};


// template function to call ungtr
template< typename MatrixA, typename VectorTAU, typename Workspace >
inline integer_t ungtr( const integer_t n, MatrixA& a,
        const VectorTAU& tau, Workspace work ) {
    typedef typename traits::matrix_traits< MatrixA >::value_type value_type;
    integer_t info(0);
    ungtr_impl< value_type >::invoke( n, a, tau, info, work );
    return info;
}

// template function to call ungtr, default workspace type
template< typename MatrixA, typename VectorTAU >
inline integer_t ungtr( const integer_t n, MatrixA& a,
        const VectorTAU& tau ) {
    typedef typename traits::matrix_traits< MatrixA >::value_type value_type;
    integer_t info(0);
    ungtr_impl< value_type >::invoke( n, a, tau, info,
            optimal_workspace() );
    return info;
}

} // namespace lapack
} // namespace bindings
} // namespace numeric
} // namespace boost

#endif
