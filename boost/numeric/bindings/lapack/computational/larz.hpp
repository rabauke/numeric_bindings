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

#ifndef BOOST_NUMERIC_BINDINGS_LAPACK_COMPUTATIONAL_LARZ_HPP
#define BOOST_NUMERIC_BINDINGS_LAPACK_COMPUTATIONAL_LARZ_HPP

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

inline void larz( const char side, const integer_t m, const integer_t n,
        const integer_t l, const float* v, const integer_t incv,
        const float tau, float* c, const integer_t ldc, float* work ) {
    LAPACK_SLARZ( &side, &m, &n, &l, v, &incv, &tau, c, &ldc, work );
}

inline void larz( const char side, const integer_t m, const integer_t n,
        const integer_t l, const double* v, const integer_t incv,
        const double tau, double* c, const integer_t ldc, double* work ) {
    LAPACK_DLARZ( &side, &m, &n, &l, v, &incv, &tau, c, &ldc, work );
}

inline void larz( const char side, const integer_t m, const integer_t n,
        const integer_t l, const traits::complex_f* v, const integer_t incv,
        const traits::complex_f tau, traits::complex_f* c,
        const integer_t ldc, traits::complex_f* work ) {
    LAPACK_CLARZ( &side, &m, &n, &l, traits::complex_ptr(v), &incv,
            traits::complex_ptr(&tau), traits::complex_ptr(c), &ldc,
            traits::complex_ptr(work) );
}

inline void larz( const char side, const integer_t m, const integer_t n,
        const integer_t l, const traits::complex_d* v, const integer_t incv,
        const traits::complex_d tau, traits::complex_d* c,
        const integer_t ldc, traits::complex_d* work ) {
    LAPACK_ZLARZ( &side, &m, &n, &l, traits::complex_ptr(v), &incv,
            traits::complex_ptr(&tau), traits::complex_ptr(c), &ldc,
            traits::complex_ptr(work) );
}

} // namespace detail

// value-type based template
template< typename ValueType, typename Enable = void >
struct larz_impl{};

// real specialization
template< typename ValueType >
struct larz_impl< ValueType, typename boost::enable_if< traits::is_real<ValueType> >::type > {

    typedef ValueType value_type;
    typedef typename traits::type_traits<ValueType>::real_type real_type;

    // user-defined workspace specialization
    template< typename VectorV, typename MatrixC, typename WORK >
    static void invoke( const char side, const integer_t l, const VectorV& v,
            const integer_t incv, const real_type tau, MatrixC& c,
            detail::workspace1< WORK > work ) {
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::vector_traits<
                VectorV >::value_type, typename traits::matrix_traits<
                MatrixC >::value_type >::value) );
        BOOST_ASSERT( side == 'L' || side == 'R' );
        BOOST_ASSERT( traits::leading_dimension(c) >= std::max(1,
                traits::matrix_num_rows(c)) );
        BOOST_ASSERT( traits::vector_size(work.select(real_type())) >=
                min_size_work( side, traits::matrix_num_rows(c),
                traits::matrix_num_columns(c) ));
        detail::larz( side, traits::matrix_num_rows(c),
                traits::matrix_num_columns(c), l, traits::vector_storage(v),
                incv, tau, traits::matrix_storage(c),
                traits::leading_dimension(c),
                traits::vector_storage(work.select(real_type())) );
    }

    // minimal workspace specialization
    template< typename VectorV, typename MatrixC >
    static void invoke( const char side, const integer_t l, const VectorV& v,
            const integer_t incv, const real_type tau, MatrixC& c,
            minimal_workspace work ) {
        traits::detail::array< real_type > tmp_work( min_size_work( side,
                traits::matrix_num_rows(c), traits::matrix_num_columns(c) ) );
        invoke( side, l, v, incv, tau, c, workspace( tmp_work ) );
    }

    // optimal workspace specialization
    template< typename VectorV, typename MatrixC >
    static void invoke( const char side, const integer_t l, const VectorV& v,
            const integer_t incv, const real_type tau, MatrixC& c,
            optimal_workspace work ) {
        invoke( side, l, v, incv, tau, c, minimal_workspace() );
    }

    static integer_t min_size_work( const char side, const integer_t m,
            const integer_t n ) {
        if ( side == 'L' )
            return n;
        else
            return m;
    }
};

// complex specialization
template< typename ValueType >
struct larz_impl< ValueType, typename boost::enable_if< traits::is_complex<ValueType> >::type > {

    typedef ValueType value_type;
    typedef typename traits::type_traits<ValueType>::real_type real_type;

    // user-defined workspace specialization
    template< typename VectorV, typename MatrixC, typename WORK >
    static void invoke( const char side, const integer_t l, const VectorV& v,
            const integer_t incv, const value_type tau, MatrixC& c,
            detail::workspace1< WORK > work ) {
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::vector_traits<
                VectorV >::value_type, typename traits::matrix_traits<
                MatrixC >::value_type >::value) );
        BOOST_ASSERT( side == 'L' || side == 'R' );
        BOOST_ASSERT( traits::leading_dimension(c) >= std::max(1,
                traits::matrix_num_rows(c)) );
        BOOST_ASSERT( traits::vector_size(work.select(value_type())) >=
                min_size_work( side, traits::matrix_num_rows(c),
                traits::matrix_num_columns(c) ));
        detail::larz( side, traits::matrix_num_rows(c),
                traits::matrix_num_columns(c), l, traits::vector_storage(v),
                incv, tau, traits::matrix_storage(c),
                traits::leading_dimension(c),
                traits::vector_storage(work.select(value_type())) );
    }

    // minimal workspace specialization
    template< typename VectorV, typename MatrixC >
    static void invoke( const char side, const integer_t l, const VectorV& v,
            const integer_t incv, const value_type tau, MatrixC& c,
            minimal_workspace work ) {
        traits::detail::array< value_type > tmp_work( min_size_work( side,
                traits::matrix_num_rows(c), traits::matrix_num_columns(c) ) );
        invoke( side, l, v, incv, tau, c, workspace( tmp_work ) );
    }

    // optimal workspace specialization
    template< typename VectorV, typename MatrixC >
    static void invoke( const char side, const integer_t l, const VectorV& v,
            const integer_t incv, const value_type tau, MatrixC& c,
            optimal_workspace work ) {
        invoke( side, l, v, incv, tau, c, minimal_workspace() );
    }

    static integer_t min_size_work( const char side, const integer_t m,
            const integer_t n ) {
        if ( side == 'L' )
            return n;
        else
            return m;
    }
};


// template function to call larz
template< typename VectorV, typename MatrixC, typename Workspace >
inline integer_t larz( const char side, const integer_t l,
        const VectorV& v, const integer_t incv,
        const typename traits::type_traits< typename traits::vector_traits<
        VectorV >::value_type >::real_type tau, MatrixC& c, Workspace work ) {
    typedef typename traits::vector_traits< VectorV >::value_type value_type;
    integer_t info(0);
    larz_impl< value_type >::invoke( side, l, v, incv, tau, c, work );
    return info;
}

// template function to call larz, default workspace type
template< typename VectorV, typename MatrixC >
inline integer_t larz( const char side, const integer_t l,
        const VectorV& v, const integer_t incv,
        const typename traits::type_traits< typename traits::vector_traits<
        VectorV >::value_type >::real_type tau, MatrixC& c ) {
    typedef typename traits::vector_traits< VectorV >::value_type value_type;
    integer_t info(0);
    larz_impl< value_type >::invoke( side, l, v, incv, tau, c,
            optimal_workspace() );
    return info;
}
// template function to call larz
template< typename VectorV, typename MatrixC, typename Workspace >
inline integer_t larz( const char side, const integer_t l,
        const VectorV& v, const integer_t incv,
        const typename traits::vector_traits< VectorV >::value_type tau,
        MatrixC& c, Workspace work ) {
    typedef typename traits::vector_traits< VectorV >::value_type value_type;
    integer_t info(0);
    larz_impl< value_type >::invoke( side, l, v, incv, tau, c, work );
    return info;
}

// template function to call larz, default workspace type
template< typename VectorV, typename MatrixC >
inline integer_t larz( const char side, const integer_t l,
        const VectorV& v, const integer_t incv,
        const typename traits::vector_traits< VectorV >::value_type tau,
        MatrixC& c ) {
    typedef typename traits::vector_traits< VectorV >::value_type value_type;
    integer_t info(0);
    larz_impl< value_type >::invoke( side, l, v, incv, tau, c,
            optimal_workspace() );
    return info;
}

} // namespace lapack
} // namespace bindings
} // namespace numeric
} // namespace boost

#endif
