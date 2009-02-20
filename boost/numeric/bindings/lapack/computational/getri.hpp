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

#ifndef BOOST_NUMERIC_BINDINGS_LAPACK_GETRI_HPP
#define BOOST_NUMERIC_BINDINGS_LAPACK_GETRI_HPP

#include <boost/numeric/bindings/lapack/lapack.h>
#include <boost/numeric/bindings/lapack/workspace.hpp>
#include <boost/numeric/bindings/traits/detail/array.hpp>
#include <boost/numeric/bindings/traits/detail/utils.hpp>
#include <boost/numeric/bindings/traits/is_complex.hpp>
#include <boost/numeric/bindings/traits/is_real.hpp>
#include <boost/numeric/bindings/traits/traits.hpp>
#include <boost/numeric/bindings/traits/type_traits.hpp>
#include <boost/utility/enable_if.hpp>
#include <cassert>

namespace boost {
namespace numeric {
namespace bindings {
namespace lapack {

//$DESCRIPTION

// overloaded functions to call lapack
namespace detail {
    inline void getri( integer_t const n, float* a, integer_t const lda,
            integer_t* ipiv, float* work, integer_t const lwork,
            integer_t& info ) {
        LAPACK_SGETRI( &n, a, &lda, ipiv, work, &lwork, &info );
    }
    inline void getri( integer_t const n, double* a, integer_t const lda,
            integer_t* ipiv, double* work, integer_t const lwork,
            integer_t& info ) {
        LAPACK_DGETRI( &n, a, &lda, ipiv, work, &lwork, &info );
    }
    inline void getri( integer_t const n, traits::complex_f* a,
            integer_t const lda, integer_t* ipiv, traits::complex_f* work,
            integer_t const lwork, integer_t& info ) {
        LAPACK_CGETRI( &n, traits::complex_ptr(a), &lda, ipiv,
                traits::complex_ptr(work), &lwork, &info );
    }
    inline void getri( integer_t const n, traits::complex_d* a,
            integer_t const lda, integer_t* ipiv, traits::complex_d* work,
            integer_t const lwork, integer_t& info ) {
        LAPACK_ZGETRI( &n, traits::complex_ptr(a), &lda, ipiv,
                traits::complex_ptr(work), &lwork, &info );
    }
}

// value-type based template
template< typename ValueType, typename Enable = void >
struct getri_impl{};

// real specialization
template< typename ValueType >
struct getri_impl< ValueType, typename boost::enable_if< traits::is_real<ValueType> >::type > {

    typedef ValueType value_type;
    typedef typename traits::type_traits<ValueType>::real_type real_type;

    // user-defined workspace specialization
    template< typename MatrixA, typename VectorIPIV, typename WORK >
    static void compute( MatrixA& a, VectorIPIV& ipiv, integer_t& info,
            detail::workspace1< WORK > work ) {
#ifndef NDEBUG
        assert( traits::matrix_size2(a) >= 0 );
        assert( traits::leading_dimension(a) >= std::max(1,
                traits::matrix_size2(a)) );
        assert( traits::vector_size(ipiv) >= traits::matrix_size2(a) );
        assert( traits::vector_size(work.select(real_type()) >= min_size_work(
                traits::matrix_size2(a) )));
#endif
        detail::getri( traits::matrix_size2(a), traits::matrix_storage(a),
                traits::leading_dimension(a), traits::vector_storage(ipiv),
                traits::vector_storage(work.select(real_type())),
                traits::vector_size(work.select(real_type())), info );
    }

    // minimal workspace specialization
    template< typename MatrixA, typename VectorIPIV >
    static void compute( MatrixA& a, VectorIPIV& ipiv, integer_t& info,
            minimal_workspace work ) {
        traits::detail::array< real_type > tmp_work( min_size_work(
                traits::matrix_size2(a) ) );
        compute( a, ipiv, info, workspace( tmp_work ) );
    }

    // optimal workspace specialization
    template< typename MatrixA, typename VectorIPIV >
    static void compute( MatrixA& a, VectorIPIV& ipiv, integer_t& info,
            optimal_workspace work ) {
        real_type opt_size_work;
        detail::getri( traits::matrix_size2(a),
                traits::matrix_storage(a), traits::leading_dimension(a),
                traits::vector_storage(ipiv), &opt_size_work, -1, info );
        traits::detail::array< real_type > tmp_work(
                traits::detail::to_int( opt_size_work ) );
        compute( a, ipiv, info, workspace( tmp_work ) );
    }

    static integer_t min_size_work( integer_t const n ) {
        return std::max( 1, n );
    }
};

// complex specialization
template< typename ValueType >
struct getri_impl< ValueType, typename boost::enable_if< traits::is_complex<ValueType> >::type > {

    typedef ValueType value_type;
    typedef typename traits::type_traits<ValueType>::real_type real_type;

    // user-defined workspace specialization
    template< typename MatrixA, typename VectorIPIV, typename WORK >
    static void compute( MatrixA& a, VectorIPIV& ipiv, integer_t& info,
            detail::workspace1< WORK > work ) {
#ifndef NDEBUG
        assert( traits::matrix_size2(a) >= 0 );
        assert( traits::leading_dimension(a) >= std::max(1,
                traits::matrix_size2(a)) );
        assert( traits::vector_size(ipiv) >= traits::matrix_size2(a) );
        assert( traits::vector_size(work.select(value_type()) >=
                min_size_work( traits::matrix_size2(a) )));
#endif
        detail::getri( traits::matrix_size2(a), traits::matrix_storage(a),
                traits::leading_dimension(a), traits::vector_storage(ipiv),
                traits::vector_storage(work.select(value_type())),
                traits::vector_size(work.select(value_type())), info );
    }

    // minimal workspace specialization
    template< typename MatrixA, typename VectorIPIV >
    static void compute( MatrixA& a, VectorIPIV& ipiv, integer_t& info,
            minimal_workspace work ) {
        traits::detail::array< value_type > tmp_work( min_size_work(
                traits::matrix_size2(a) ) );
        compute( a, ipiv, info, workspace( tmp_work ) );
    }

    // optimal workspace specialization
    template< typename MatrixA, typename VectorIPIV >
    static void compute( MatrixA& a, VectorIPIV& ipiv, integer_t& info,
            optimal_workspace work ) {
        value_type opt_size_work;
        detail::getri( traits::matrix_size2(a),
                traits::matrix_storage(a), traits::leading_dimension(a),
                traits::vector_storage(ipiv), &opt_size_work, -1, info );
        traits::detail::array< value_type > tmp_work(
                traits::detail::to_int( opt_size_work ) );
        compute( a, ipiv, info, workspace( tmp_work ) );
    }

    static integer_t min_size_work( integer_t const n ) {
        return std::max( 1, n );
    }
};


// template function to call getri
template< typename MatrixA, typename VectorIPIV, typename Workspace >
inline integer_t getri( MatrixA& a, VectorIPIV& ipiv,
        Workspace work = optimal_workspace() ) {
    typedef typename traits::matrix_traits< MatrixA >::value_type value_type;
    integer_t info(0);
    getri_impl< value_type >::compute( a, ipiv, info, work );
    return info;
}


}}}} // namespace boost::numeric::bindings::lapack

#endif
