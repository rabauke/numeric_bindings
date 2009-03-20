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

#ifndef BOOST_NUMERIC_BINDINGS_LAPACK_DRIVER_SYSVX_HPP
#define BOOST_NUMERIC_BINDINGS_LAPACK_DRIVER_SYSVX_HPP

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
    inline void sysvx( char const fact, char const uplo, integer_t const n,
            integer_t const nrhs, float* a, integer_t const lda, float* af,
            integer_t const ldaf, integer_t* ipiv, float* b,
            integer_t const ldb, float* x, integer_t const ldx, float& rcond,
            float* ferr, float* berr, float* work, integer_t const lwork,
            integer_t* iwork, integer_t& info ) {
        LAPACK_SSYSVX( &fact, &uplo, &n, &nrhs, a, &lda, af, &ldaf, ipiv, b,
                &ldb, x, &ldx, &rcond, ferr, berr, work, &lwork, iwork,
                &info );
    }
    inline void sysvx( char const fact, char const uplo, integer_t const n,
            integer_t const nrhs, double* a, integer_t const lda, double* af,
            integer_t const ldaf, integer_t* ipiv, double* b,
            integer_t const ldb, double* x, integer_t const ldx,
            double& rcond, double* ferr, double* berr, double* work,
            integer_t const lwork, integer_t* iwork, integer_t& info ) {
        LAPACK_DSYSVX( &fact, &uplo, &n, &nrhs, a, &lda, af, &ldaf, ipiv, b,
                &ldb, x, &ldx, &rcond, ferr, berr, work, &lwork, iwork,
                &info );
    }
    inline void sysvx( char const fact, char const uplo, integer_t const n,
            integer_t const nrhs, traits::complex_f* a, integer_t const lda,
            traits::complex_f* af, integer_t const ldaf, integer_t* ipiv,
            traits::complex_f* b, integer_t const ldb, traits::complex_f* x,
            integer_t const ldx, float& rcond, float* ferr, float* berr,
            traits::complex_f* work, integer_t const lwork, float* rwork,
            integer_t& info ) {
        LAPACK_CSYSVX( &fact, &uplo, &n, &nrhs, traits::complex_ptr(a), &lda,
                traits::complex_ptr(af), &ldaf, ipiv, traits::complex_ptr(b),
                &ldb, traits::complex_ptr(x), &ldx, &rcond, ferr, berr,
                traits::complex_ptr(work), &lwork, rwork, &info );
    }
    inline void sysvx( char const fact, char const uplo, integer_t const n,
            integer_t const nrhs, traits::complex_d* a, integer_t const lda,
            traits::complex_d* af, integer_t const ldaf, integer_t* ipiv,
            traits::complex_d* b, integer_t const ldb, traits::complex_d* x,
            integer_t const ldx, double& rcond, double* ferr, double* berr,
            traits::complex_d* work, integer_t const lwork, double* rwork,
            integer_t& info ) {
        LAPACK_ZSYSVX( &fact, &uplo, &n, &nrhs, traits::complex_ptr(a), &lda,
                traits::complex_ptr(af), &ldaf, ipiv, traits::complex_ptr(b),
                &ldb, traits::complex_ptr(x), &ldx, &rcond, ferr, berr,
                traits::complex_ptr(work), &lwork, rwork, &info );
    }
}

// value-type based template
template< typename ValueType, typename Enable = void >
struct sysvx_impl{};

// real specialization
template< typename ValueType >
struct sysvx_impl< ValueType, typename boost::enable_if< traits::is_real<ValueType> >::type > {

    typedef ValueType value_type;
    typedef typename traits::type_traits<ValueType>::real_type real_type;

    // user-defined workspace specialization
    template< typename MatrixA, typename MatrixAF, typename VectorIPIV,
            typename MatrixB, typename MatrixX, typename VectorFERR,
            typename VectorBERR, typename WORK, typename IWORK >
    static void invoke( char const fact, MatrixA& a, MatrixAF& af,
            VectorIPIV& ipiv, MatrixB& b, MatrixX& x, real_type& rcond,
            VectorFERR& ferr, VectorBERR& berr, integer_t& info,
            detail::workspace2< WORK, IWORK > work ) {
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixA >::value_type, typename traits::matrix_traits<
                MatrixAF >::value_type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixA >::value_type, typename traits::matrix_traits<
                MatrixB >::value_type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixA >::value_type, typename traits::matrix_traits<
                MatrixX >::value_type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixA >::value_type, typename traits::vector_traits<
                VectorFERR >::value_type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixA >::value_type, typename traits::vector_traits<
                VectorBERR >::value_type >::value) );
        BOOST_ASSERT( fact == 'F' || fact == 'N' );
        BOOST_ASSERT( traits::matrix_uplo_tag(a) == 'U' ||
                traits::matrix_uplo_tag(a) == 'L' );
        BOOST_ASSERT( traits::matrix_num_columns(a) >= 0 );
        BOOST_ASSERT( traits::matrix_num_columns(x) >= 0 );
        BOOST_ASSERT( traits::leading_dimension(a) >= std::max(1,
                traits::matrix_num_columns(a)) );
        BOOST_ASSERT( traits::leading_dimension(af) >= std::max(1,
                traits::matrix_num_columns(a)) );
        BOOST_ASSERT( traits::leading_dimension(b) >= std::max(1,
                traits::matrix_num_columns(a)) );
        BOOST_ASSERT( traits::leading_dimension(x) >= std::max(1,
                traits::matrix_num_columns(a)) );
        BOOST_ASSERT( traits::vector_size(berr) >=
                traits::matrix_num_columns(x) );
        BOOST_ASSERT( traits::vector_size(work.select(real_type())) >=
                min_size_work( traits::matrix_num_columns(a) ));
        BOOST_ASSERT( traits::vector_size(work.select(integer_t())) >=
                min_size_iwork( traits::matrix_num_columns(a) ));
        detail::sysvx( fact, traits::matrix_uplo_tag(a),
                traits::matrix_num_columns(a), traits::matrix_num_columns(x),
                traits::matrix_storage(a), traits::leading_dimension(a),
                traits::matrix_storage(af), traits::leading_dimension(af),
                traits::vector_storage(ipiv), traits::matrix_storage(b),
                traits::leading_dimension(b), traits::matrix_storage(x),
                traits::leading_dimension(x), rcond,
                traits::vector_storage(ferr), traits::vector_storage(berr),
                traits::vector_storage(work.select(real_type())),
                traits::vector_size(work.select(real_type())),
                traits::vector_storage(work.select(integer_t())), info );
    }

    // minimal workspace specialization
    template< typename MatrixA, typename MatrixAF, typename VectorIPIV,
            typename MatrixB, typename MatrixX, typename VectorFERR,
            typename VectorBERR >
    static void invoke( char const fact, MatrixA& a, MatrixAF& af,
            VectorIPIV& ipiv, MatrixB& b, MatrixX& x, real_type& rcond,
            VectorFERR& ferr, VectorBERR& berr, integer_t& info,
            minimal_workspace work ) {
        traits::detail::array< real_type > tmp_work( min_size_work(
                traits::matrix_num_columns(a) ) );
        traits::detail::array< integer_t > tmp_iwork( min_size_iwork(
                traits::matrix_num_columns(a) ) );
        invoke( fact, a, af, ipiv, b, x, rcond, ferr, berr, info,
                workspace( tmp_work, tmp_iwork ) );
    }

    // optimal workspace specialization
    template< typename MatrixA, typename MatrixAF, typename VectorIPIV,
            typename MatrixB, typename MatrixX, typename VectorFERR,
            typename VectorBERR >
    static void invoke( char const fact, MatrixA& a, MatrixAF& af,
            VectorIPIV& ipiv, MatrixB& b, MatrixX& x, real_type& rcond,
            VectorFERR& ferr, VectorBERR& berr, integer_t& info,
            optimal_workspace work ) {
        real_type opt_size_work;
        traits::detail::array< integer_t > tmp_iwork( min_size_iwork(
                traits::matrix_num_columns(a) ) );
        detail::sysvx( fact, traits::matrix_uplo_tag(a),
                traits::matrix_num_columns(a), traits::matrix_num_columns(x),
                traits::matrix_storage(a), traits::leading_dimension(a),
                traits::matrix_storage(af), traits::leading_dimension(af),
                traits::vector_storage(ipiv), traits::matrix_storage(b),
                traits::leading_dimension(b), traits::matrix_storage(x),
                traits::leading_dimension(x), rcond,
                traits::vector_storage(ferr), traits::vector_storage(berr),
                &opt_size_work, -1, traits::vector_storage(tmp_iwork), info );
        traits::detail::array< real_type > tmp_work(
                traits::detail::to_int( opt_size_work ) );
        invoke( fact, a, af, ipiv, b, x, rcond, ferr, berr, info,
                workspace( tmp_work, tmp_iwork ) );
    }

    static integer_t min_size_work( integer_t const n ) {
        return std::max( 1, 3*n );
    }

    static integer_t min_size_iwork( integer_t const n ) {
        return n;
    }
};

// complex specialization
template< typename ValueType >
struct sysvx_impl< ValueType, typename boost::enable_if< traits::is_complex<ValueType> >::type > {

    typedef ValueType value_type;
    typedef typename traits::type_traits<ValueType>::real_type real_type;

    // user-defined workspace specialization
    template< typename MatrixA, typename MatrixAF, typename VectorIPIV,
            typename MatrixB, typename MatrixX, typename VectorFERR,
            typename VectorBERR, typename WORK, typename RWORK >
    static void invoke( char const fact, MatrixA& a, MatrixAF& af,
            VectorIPIV& ipiv, MatrixB& b, MatrixX& x, real_type& rcond,
            VectorFERR& ferr, VectorBERR& berr, integer_t& info,
            detail::workspace2< WORK, RWORK > work ) {
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::vector_traits<
                VectorFERR >::value_type, typename traits::vector_traits<
                VectorBERR >::value_type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixA >::value_type, typename traits::matrix_traits<
                MatrixAF >::value_type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixA >::value_type, typename traits::matrix_traits<
                MatrixB >::value_type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixA >::value_type, typename traits::matrix_traits<
                MatrixX >::value_type >::value) );
        BOOST_ASSERT( fact == 'F' || fact == 'N' );
        BOOST_ASSERT( traits::matrix_uplo_tag(a) == 'U' ||
                traits::matrix_uplo_tag(a) == 'L' );
        BOOST_ASSERT( traits::matrix_num_columns(a) >= 0 );
        BOOST_ASSERT( traits::matrix_num_columns(x) >= 0 );
        BOOST_ASSERT( traits::leading_dimension(a) >= std::max(1,
                traits::matrix_num_columns(a)) );
        BOOST_ASSERT( traits::leading_dimension(af) >= std::max(1,
                traits::matrix_num_columns(a)) );
        BOOST_ASSERT( traits::leading_dimension(b) >= std::max(1,
                traits::matrix_num_columns(a)) );
        BOOST_ASSERT( traits::leading_dimension(x) >= std::max(1,
                traits::matrix_num_columns(a)) );
        BOOST_ASSERT( traits::vector_size(berr) >=
                traits::matrix_num_columns(x) );
        BOOST_ASSERT( traits::vector_size(work.select(value_type())) >=
                min_size_work( traits::matrix_num_columns(a) ));
        BOOST_ASSERT( traits::vector_size(work.select(real_type())) >=
                min_size_rwork( traits::matrix_num_columns(a) ));
        detail::sysvx( fact, traits::matrix_uplo_tag(a),
                traits::matrix_num_columns(a), traits::matrix_num_columns(x),
                traits::matrix_storage(a), traits::leading_dimension(a),
                traits::matrix_storage(af), traits::leading_dimension(af),
                traits::vector_storage(ipiv), traits::matrix_storage(b),
                traits::leading_dimension(b), traits::matrix_storage(x),
                traits::leading_dimension(x), rcond,
                traits::vector_storage(ferr), traits::vector_storage(berr),
                traits::vector_storage(work.select(value_type())),
                traits::vector_size(work.select(value_type())),
                traits::vector_storage(work.select(real_type())), info );
    }

    // minimal workspace specialization
    template< typename MatrixA, typename MatrixAF, typename VectorIPIV,
            typename MatrixB, typename MatrixX, typename VectorFERR,
            typename VectorBERR >
    static void invoke( char const fact, MatrixA& a, MatrixAF& af,
            VectorIPIV& ipiv, MatrixB& b, MatrixX& x, real_type& rcond,
            VectorFERR& ferr, VectorBERR& berr, integer_t& info,
            minimal_workspace work ) {
        traits::detail::array< value_type > tmp_work( min_size_work(
                traits::matrix_num_columns(a) ) );
        traits::detail::array< real_type > tmp_rwork( min_size_rwork(
                traits::matrix_num_columns(a) ) );
        invoke( fact, a, af, ipiv, b, x, rcond, ferr, berr, info,
                workspace( tmp_work, tmp_rwork ) );
    }

    // optimal workspace specialization
    template< typename MatrixA, typename MatrixAF, typename VectorIPIV,
            typename MatrixB, typename MatrixX, typename VectorFERR,
            typename VectorBERR >
    static void invoke( char const fact, MatrixA& a, MatrixAF& af,
            VectorIPIV& ipiv, MatrixB& b, MatrixX& x, real_type& rcond,
            VectorFERR& ferr, VectorBERR& berr, integer_t& info,
            optimal_workspace work ) {
        value_type opt_size_work;
        traits::detail::array< real_type > tmp_rwork( min_size_rwork(
                traits::matrix_num_columns(a) ) );
        detail::sysvx( fact, traits::matrix_uplo_tag(a),
                traits::matrix_num_columns(a), traits::matrix_num_columns(x),
                traits::matrix_storage(a), traits::leading_dimension(a),
                traits::matrix_storage(af), traits::leading_dimension(af),
                traits::vector_storage(ipiv), traits::matrix_storage(b),
                traits::leading_dimension(b), traits::matrix_storage(x),
                traits::leading_dimension(x), rcond,
                traits::vector_storage(ferr), traits::vector_storage(berr),
                &opt_size_work, -1, traits::vector_storage(tmp_rwork), info );
        traits::detail::array< value_type > tmp_work(
                traits::detail::to_int( opt_size_work ) );
        invoke( fact, a, af, ipiv, b, x, rcond, ferr, berr, info,
                workspace( tmp_work, tmp_rwork ) );
    }

    static integer_t min_size_work( integer_t const n ) {
        return std::max( 1, 2*n );
    }

    static integer_t min_size_rwork( integer_t const n ) {
        return n;
    }
};


// template function to call sysvx
template< typename MatrixA, typename MatrixAF, typename VectorIPIV,
        typename MatrixB, typename MatrixX, typename VectorFERR,
        typename VectorBERR, typename Workspace >
inline integer_t sysvx( char const fact, MatrixA& a, MatrixAF& af,
        VectorIPIV& ipiv, MatrixB& b, MatrixX& x,
        typename traits::matrix_traits< MatrixA >::value_type& rcond,
        VectorFERR& ferr, VectorBERR& berr, Workspace work ) {
    typedef typename traits::matrix_traits< MatrixA >::value_type value_type;
    integer_t info(0);
    sysvx_impl< value_type >::invoke( fact, a, af, ipiv, b, x, rcond,
            ferr, berr, info, work );
    return info;
}

// template function to call sysvx, default workspace type
template< typename MatrixA, typename MatrixAF, typename VectorIPIV,
        typename MatrixB, typename MatrixX, typename VectorFERR,
        typename VectorBERR >
inline integer_t sysvx( char const fact, MatrixA& a, MatrixAF& af,
        VectorIPIV& ipiv, MatrixB& b, MatrixX& x,
        typename traits::matrix_traits< MatrixA >::value_type& rcond,
        VectorFERR& ferr, VectorBERR& berr ) {
    typedef typename traits::matrix_traits< MatrixA >::value_type value_type;
    integer_t info(0);
    sysvx_impl< value_type >::invoke( fact, a, af, ipiv, b, x, rcond,
            ferr, berr, info, optimal_workspace() );
    return info;
}

}}}} // namespace boost::numeric::bindings::lapack

#endif
