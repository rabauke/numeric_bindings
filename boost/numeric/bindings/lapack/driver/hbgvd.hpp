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

#ifndef BOOST_NUMERIC_BINDINGS_LAPACK_DRIVER_HBGVD_HPP
#define BOOST_NUMERIC_BINDINGS_LAPACK_DRIVER_HBGVD_HPP

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
    inline void hbgvd( const char jobz, const char uplo, const integer_t n,
            const integer_t ka, const integer_t kb, traits::complex_f* ab,
            const integer_t ldab, traits::complex_f* bb, const integer_t ldbb,
            float* w, traits::complex_f* z, const integer_t ldz,
            traits::complex_f* work, const integer_t lwork, float* rwork,
            const integer_t lrwork, integer_t* iwork, const integer_t liwork,
            integer_t& info ) {
        LAPACK_CHBGVD( &jobz, &uplo, &n, &ka, &kb, traits::complex_ptr(ab),
                &ldab, traits::complex_ptr(bb), &ldbb, w,
                traits::complex_ptr(z), &ldz, traits::complex_ptr(work),
                &lwork, rwork, &lrwork, iwork, &liwork, &info );
    }
    inline void hbgvd( const char jobz, const char uplo, const integer_t n,
            const integer_t ka, const integer_t kb, traits::complex_d* ab,
            const integer_t ldab, traits::complex_d* bb, const integer_t ldbb,
            double* w, traits::complex_d* z, const integer_t ldz,
            traits::complex_d* work, const integer_t lwork, double* rwork,
            const integer_t lrwork, integer_t* iwork, const integer_t liwork,
            integer_t& info ) {
        LAPACK_ZHBGVD( &jobz, &uplo, &n, &ka, &kb, traits::complex_ptr(ab),
                &ldab, traits::complex_ptr(bb), &ldbb, w,
                traits::complex_ptr(z), &ldz, traits::complex_ptr(work),
                &lwork, rwork, &lrwork, iwork, &liwork, &info );
    }
}

// value-type based template
template< typename ValueType >
struct hbgvd_impl {

    typedef ValueType value_type;
    typedef typename traits::type_traits<ValueType>::real_type real_type;

    // user-defined workspace specialization
    template< typename MatrixAB, typename MatrixBB, typename VectorW,
            typename MatrixZ, typename WORK, typename RWORK, typename IWORK >
    static void invoke( const char jobz, const integer_t n,
            const integer_t ka, const integer_t kb, MatrixAB& ab,
            MatrixBB& bb, VectorW& w, MatrixZ& z, integer_t& info,
            detail::workspace3< WORK, RWORK, IWORK > work ) {
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixAB >::value_type, typename traits::matrix_traits<
                MatrixBB >::value_type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixAB >::value_type, typename traits::matrix_traits<
                MatrixZ >::value_type >::value) );
        BOOST_ASSERT( jobz == 'N' || jobz == 'V' );
        BOOST_ASSERT( traits::matrix_uplo_tag(ab) == 'U' ||
                traits::matrix_uplo_tag(ab) == 'L' );
        BOOST_ASSERT( n >= 0 );
        BOOST_ASSERT( ka >= 0 );
        BOOST_ASSERT( kb >= 0 );
        BOOST_ASSERT( traits::leading_dimension(ab) >= ka+1 );
        BOOST_ASSERT( traits::leading_dimension(bb) >= kb+1 );
        BOOST_ASSERT( traits::vector_size(work.select(value_type())) >=
                min_size_work( jobz, n ));
        BOOST_ASSERT( traits::vector_size(work.select(real_type())) >=
                min_size_rwork( jobz, n ));
        BOOST_ASSERT( traits::vector_size(work.select(integer_t())) >=
                min_size_iwork( jobz, n ));
        detail::hbgvd( jobz, traits::matrix_uplo_tag(ab), n, ka, kb,
                traits::matrix_storage(ab), traits::leading_dimension(ab),
                traits::matrix_storage(bb), traits::leading_dimension(bb),
                traits::vector_storage(w), traits::matrix_storage(z),
                traits::leading_dimension(z),
                traits::vector_storage(work.select(value_type())),
                traits::vector_size(work.select(value_type())),
                traits::vector_storage(work.select(real_type())),
                traits::vector_size(work.select(real_type())),
                traits::vector_storage(work.select(integer_t())),
                traits::vector_size(work.select(integer_t())), info );
    }

    // minimal workspace specialization
    template< typename MatrixAB, typename MatrixBB, typename VectorW,
            typename MatrixZ >
    static void invoke( const char jobz, const integer_t n,
            const integer_t ka, const integer_t kb, MatrixAB& ab,
            MatrixBB& bb, VectorW& w, MatrixZ& z, integer_t& info,
            minimal_workspace work ) {
        traits::detail::array< value_type > tmp_work( min_size_work( jobz,
                n ) );
        traits::detail::array< real_type > tmp_rwork( min_size_rwork( jobz,
                n ) );
        traits::detail::array< integer_t > tmp_iwork( min_size_iwork( jobz,
                n ) );
        invoke( jobz, n, ka, kb, ab, bb, w, z, info, workspace( tmp_work,
                tmp_rwork, tmp_iwork ) );
    }

    // optimal workspace specialization
    template< typename MatrixAB, typename MatrixBB, typename VectorW,
            typename MatrixZ >
    static void invoke( const char jobz, const integer_t n,
            const integer_t ka, const integer_t kb, MatrixAB& ab,
            MatrixBB& bb, VectorW& w, MatrixZ& z, integer_t& info,
            optimal_workspace work ) {
        value_type opt_size_work;
        real_type opt_size_rwork;
        integer_t opt_size_iwork;
        detail::hbgvd( jobz, traits::matrix_uplo_tag(ab), n, ka, kb,
                traits::matrix_storage(ab), traits::leading_dimension(ab),
                traits::matrix_storage(bb), traits::leading_dimension(bb),
                traits::vector_storage(w), traits::matrix_storage(z),
                traits::leading_dimension(z), &opt_size_work, -1,
                &opt_size_rwork, -1, &opt_size_iwork, -1, info );
        traits::detail::array< value_type > tmp_work(
                traits::detail::to_int( opt_size_work ) );
        traits::detail::array< real_type > tmp_rwork(
                traits::detail::to_int( opt_size_rwork ) );
        traits::detail::array< integer_t > tmp_iwork( opt_size_iwork );
        invoke( jobz, n, ka, kb, ab, bb, w, z, info, workspace( tmp_work,
                tmp_rwork, tmp_iwork ) );
    }

    static integer_t min_size_work( const char jobz, const integer_t n ) {
        if ( n < 2 )
            return 1;
        else {
            if ( jobz == 'N' )
                return n;
            else
                return 2*n*n;
        }
    }

    static integer_t min_size_rwork( const char jobz, const integer_t n ) {
        if ( n < 2 )
            return 1;
        else {
            if ( jobz == 'N' )
                return n;
            else
                return 1 + 5*n + 2*n*n;
        }
    }

    static integer_t min_size_iwork( const char jobz, const integer_t n ) {
        if ( jobz == 'N' || n < 2 )
            return 1;
        else
            return 3 + 5*n;
    }
};


// template function to call hbgvd
template< typename MatrixAB, typename MatrixBB, typename VectorW,
        typename MatrixZ, typename Workspace >
inline integer_t hbgvd( const char jobz, const integer_t n,
        const integer_t ka, const integer_t kb, MatrixAB& ab, MatrixBB& bb,
        VectorW& w, MatrixZ& z, Workspace work ) {
    typedef typename traits::matrix_traits< MatrixAB >::value_type value_type;
    integer_t info(0);
    hbgvd_impl< value_type >::invoke( jobz, n, ka, kb, ab, bb, w, z,
            info, work );
    return info;
}

// template function to call hbgvd, default workspace type
template< typename MatrixAB, typename MatrixBB, typename VectorW,
        typename MatrixZ >
inline integer_t hbgvd( const char jobz, const integer_t n,
        const integer_t ka, const integer_t kb, MatrixAB& ab, MatrixBB& bb,
        VectorW& w, MatrixZ& z ) {
    typedef typename traits::matrix_traits< MatrixAB >::value_type value_type;
    integer_t info(0);
    hbgvd_impl< value_type >::invoke( jobz, n, ka, kb, ab, bb, w, z,
            info, optimal_workspace() );
    return info;
}

}}}} // namespace boost::numeric::bindings::lapack

#endif
