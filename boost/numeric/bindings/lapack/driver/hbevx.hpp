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

#ifndef BOOST_NUMERIC_BINDINGS_LAPACK_DRIVER_HBEVX_HPP
#define BOOST_NUMERIC_BINDINGS_LAPACK_DRIVER_HBEVX_HPP

#include <boost/assert.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/numeric/bindings/lapack/detail/lapack.h>
#include <boost/numeric/bindings/lapack/workspace.hpp>
#include <boost/numeric/bindings/traits/detail/array.hpp>
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
    inline void hbevx( const char jobz, const char range, const char uplo,
            const integer_t n, const integer_t kd, traits::complex_f* ab,
            const integer_t ldab, traits::complex_f* q, const integer_t ldq,
            const float vl, const float vu, const integer_t il,
            const integer_t iu, const float abstol, integer_t& m, float* w,
            traits::complex_f* z, const integer_t ldz,
            traits::complex_f* work, float* rwork, integer_t* iwork,
            integer_t* ifail, integer_t& info ) {
        LAPACK_CHBEVX( &jobz, &range, &uplo, &n, &kd, traits::complex_ptr(ab),
                &ldab, traits::complex_ptr(q), &ldq, &vl, &vu, &il, &iu,
                &abstol, &m, w, traits::complex_ptr(z), &ldz,
                traits::complex_ptr(work), rwork, iwork, ifail, &info );
    }
    inline void hbevx( const char jobz, const char range, const char uplo,
            const integer_t n, const integer_t kd, traits::complex_d* ab,
            const integer_t ldab, traits::complex_d* q, const integer_t ldq,
            const double vl, const double vu, const integer_t il,
            const integer_t iu, const double abstol, integer_t& m, double* w,
            traits::complex_d* z, const integer_t ldz,
            traits::complex_d* work, double* rwork, integer_t* iwork,
            integer_t* ifail, integer_t& info ) {
        LAPACK_ZHBEVX( &jobz, &range, &uplo, &n, &kd, traits::complex_ptr(ab),
                &ldab, traits::complex_ptr(q), &ldq, &vl, &vu, &il, &iu,
                &abstol, &m, w, traits::complex_ptr(z), &ldz,
                traits::complex_ptr(work), rwork, iwork, ifail, &info );
    }
}

// value-type based template
template< typename ValueType >
struct hbevx_impl {

    typedef ValueType value_type;
    typedef typename traits::type_traits<ValueType>::real_type real_type;

    // user-defined workspace specialization
    template< typename MatrixAB, typename MatrixQ, typename VectorW,
            typename MatrixZ, typename VectorIFAIL, typename WORK,
            typename RWORK, typename IWORK >
    static void invoke( const char jobz, const char range, const integer_t n,
            const integer_t kd, MatrixAB& ab, MatrixQ& q, const real_type vl,
            const real_type vu, const integer_t il, const integer_t iu,
            const real_type abstol, integer_t& m, VectorW& w, MatrixZ& z,
            VectorIFAIL& ifail, integer_t& info, detail::workspace3< WORK,
            RWORK, IWORK > work ) {
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixAB >::value_type, typename traits::matrix_traits<
                MatrixQ >::value_type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixAB >::value_type, typename traits::matrix_traits<
                MatrixZ >::value_type >::value) );
        BOOST_ASSERT( jobz == 'N' || jobz == 'V' );
        BOOST_ASSERT( range == 'A' || range == 'V' || range == 'I' );
        BOOST_ASSERT( traits::matrix_uplo_tag(ab) == 'U' ||
                traits::matrix_uplo_tag(ab) == 'L' );
        BOOST_ASSERT( n >= 0 );
        BOOST_ASSERT( kd >= 0 );
        BOOST_ASSERT( traits::leading_dimension(ab) >= kd );
        BOOST_ASSERT( traits::leading_dimension(q) >= std::max(1,n) );
        BOOST_ASSERT( traits::vector_size(w) >= n );
        BOOST_ASSERT( traits::vector_size(work.select(value_type())) >=
                min_size_work( n ));
        BOOST_ASSERT( traits::vector_size(work.select(real_type())) >=
                min_size_rwork( n ));
        BOOST_ASSERT( traits::vector_size(work.select(integer_t())) >=
                min_size_iwork( n ));
        detail::hbevx( jobz, range, traits::matrix_uplo_tag(ab), n, kd,
                traits::matrix_storage(ab), traits::leading_dimension(ab),
                traits::matrix_storage(q), traits::leading_dimension(q), vl,
                vu, il, iu, abstol, m, traits::vector_storage(w),
                traits::matrix_storage(z), traits::leading_dimension(z),
                traits::vector_storage(work.select(value_type())),
                traits::vector_storage(work.select(real_type())),
                traits::vector_storage(work.select(integer_t())),
                traits::vector_storage(ifail), info );
    }

    // minimal workspace specialization
    template< typename MatrixAB, typename MatrixQ, typename VectorW,
            typename MatrixZ, typename VectorIFAIL >
    static void invoke( const char jobz, const char range, const integer_t n,
            const integer_t kd, MatrixAB& ab, MatrixQ& q, const real_type vl,
            const real_type vu, const integer_t il, const integer_t iu,
            const real_type abstol, integer_t& m, VectorW& w, MatrixZ& z,
            VectorIFAIL& ifail, integer_t& info, minimal_workspace work ) {
        traits::detail::array< value_type > tmp_work( min_size_work( n ) );
        traits::detail::array< real_type > tmp_rwork( min_size_rwork( n ) );
        traits::detail::array< integer_t > tmp_iwork( min_size_iwork( n ) );
        invoke( jobz, range, n, kd, ab, q, vl, vu, il, iu, abstol, m, w, z,
                ifail, info, workspace( tmp_work, tmp_rwork, tmp_iwork ) );
    }

    // optimal workspace specialization
    template< typename MatrixAB, typename MatrixQ, typename VectorW,
            typename MatrixZ, typename VectorIFAIL >
    static void invoke( const char jobz, const char range, const integer_t n,
            const integer_t kd, MatrixAB& ab, MatrixQ& q, const real_type vl,
            const real_type vu, const integer_t il, const integer_t iu,
            const real_type abstol, integer_t& m, VectorW& w, MatrixZ& z,
            VectorIFAIL& ifail, integer_t& info, optimal_workspace work ) {
        invoke( jobz, range, n, kd, ab, q, vl, vu, il, iu, abstol, m, w, z,
                ifail, info, minimal_workspace() );
    }

    static integer_t min_size_work( const integer_t n ) {
        return n;
    }

    static integer_t min_size_rwork( const integer_t n ) {
        return 7*n;
    }

    static integer_t min_size_iwork( const integer_t n ) {
        return 5*n;
    }
};


// template function to call hbevx
template< typename MatrixAB, typename MatrixQ, typename VectorW,
        typename MatrixZ, typename VectorIFAIL, typename Workspace >
inline integer_t hbevx( const char jobz, const char range,
        const integer_t n, const integer_t kd, MatrixAB& ab, MatrixQ& q,
        const typename traits::type_traits< typename traits::matrix_traits<
        MatrixAB >::value_type >::real_type vl,
        const typename traits::type_traits< typename traits::matrix_traits<
        MatrixAB >::value_type >::real_type vu, const integer_t il,
        const integer_t iu, const typename traits::type_traits<
        typename traits::matrix_traits<
        MatrixAB >::value_type >::real_type abstol, integer_t& m, VectorW& w,
        MatrixZ& z, VectorIFAIL& ifail, Workspace work ) {
    typedef typename traits::matrix_traits< MatrixAB >::value_type value_type;
    integer_t info(0);
    hbevx_impl< value_type >::invoke( jobz, range, n, kd, ab, q, vl, vu,
            il, iu, abstol, m, w, z, ifail, info, work );
    return info;
}

// template function to call hbevx, default workspace type
template< typename MatrixAB, typename MatrixQ, typename VectorW,
        typename MatrixZ, typename VectorIFAIL >
inline integer_t hbevx( const char jobz, const char range,
        const integer_t n, const integer_t kd, MatrixAB& ab, MatrixQ& q,
        const typename traits::type_traits< typename traits::matrix_traits<
        MatrixAB >::value_type >::real_type vl,
        const typename traits::type_traits< typename traits::matrix_traits<
        MatrixAB >::value_type >::real_type vu, const integer_t il,
        const integer_t iu, const typename traits::type_traits<
        typename traits::matrix_traits<
        MatrixAB >::value_type >::real_type abstol, integer_t& m, VectorW& w,
        MatrixZ& z, VectorIFAIL& ifail ) {
    typedef typename traits::matrix_traits< MatrixAB >::value_type value_type;
    integer_t info(0);
    hbevx_impl< value_type >::invoke( jobz, range, n, kd, ab, q, vl, vu,
            il, iu, abstol, m, w, z, ifail, info, optimal_workspace() );
    return info;
}

}}}} // namespace boost::numeric::bindings::lapack

#endif
