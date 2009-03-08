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

#ifndef BOOST_NUMERIC_BINDINGS_LAPACK_HEGVD_HPP
#define BOOST_NUMERIC_BINDINGS_LAPACK_HEGVD_HPP

#include <boost/numeric/bindings/lapack/lapack.h>
#include <boost/numeric/bindings/lapack/workspace.hpp>
#include <boost/numeric/bindings/traits/detail/array.hpp>
#include <boost/numeric/bindings/traits/detail/utils.hpp>
#include <boost/numeric/bindings/traits/traits.hpp>
#include <boost/numeric/bindings/traits/type_traits.hpp>
#include <boost/static_assert.hpp>
#include <boost/type_traits/is_same.hpp>
#include <cassert>

namespace boost {
namespace numeric {
namespace bindings {
namespace lapack {

//$DESCRIPTION

// overloaded functions to call lapack
namespace detail {
    inline void hegvd( integer_t const itype, char const jobz,
            char const uplo, integer_t const n, traits::complex_f* a,
            integer_t const lda, traits::complex_f* b, integer_t const ldb,
            float* w, traits::complex_f* work, integer_t const lwork,
            float* rwork, integer_t const lrwork, integer_t* iwork,
            integer_t const liwork, integer_t& info ) {
        LAPACK_CHEGVD( &itype, &jobz, &uplo, &n, traits::complex_ptr(a), &lda,
                traits::complex_ptr(b), &ldb, w, traits::complex_ptr(work),
                &lwork, rwork, &lrwork, iwork, &liwork, &info );
    }
    inline void hegvd( integer_t const itype, char const jobz,
            char const uplo, integer_t const n, traits::complex_d* a,
            integer_t const lda, traits::complex_d* b, integer_t const ldb,
            double* w, traits::complex_d* work, integer_t const lwork,
            double* rwork, integer_t const lrwork, integer_t* iwork,
            integer_t const liwork, integer_t& info ) {
        LAPACK_ZHEGVD( &itype, &jobz, &uplo, &n, traits::complex_ptr(a), &lda,
                traits::complex_ptr(b), &ldb, w, traits::complex_ptr(work),
                &lwork, rwork, &lrwork, iwork, &liwork, &info );
    }
}

// value-type based template
template< typename ValueType >
struct hegvd_impl {

    typedef ValueType value_type;
    typedef typename traits::type_traits<ValueType>::real_type real_type;

    // user-defined workspace specialization
    template< typename MatrixA, typename MatrixB, typename VectorW,
            typename WORK, typename RWORK, typename IWORK >
    static void compute( integer_t const itype, char const jobz,
            integer_t const n, MatrixA& a, MatrixB& b, VectorW& w,
            integer_t& info, detail::workspace3< WORK, RWORK, IWORK > work ) {
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixA >::value_type, typename traits::matrix_traits<
                MatrixB >::value_type >::value) );
#ifndef NDEBUG
        assert( jobz == 'N' || jobz == 'V' );
        assert( traits::matrix_uplo_tag(a) == 'U' ||
                traits::matrix_uplo_tag(a) == 'L' );
        assert( n >= 0 );
        assert( traits::leading_dimension(a) >= std::max(1,n) );
        assert( traits::leading_dimension(b) >= std::max(1,n) );
        assert( traits::vector_size(work.select(value_type()) >=
                min_size_work( jobz, n )));
        assert( traits::vector_size(work.select(real_type()) >=
                min_size_rwork( jobz, n )));
        assert( traits::vector_size(work.select(integer_t()) >=
                min_size_iwork( jobz, n )));
#endif
        detail::hegvd( itype, jobz, traits::matrix_uplo_tag(a), n,
                traits::matrix_storage(a), traits::leading_dimension(a),
                traits::matrix_storage(b), traits::leading_dimension(b),
                traits::vector_storage(w),
                traits::vector_storage(work.select(value_type())),
                traits::vector_size(work.select(value_type())),
                traits::vector_storage(work.select(real_type())),
                traits::vector_size(work.select(real_type())),
                traits::vector_storage(work.select(integer_t())),
                traits::vector_size(work.select(integer_t())), info );
    }

    // minimal workspace specialization
    template< typename MatrixA, typename MatrixB, typename VectorW >
    static void compute( integer_t const itype, char const jobz,
            integer_t const n, MatrixA& a, MatrixB& b, VectorW& w,
            integer_t& info, minimal_workspace work ) {
        traits::detail::array< value_type > tmp_work( min_size_work( jobz,
                n ) );
        traits::detail::array< real_type > tmp_rwork( min_size_rwork( jobz,
                n ) );
        traits::detail::array< integer_t > tmp_iwork( min_size_iwork( jobz,
                n ) );
        compute( itype, jobz, n, a, b, w, info, workspace( tmp_work,
                tmp_rwork, tmp_iwork ) );
    }

    // optimal workspace specialization
    template< typename MatrixA, typename MatrixB, typename VectorW >
    static void compute( integer_t const itype, char const jobz,
            integer_t const n, MatrixA& a, MatrixB& b, VectorW& w,
            integer_t& info, optimal_workspace work ) {
        value_type opt_size_work;
        real_type opt_size_rwork;
        integer_t opt_size_iwork;
        detail::hegvd( itype, jobz, traits::matrix_uplo_tag(a), n,
                traits::matrix_storage(a), traits::leading_dimension(a),
                traits::matrix_storage(b), traits::leading_dimension(b),
                traits::vector_storage(w), &opt_size_work, -1,
                &opt_size_rwork, -1, &opt_size_iwork, -1, info );
        traits::detail::array< value_type > tmp_work(
                traits::detail::to_int( opt_size_work ) );
        traits::detail::array< real_type > tmp_rwork(
                traits::detail::to_int( opt_size_rwork ) );
        traits::detail::array< integer_t > tmp_iwork( opt_size_iwork );
        compute( itype, jobz, n, a, b, w, info, workspace( tmp_work,
                tmp_rwork, tmp_iwork ) );
    }

    static integer_t min_size_work( char const jobz, integer_t const n ) {
        if ( n < 2 )
            return 1;
        else {
            if ( jobz == 'N' )
                return n+1;
            else
                return 2*n + n*n;
        }
    }

    static integer_t min_size_rwork( char const jobz, integer_t const n ) {
        if ( n < 2 )
            return 1;
        else {
            if ( jobz == 'N' )
                return n;
            else
                return 1 + 5*n + 2*n*n;
        }
    }

    static integer_t min_size_iwork( char const jobz, integer_t const n ) {
        if ( jobz == 'N' || n < 2 )
            return 1;
        else
            return 3 + 5*n;
    }
};


// template function to call hegvd
template< typename MatrixA, typename MatrixB, typename VectorW,
        typename Workspace >
inline integer_t hegvd( integer_t const itype, char const jobz,
        integer_t const n, MatrixA& a, MatrixB& b, VectorW& w,
        Workspace work = optimal_workspace() ) {
    typedef typename traits::matrix_traits< MatrixA >::value_type value_type;
    integer_t info(0);
    hegvd_impl< value_type >::compute( itype, jobz, n, a, b, w, info,
            work );
    return info;
}

}}}} // namespace boost::numeric::bindings::lapack

#endif
