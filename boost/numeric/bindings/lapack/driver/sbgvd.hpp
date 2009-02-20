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

#ifndef BOOST_NUMERIC_BINDINGS_LAPACK_SBGVD_HPP
#define BOOST_NUMERIC_BINDINGS_LAPACK_SBGVD_HPP

#include <boost/numeric/bindings/lapack/lapack.h>
#include <boost/numeric/bindings/lapack/workspace.hpp>
#include <boost/numeric/bindings/traits/detail/array.hpp>
#include <boost/numeric/bindings/traits/detail/utils.hpp>
#include <boost/numeric/bindings/traits/traits.hpp>
#include <boost/numeric/bindings/traits/type_traits.hpp>
#include <cassert>

namespace boost {
namespace numeric {
namespace bindings {
namespace lapack {

//$DESCRIPTION

// overloaded functions to call lapack
namespace detail {
    inline void sbgvd( char const jobz, char const uplo, integer_t const n,
            integer_t const ka, integer_t const kb, float* ab,
            integer_t const ldab, float* bb, integer_t const ldbb, float* w,
            float* z, integer_t const ldz, float* work, integer_t const lwork,
            integer_t* iwork, integer_t const liwork, integer_t& info ) {
        LAPACK_SSBGVD( &jobz, &uplo, &n, &ka, &kb, ab, &ldab, bb, &ldbb, w, z,
                &ldz, work, &lwork, iwork, &liwork, &info );
    }
    inline void sbgvd( char const jobz, char const uplo, integer_t const n,
            integer_t const ka, integer_t const kb, double* ab,
            integer_t const ldab, double* bb, integer_t const ldbb, double* w,
            double* z, integer_t const ldz, double* work,
            integer_t const lwork, integer_t* iwork, integer_t const liwork,
            integer_t& info ) {
        LAPACK_DSBGVD( &jobz, &uplo, &n, &ka, &kb, ab, &ldab, bb, &ldbb, w, z,
                &ldz, work, &lwork, iwork, &liwork, &info );
    }
}

// value-type based template
template< typename ValueType >
struct sbgvd_impl {

    typedef ValueType value_type;
    typedef typename traits::type_traits<ValueType>::real_type real_type;

    // user-defined workspace specialization
    template< typename MatrixAB, typename MatrixBB, typename VectorW,
            typename MatrixZ, typename WORK, typename IWORK >
    static void compute( char const jobz, integer_t const n,
            integer_t const ka, integer_t const kb, MatrixAB& ab,
            MatrixBB& bb, VectorW& w, MatrixZ& z, integer_t& info,
            detail::workspace2< WORK, IWORK > work ) {
#ifndef NDEBUG
        assert( jobz == 'N' || jobz == 'V' );
        assert( traits::matrix_uplo_tag(ab) == 'U' ||
                traits::matrix_uplo_tag(ab) == 'L' );
        assert( n >= 0 );
        assert( ka >= 0 );
        assert( kb >= 0 );
        assert( traits::leading_dimension(ab) >= ka+1 );
        assert( traits::leading_dimension(bb) >= kb+1 );
        assert( traits::vector_size(work.select(real_type()) >= min_size_work(
                jobz, n )));
        assert( traits::vector_size(work.select(integer_t()) >=
                min_size_iwork( jobz, n )));
#endif
        detail::sbgvd( jobz, traits::matrix_uplo_tag(ab), n, ka, kb,
                traits::matrix_storage(ab), traits::leading_dimension(ab),
                traits::matrix_storage(bb), traits::leading_dimension(bb),
                traits::vector_storage(w), traits::matrix_storage(z),
                traits::leading_dimension(z),
                traits::vector_storage(work.select(real_type())),
                traits::vector_size(work.select(real_type())),
                traits::vector_storage(work.select(integer_t())),
                traits::vector_size(work.select(integer_t())), info );
    }

    // minimal workspace specialization
    template< typename MatrixAB, typename MatrixBB, typename VectorW,
            typename MatrixZ >
    static void compute( char const jobz, integer_t const n,
            integer_t const ka, integer_t const kb, MatrixAB& ab,
            MatrixBB& bb, VectorW& w, MatrixZ& z, integer_t& info,
            minimal_workspace work ) {
        traits::detail::array< real_type > tmp_work( min_size_work( jobz,
                n ) );
        traits::detail::array< integer_t > tmp_iwork( min_size_iwork( jobz,
                n ) );
        compute( jobz, n, ka, kb, ab, bb, w, z, info, workspace( tmp_work,
                tmp_iwork ) );
    }

    // optimal workspace specialization
    template< typename MatrixAB, typename MatrixBB, typename VectorW,
            typename MatrixZ >
    static void compute( char const jobz, integer_t const n,
            integer_t const ka, integer_t const kb, MatrixAB& ab,
            MatrixBB& bb, VectorW& w, MatrixZ& z, integer_t& info,
            optimal_workspace work ) {
        real_type opt_size_work;
        integer_t opt_size_iwork;
        detail::sbgvd( jobz, traits::matrix_uplo_tag(ab), n, ka, kb,
                traits::matrix_storage(ab), traits::leading_dimension(ab),
                traits::matrix_storage(bb), traits::leading_dimension(bb),
                traits::vector_storage(w), traits::matrix_storage(z),
                traits::leading_dimension(z), &opt_size_work, -1,
                &opt_size_iwork, -1, info );
        traits::detail::array< real_type > tmp_work(
                traits::detail::to_int( opt_size_work ) );
        traits::detail::array< integer_t > tmp_iwork( opt_size_iwork );
        compute( jobz, n, ka, kb, ab, bb, w, z, info, workspace( tmp_work,
                tmp_iwork ) );
    }

    static integer_t min_size_work( char const jobz, integer_t const n ) {
        if ( n < 2 )
            return 1;
        else {
            if ( jobz == 'N' )
                return 3*n;
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


// template function to call sbgvd
template< typename MatrixAB, typename MatrixBB, typename VectorW,
        typename MatrixZ, typename Workspace >
inline integer_t sbgvd( char const jobz, integer_t const n,
        integer_t const ka, integer_t const kb, MatrixAB& ab, MatrixBB& bb,
        VectorW& w, MatrixZ& z, Workspace work = optimal_workspace() ) {
    typedef typename traits::matrix_traits< MatrixAB >::value_type value_type;
    integer_t info(0);
    sbgvd_impl< value_type >::compute( jobz, n, ka, kb, ab, bb, w, z,
            info, work );
    return info;
}


}}}} // namespace boost::numeric::bindings::lapack

#endif
