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

#ifndef BOOST_NUMERIC_BINDINGS_LAPACK_HBEV_HPP
#define BOOST_NUMERIC_BINDINGS_LAPACK_HBEV_HPP

#include <boost/numeric/bindings/lapack/lapack.h>
#include <boost/numeric/bindings/lapack/workspace.hpp>
#include <boost/numeric/bindings/traits/detail/array.hpp>
#include <boost/numeric/bindings/traits/traits.hpp>
#include <boost/numeric/bindings/traits/type_traits.hpp>
#include <boost/static_assert.hpp
#include <boost/type_traits/is_same.hpp>
#include <cassert>

namespace boost {
namespace numeric {
namespace bindings {
namespace lapack {

//$DESCRIPTION

// overloaded functions to call lapack
namespace detail {
    inline void hbev( char const jobz, char const uplo, integer_t const n,
            integer_t const kd, traits::complex_f* ab, integer_t const ldab,
            float* w, traits::complex_f* z, integer_t const ldz,
            traits::complex_f* work, float* rwork, integer_t& info ) {
        LAPACK_CHBEV( &jobz, &uplo, &n, &kd, traits::complex_ptr(ab), &ldab,
                w, traits::complex_ptr(z), &ldz, traits::complex_ptr(work),
                rwork, &info );
    }
    inline void hbev( char const jobz, char const uplo, integer_t const n,
            integer_t const kd, traits::complex_d* ab, integer_t const ldab,
            double* w, traits::complex_d* z, integer_t const ldz,
            traits::complex_d* work, double* rwork, integer_t& info ) {
        LAPACK_ZHBEV( &jobz, &uplo, &n, &kd, traits::complex_ptr(ab), &ldab,
                w, traits::complex_ptr(z), &ldz, traits::complex_ptr(work),
                rwork, &info );
    }
}

// value-type based template
template< typename ValueType >
struct hbev_impl {

    typedef ValueType value_type;
    typedef typename traits::type_traits<ValueType>::real_type real_type;

    // user-defined workspace specialization
    template< typename MatrixAB, typename VectorW, typename MatrixZ,
            typename WORK, typename RWORK >
    static void compute( char const jobz, integer_t const n,
            integer_t const kd, MatrixAB& ab, VectorW& w, MatrixZ& z,
            integer_t& info, detail::workspace2< WORK, RWORK > work ) {
        BOOST_STATIC_ASSERT( boost::is_same< typename traits::matrix_traits<
                MatrixAB >::value_type, typename traits::matrix_traits<
                MatrixZ >::value_type > );
#ifndef NDEBUG
        assert( jobz == 'N' || jobz == 'V' );
        assert( traits::matrix_uplo_tag(ab) == 'U' ||
                traits::matrix_uplo_tag(ab) == 'L' );
        assert( n >= 0 );
        assert( kd >= 0 );
        assert( traits::leading_dimension(ab) >= kd );
        assert( traits::vector_size(work.select(value_type()) >=
                min_size_work( n )));
        assert( traits::vector_size(work.select(real_type()) >=
                min_size_rwork( n )));
#endif
        detail::hbev( jobz, traits::matrix_uplo_tag(ab), n, kd,
                traits::matrix_storage(ab), traits::leading_dimension(ab),
                traits::vector_storage(w), traits::matrix_storage(z),
                traits::leading_dimension(z),
                traits::vector_storage(work.select(value_type())),
                traits::vector_storage(work.select(real_type())), info );
    }

    // minimal workspace specialization
    template< typename MatrixAB, typename VectorW, typename MatrixZ >
    static void compute( char const jobz, integer_t const n,
            integer_t const kd, MatrixAB& ab, VectorW& w, MatrixZ& z,
            integer_t& info, minimal_workspace work ) {
        traits::detail::array< value_type > tmp_work( min_size_work( n ) );
        traits::detail::array< real_type > tmp_rwork( min_size_rwork( n ) );
        compute( jobz, n, kd, ab, w, z, info, workspace( tmp_work,
                tmp_rwork ) );
    }

    // optimal workspace specialization
    template< typename MatrixAB, typename VectorW, typename MatrixZ >
    static void compute( char const jobz, integer_t const n,
            integer_t const kd, MatrixAB& ab, VectorW& w, MatrixZ& z,
            integer_t& info, optimal_workspace work ) {
        compute( jobz, n, kd, ab, w, z, info, minimal_workspace() );
    }

    static integer_t min_size_work( integer_t const n ) {
        return n;
    }

    static integer_t min_size_rwork( integer_t const n ) {
        return std::max(1,3*n-2);
    }
};


// template function to call hbev
template< typename MatrixAB, typename VectorW, typename MatrixZ,
        typename Workspace >
inline integer_t hbev( char const jobz, integer_t const n,
        integer_t const kd, MatrixAB& ab, VectorW& w, MatrixZ& z,
        Workspace work = optimal_workspace() ) {
    typedef typename traits::matrix_traits< MatrixAB >::value_type value_type;
    integer_t info(0);
    hbev_impl< value_type >::compute( jobz, n, kd, ab, w, z, info,
            work );
    return info;
}

}}}} // namespace boost::numeric::bindings::lapack

#endif
