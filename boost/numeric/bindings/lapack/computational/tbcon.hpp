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

#ifndef BOOST_NUMERIC_BINDINGS_LAPACK_TBCON_HPP
#define BOOST_NUMERIC_BINDINGS_LAPACK_TBCON_HPP

#include <boost/numeric/bindings/lapack/lapack.h>
#include <boost/numeric/bindings/lapack/workspace.hpp>
#include <boost/numeric/bindings/traits/detail/array.hpp>
#include <boost/numeric/bindings/traits/is_complex.hpp>
#include <boost/numeric/bindings/traits/is_real.hpp>
#include <boost/numeric/bindings/traits/traits.hpp>
#include <boost/numeric/bindings/traits/type_traits.hpp>
#include <boost/static_assert.hpp
#include <boost/type_traits/is_same.hpp>
#include <boost/utility/enable_if.hpp>
#include <cassert>

namespace boost {
namespace numeric {
namespace bindings {
namespace lapack {

//$DESCRIPTION

// overloaded functions to call lapack
namespace detail {
    inline void tbcon( char const norm, char const uplo, char const diag,
            integer_t const n, integer_t const kd, float* ab,
            integer_t const ldab, float& rcond, float* work, integer_t* iwork,
            integer_t& info ) {
        LAPACK_STBCON( &norm, &uplo, &diag, &n, &kd, ab, &ldab, &rcond, work,
                iwork, &info );
    }
    inline void tbcon( char const norm, char const uplo, char const diag,
            integer_t const n, integer_t const kd, double* ab,
            integer_t const ldab, double& rcond, double* work,
            integer_t* iwork, integer_t& info ) {
        LAPACK_DTBCON( &norm, &uplo, &diag, &n, &kd, ab, &ldab, &rcond, work,
                iwork, &info );
    }
    inline void tbcon( char const norm, char const uplo, char const diag,
            integer_t const n, integer_t const kd, traits::complex_f* ab,
            integer_t const ldab, float& rcond, traits::complex_f* work,
            float* rwork, integer_t& info ) {
        LAPACK_CTBCON( &norm, &uplo, &diag, &n, &kd, traits::complex_ptr(ab),
                &ldab, &rcond, traits::complex_ptr(work), rwork, &info );
    }
    inline void tbcon( char const norm, char const uplo, char const diag,
            integer_t const n, integer_t const kd, traits::complex_d* ab,
            integer_t const ldab, double& rcond, traits::complex_d* work,
            double* rwork, integer_t& info ) {
        LAPACK_ZTBCON( &norm, &uplo, &diag, &n, &kd, traits::complex_ptr(ab),
                &ldab, &rcond, traits::complex_ptr(work), rwork, &info );
    }
}

// value-type based template
template< typename ValueType, typename Enable = void >
struct tbcon_impl{};

// real specialization
template< typename ValueType >
struct tbcon_impl< ValueType, typename boost::enable_if< traits::is_real<ValueType> >::type > {

    typedef ValueType value_type;
    typedef typename traits::type_traits<ValueType>::real_type real_type;

    // user-defined workspace specialization
    template< typename MatrixAB, typename WORK, typename IWORK >
    static void compute( char const norm, char const uplo, char const diag,
            integer_t const n, integer_t const kd, MatrixAB& ab,
            real_type& rcond, integer_t& info, detail::workspace2< WORK,
            IWORK > work ) {
        
#ifndef NDEBUG
        assert( norm == '1' || norm == 'O' || norm == 'I' );
        assert( uplo == 'U' || uplo == 'L' );
        assert( diag == 'N' || diag == 'U' );
        assert( n >= 0 );
        assert( kd >= 0 );
        assert( traits::leading_dimension(ab) >= kd+1 );
        assert( traits::vector_size(work.select(real_type()) >= min_size_work(
                n )));
        assert( traits::vector_size(work.select(integer_t()) >=
                min_size_iwork( n )));
#endif
        detail::tbcon( norm, uplo, diag, n, kd, traits::matrix_storage(ab),
                traits::leading_dimension(ab), rcond,
                traits::vector_storage(work.select(real_type())),
                traits::vector_storage(work.select(integer_t())), info );
    }

    // minimal workspace specialization
    template< typename MatrixAB >
    static void compute( char const norm, char const uplo, char const diag,
            integer_t const n, integer_t const kd, MatrixAB& ab,
            real_type& rcond, integer_t& info, minimal_workspace work ) {
        traits::detail::array< real_type > tmp_work( min_size_work( n ) );
        traits::detail::array< integer_t > tmp_iwork( min_size_iwork( n ) );
        compute( norm, uplo, diag, n, kd, ab, rcond, info,
                workspace( tmp_work, tmp_iwork ) );
    }

    // optimal workspace specialization
    template< typename MatrixAB >
    static void compute( char const norm, char const uplo, char const diag,
            integer_t const n, integer_t const kd, MatrixAB& ab,
            real_type& rcond, integer_t& info, optimal_workspace work ) {
        compute( norm, uplo, diag, n, kd, ab, rcond, info,
                minimal_workspace() );
    }

    static integer_t min_size_work( integer_t const n ) {
        return 3*n;
    }

    static integer_t min_size_iwork( integer_t const n ) {
        return n;
    }
};

// complex specialization
template< typename ValueType >
struct tbcon_impl< ValueType, typename boost::enable_if< traits::is_complex<ValueType> >::type > {

    typedef ValueType value_type;
    typedef typename traits::type_traits<ValueType>::real_type real_type;

    // user-defined workspace specialization
    template< typename MatrixAB, typename WORK, typename RWORK >
    static void compute( char const norm, char const uplo, char const diag,
            integer_t const n, integer_t const kd, MatrixAB& ab,
            real_type& rcond, integer_t& info, detail::workspace2< WORK,
            RWORK > work ) {
        
#ifndef NDEBUG
        assert( norm == '1' || norm == 'O' || norm == 'I' );
        assert( uplo == 'U' || uplo == 'L' );
        assert( diag == 'N' || diag == 'U' );
        assert( n >= 0 );
        assert( kd >= 0 );
        assert( traits::leading_dimension(ab) >= kd+1 );
        assert( traits::vector_size(work.select(value_type()) >=
                min_size_work( n )));
        assert( traits::vector_size(work.select(real_type()) >=
                min_size_rwork( n )));
#endif
        detail::tbcon( norm, uplo, diag, n, kd, traits::matrix_storage(ab),
                traits::leading_dimension(ab), rcond,
                traits::vector_storage(work.select(value_type())),
                traits::vector_storage(work.select(real_type())), info );
    }

    // minimal workspace specialization
    template< typename MatrixAB >
    static void compute( char const norm, char const uplo, char const diag,
            integer_t const n, integer_t const kd, MatrixAB& ab,
            real_type& rcond, integer_t& info, minimal_workspace work ) {
        traits::detail::array< value_type > tmp_work( min_size_work( n ) );
        traits::detail::array< real_type > tmp_rwork( min_size_rwork( n ) );
        compute( norm, uplo, diag, n, kd, ab, rcond, info,
                workspace( tmp_work, tmp_rwork ) );
    }

    // optimal workspace specialization
    template< typename MatrixAB >
    static void compute( char const norm, char const uplo, char const diag,
            integer_t const n, integer_t const kd, MatrixAB& ab,
            real_type& rcond, integer_t& info, optimal_workspace work ) {
        compute( norm, uplo, diag, n, kd, ab, rcond, info,
                minimal_workspace() );
    }

    static integer_t min_size_work( integer_t const n ) {
        return 2*n;
    }

    static integer_t min_size_rwork( integer_t const n ) {
        return n;
    }
};


// template function to call tbcon
template< typename MatrixAB, typename Workspace >
inline integer_t tbcon( char const norm, char const uplo,
        char const diag, integer_t const n, integer_t const kd, MatrixAB& ab,
        typename traits::matrix_traits< MatrixAB >::value_type& rcond,
        Workspace work = optimal_workspace() ) {
    typedef typename traits::matrix_traits< MatrixAB >::value_type value_type;
    integer_t info(0);
    tbcon_impl< value_type >::compute( norm, uplo, diag, n, kd, ab,
            rcond, info, work );
    return info;
}

}}}} // namespace boost::numeric::bindings::lapack

#endif
