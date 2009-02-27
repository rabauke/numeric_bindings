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

#ifndef BOOST_NUMERIC_BINDINGS_LAPACK_BDSDC_HPP
#define BOOST_NUMERIC_BINDINGS_LAPACK_BDSDC_HPP

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
    inline void bdsdc( char const uplo, char const compq, integer_t const n,
            float* d, float* e, float* u, integer_t const ldu, float* vt,
            integer_t const ldvt, float* q, integer_t* iq, float* work,
            integer_t* iwork, integer_t& info ) {
        LAPACK_SBDSDC( &uplo, &compq, &n, d, e, u, &ldu, vt, &ldvt, q, iq,
                work, iwork, &info );
    }
    inline void bdsdc( char const uplo, char const compq, integer_t const n,
            double* d, double* e, double* u, integer_t const ldu, double* vt,
            integer_t const ldvt, double* q, integer_t* iq, double* work,
            integer_t* iwork, integer_t& info ) {
        LAPACK_DBDSDC( &uplo, &compq, &n, d, e, u, &ldu, vt, &ldvt, q, iq,
                work, iwork, &info );
    }
}

// value-type based template
template< typename ValueType >
struct bdsdc_impl {

    typedef ValueType value_type;
    typedef typename traits::type_traits<ValueType>::real_type real_type;

    // user-defined workspace specialization
    template< typename VectorD, typename VectorE, typename MatrixU,
            typename MatrixVT, typename VectorQ, typename VectorIQ,
            typename WORK, typename IWORK >
    static void compute( char const uplo, char const compq, integer_t const n,
            VectorD& d, VectorE& e, MatrixU& u, MatrixVT& vt, VectorQ& q,
            VectorIQ& iq, integer_t& info, detail::workspace2< WORK,
            IWORK > work ) {
        BOOST_STATIC_ASSERT( boost::is_same< typename traits::vector_traits<
                VectorD >::value_type, typename traits::vector_traits<
                VectorE >::value_type > );
        BOOST_STATIC_ASSERT( boost::is_same< typename traits::vector_traits<
                VectorD >::value_type, typename traits::matrix_traits<
                MatrixU >::value_type > );
        BOOST_STATIC_ASSERT( boost::is_same< typename traits::vector_traits<
                VectorD >::value_type, typename traits::matrix_traits<
                MatrixVT >::value_type > );
        BOOST_STATIC_ASSERT( boost::is_same< typename traits::vector_traits<
                VectorD >::value_type, typename traits::vector_traits<
                VectorQ >::value_type > );
#ifndef NDEBUG
        assert( uplo == 'U' || uplo == 'L' );
        assert( compq == 'N' || compq == 'P' || compq == 'I' );
        assert( n >= 0 );
        assert( traits::vector_size(e) >= n-1 );
        assert( traits::vector_size(work.select(real_type()) >= min_size_work(
                compq, n )));
        assert( traits::vector_size(work.select(integer_t()) >=
                min_size_iwork( n )));
#endif
        detail::bdsdc( uplo, compq, n, traits::vector_storage(d),
                traits::vector_storage(e), traits::matrix_storage(u),
                traits::leading_dimension(u), traits::matrix_storage(vt),
                traits::leading_dimension(vt), traits::vector_storage(q),
                traits::vector_storage(iq),
                traits::vector_storage(work.select(real_type())),
                traits::vector_storage(work.select(integer_t())), info );
    }

    // minimal workspace specialization
    template< typename VectorD, typename VectorE, typename MatrixU,
            typename MatrixVT, typename VectorQ, typename VectorIQ >
    static void compute( char const uplo, char const compq, integer_t const n,
            VectorD& d, VectorE& e, MatrixU& u, MatrixVT& vt, VectorQ& q,
            VectorIQ& iq, integer_t& info, minimal_workspace work ) {
        traits::detail::array< real_type > tmp_work( min_size_work( compq,
                n ) );
        traits::detail::array< integer_t > tmp_iwork( min_size_iwork( n ) );
        compute( uplo, compq, n, d, e, u, vt, q, iq, info,
                workspace( tmp_work, tmp_iwork ) );
    }

    // optimal workspace specialization
    template< typename VectorD, typename VectorE, typename MatrixU,
            typename MatrixVT, typename VectorQ, typename VectorIQ >
    static void compute( char const uplo, char const compq, integer_t const n,
            VectorD& d, VectorE& e, MatrixU& u, MatrixVT& vt, VectorQ& q,
            VectorIQ& iq, integer_t& info, optimal_workspace work ) {
        compute( uplo, compq, n, d, e, u, vt, q, iq, info,
                minimal_workspace() );
    }

    static integer_t min_size_work( char const compq, integer_t const n ) {
        switch ( compq ) {
            case 'N': return 4*n;
            case 'P': return 6*n;
            case 'I': return 3*n*n + 4*n;
        }
    }

    static integer_t min_size_iwork( integer_t const n ) {
        return 8*n;
    }
};


// template function to call bdsdc
template< typename VectorD, typename VectorE, typename MatrixU,
        typename MatrixVT, typename VectorQ, typename VectorIQ,
        typename Workspace >
inline integer_t bdsdc( char const uplo, char const compq,
        integer_t const n, VectorD& d, VectorE& e, MatrixU& u, MatrixVT& vt,
        VectorQ& q, VectorIQ& iq, Workspace work = optimal_workspace() ) {
    typedef typename traits::vector_traits< VectorD >::value_type value_type;
    integer_t info(0);
    bdsdc_impl< value_type >::compute( uplo, compq, n, d, e, u, vt, q,
            iq, info, work );
    return info;
}

}}}} // namespace boost::numeric::bindings::lapack

#endif
