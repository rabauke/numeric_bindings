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

#ifndef BOOST_NUMERIC_BINDINGS_LAPACK_COMPUTATIONAL_STEMR_HPP
#define BOOST_NUMERIC_BINDINGS_LAPACK_COMPUTATIONAL_STEMR_HPP

#include <boost/assert.hpp>
#include <boost/mpl/bool.hpp>
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
    inline void stemr( const char jobz, const char range, const integer_t n,
            float* d, float* e, const float vl, const float vu,
            const integer_t il, const integer_t iu, integer_t& m, float* w,
            float* z, const integer_t ldz, const integer_t nzc,
            integer_t* isuppz, logical_t& tryrac, float* work,
            const integer_t lwork, integer_t* iwork, const integer_t liwork,
            integer_t& info ) {
        LAPACK_SSTEMR( &jobz, &range, &n, d, e, &vl, &vu, &il, &iu, &m, w, z,
                &ldz, &nzc, isuppz, &tryrac, work, &lwork, iwork, &liwork,
                &info );
    }
    inline void stemr( const char jobz, const char range, const integer_t n,
            double* d, double* e, const double vl, const double vu,
            const integer_t il, const integer_t iu, integer_t& m, double* w,
            double* z, const integer_t ldz, const integer_t nzc,
            integer_t* isuppz, logical_t& tryrac, double* work,
            const integer_t lwork, integer_t* iwork, const integer_t liwork,
            integer_t& info ) {
        LAPACK_DSTEMR( &jobz, &range, &n, d, e, &vl, &vu, &il, &iu, &m, w, z,
                &ldz, &nzc, isuppz, &tryrac, work, &lwork, iwork, &liwork,
                &info );
    }
    inline void stemr( const char jobz, const char range, const integer_t n,
            float* d, float* e, const float vl, const float vu,
            const integer_t il, const integer_t iu, integer_t& m, float* w,
            traits::complex_f* z, const integer_t ldz, const integer_t nzc,
            integer_t* isuppz, logical_t& tryrac, float* work,
            const integer_t lwork, integer_t* iwork, const integer_t liwork,
            integer_t& info ) {
        LAPACK_CSTEMR( &jobz, &range, &n, d, e, &vl, &vu, &il, &iu, &m, w,
                traits::complex_ptr(z), &ldz, &nzc, isuppz, &tryrac, work,
                &lwork, iwork, &liwork, &info );
    }
    inline void stemr( const char jobz, const char range, const integer_t n,
            double* d, double* e, const double vl, const double vu,
            const integer_t il, const integer_t iu, integer_t& m, double* w,
            traits::complex_d* z, const integer_t ldz, const integer_t nzc,
            integer_t* isuppz, logical_t& tryrac, double* work,
            const integer_t lwork, integer_t* iwork, const integer_t liwork,
            integer_t& info ) {
        LAPACK_ZSTEMR( &jobz, &range, &n, d, e, &vl, &vu, &il, &iu, &m, w,
                traits::complex_ptr(z), &ldz, &nzc, isuppz, &tryrac, work,
                &lwork, iwork, &liwork, &info );
    }
}

// value-type based template
template< typename ValueType, typename Enable = void >
struct stemr_impl{};

// real specialization
template< typename ValueType >
struct stemr_impl< ValueType, typename boost::enable_if< traits::is_real<ValueType> >::type > {

    typedef ValueType value_type;
    typedef typename traits::type_traits<ValueType>::real_type real_type;

    // user-defined workspace specialization
    template< typename VectorD, typename VectorE, typename VectorW,
            typename MatrixZ, typename VectorISUPPZ, typename WORK,
            typename IWORK >
    static void invoke( const char jobz, const char range, const integer_t n,
            VectorD& d, VectorE& e, const real_type vl, const real_type vu,
            const integer_t il, const integer_t iu, integer_t& m, VectorW& w,
            MatrixZ& z, const integer_t nzc, VectorISUPPZ& isuppz,
            logical_t& tryrac, integer_t& info, detail::workspace2< WORK,
            IWORK > work ) {
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::vector_traits<
                VectorD >::value_type, typename traits::vector_traits<
                VectorE >::value_type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::vector_traits<
                VectorD >::value_type, typename traits::vector_traits<
                VectorW >::value_type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::vector_traits<
                VectorD >::value_type, typename traits::matrix_traits<
                MatrixZ >::value_type >::value) );
        BOOST_ASSERT( jobz == 'N' || jobz == 'V' );
        BOOST_ASSERT( range == 'A' || range == 'V' || range == 'I' );
        BOOST_ASSERT( n >= 0 );
        BOOST_ASSERT( traits::vector_size(d) >= n );
        BOOST_ASSERT( traits::vector_size(e) >= n );
        BOOST_ASSERT( traits::vector_size(w) >= n );
        BOOST_ASSERT( traits::vector_size(work.select(real_type())) >=
                min_size_work( n, jobz ));
        BOOST_ASSERT( traits::vector_size(work.select(integer_t())) >=
                min_size_iwork( n, jobz ));
        detail::stemr( jobz, range, n, traits::vector_storage(d),
                traits::vector_storage(e), vl, vu, il, iu, m,
                traits::vector_storage(w), traits::matrix_storage(z),
                traits::leading_dimension(z), nzc,
                traits::vector_storage(isuppz), tryrac,
                traits::vector_storage(work.select(real_type())),
                traits::vector_size(work.select(real_type())),
                traits::vector_storage(work.select(integer_t())),
                traits::vector_size(work.select(integer_t())), info );
    }

    // minimal workspace specialization
    template< typename VectorD, typename VectorE, typename VectorW,
            typename MatrixZ, typename VectorISUPPZ >
    static void invoke( const char jobz, const char range, const integer_t n,
            VectorD& d, VectorE& e, const real_type vl, const real_type vu,
            const integer_t il, const integer_t iu, integer_t& m, VectorW& w,
            MatrixZ& z, const integer_t nzc, VectorISUPPZ& isuppz,
            logical_t& tryrac, integer_t& info, minimal_workspace work ) {
        traits::detail::array< real_type > tmp_work( min_size_work( n,
                jobz ) );
        traits::detail::array< integer_t > tmp_iwork( min_size_iwork( n,
                jobz ) );
        invoke( jobz, range, n, d, e, vl, vu, il, iu, m, w, z, nzc, isuppz,
                tryrac, info, workspace( tmp_work, tmp_iwork ) );
    }

    // optimal workspace specialization
    template< typename VectorD, typename VectorE, typename VectorW,
            typename MatrixZ, typename VectorISUPPZ >
    static void invoke( const char jobz, const char range, const integer_t n,
            VectorD& d, VectorE& e, const real_type vl, const real_type vu,
            const integer_t il, const integer_t iu, integer_t& m, VectorW& w,
            MatrixZ& z, const integer_t nzc, VectorISUPPZ& isuppz,
            logical_t& tryrac, integer_t& info, optimal_workspace work ) {
        real_type opt_size_work;
        integer_t opt_size_iwork;
        detail::stemr( jobz, range, n, traits::vector_storage(d),
                traits::vector_storage(e), vl, vu, il, iu, m,
                traits::vector_storage(w), traits::matrix_storage(z),
                traits::leading_dimension(z), nzc,
                traits::vector_storage(isuppz), tryrac, &opt_size_work, -1,
                &opt_size_iwork, -1, info );
        traits::detail::array< real_type > tmp_work(
                traits::detail::to_int( opt_size_work ) );
        traits::detail::array< integer_t > tmp_iwork( opt_size_iwork );
        invoke( jobz, range, n, d, e, vl, vu, il, iu, m, w, z, nzc, isuppz,
                tryrac, info, workspace( tmp_work, tmp_iwork ) );
    }

    static integer_t min_size_work( const integer_t n, const char jobz ) {
        if ( jobz == 'V' ) {
            return std::max( 1, 18*n );
        } else {
            return std::max( 1, 12*n );
        }
    }

    static integer_t min_size_iwork( const integer_t n, const char jobz ) {
        if ( jobz == 'V' ) {
            return std::max( 1, 10*n );
        } else {
            return std::max( 1, 8*n );
        }
    }
};

// complex specialization
template< typename ValueType >
struct stemr_impl< ValueType, typename boost::enable_if< traits::is_complex<ValueType> >::type > {

    typedef ValueType value_type;
    typedef typename traits::type_traits<ValueType>::real_type real_type;

    // user-defined workspace specialization
    template< typename VectorD, typename VectorE, typename VectorW,
            typename MatrixZ, typename VectorISUPPZ, typename WORK,
            typename IWORK >
    static void invoke( const char jobz, const char range, const integer_t n,
            VectorD& d, VectorE& e, const real_type vl, const real_type vu,
            const integer_t il, const integer_t iu, integer_t& m, VectorW& w,
            MatrixZ& z, const integer_t nzc, VectorISUPPZ& isuppz,
            logical_t& tryrac, integer_t& info, detail::workspace2< WORK,
            IWORK > work ) {
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::vector_traits<
                VectorD >::value_type, typename traits::vector_traits<
                VectorE >::value_type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::vector_traits<
                VectorD >::value_type, typename traits::vector_traits<
                VectorW >::value_type >::value) );
        BOOST_ASSERT( jobz == 'N' || jobz == 'V' );
        BOOST_ASSERT( range == 'A' || range == 'V' || range == 'I' );
        BOOST_ASSERT( n >= 0 );
        BOOST_ASSERT( traits::vector_size(d) >= n );
        BOOST_ASSERT( traits::vector_size(e) >= n );
        BOOST_ASSERT( traits::vector_size(w) >= n );
        BOOST_ASSERT( traits::vector_size(work.select(real_type())) >=
                min_size_work( n, jobz ));
        BOOST_ASSERT( traits::vector_size(work.select(integer_t())) >=
                min_size_iwork( n, jobz ));
        detail::stemr( jobz, range, n, traits::vector_storage(d),
                traits::vector_storage(e), vl, vu, il, iu, m,
                traits::vector_storage(w), traits::matrix_storage(z),
                traits::leading_dimension(z), nzc,
                traits::vector_storage(isuppz), tryrac,
                traits::vector_storage(work.select(real_type())),
                traits::vector_size(work.select(real_type())),
                traits::vector_storage(work.select(integer_t())),
                traits::vector_size(work.select(integer_t())), info );
    }

    // minimal workspace specialization
    template< typename VectorD, typename VectorE, typename VectorW,
            typename MatrixZ, typename VectorISUPPZ >
    static void invoke( const char jobz, const char range, const integer_t n,
            VectorD& d, VectorE& e, const real_type vl, const real_type vu,
            const integer_t il, const integer_t iu, integer_t& m, VectorW& w,
            MatrixZ& z, const integer_t nzc, VectorISUPPZ& isuppz,
            logical_t& tryrac, integer_t& info, minimal_workspace work ) {
        traits::detail::array< real_type > tmp_work( min_size_work( n,
                jobz ) );
        traits::detail::array< integer_t > tmp_iwork( min_size_iwork( n,
                jobz ) );
        invoke( jobz, range, n, d, e, vl, vu, il, iu, m, w, z, nzc, isuppz,
                tryrac, info, workspace( tmp_work, tmp_iwork ) );
    }

    // optimal workspace specialization
    template< typename VectorD, typename VectorE, typename VectorW,
            typename MatrixZ, typename VectorISUPPZ >
    static void invoke( const char jobz, const char range, const integer_t n,
            VectorD& d, VectorE& e, const real_type vl, const real_type vu,
            const integer_t il, const integer_t iu, integer_t& m, VectorW& w,
            MatrixZ& z, const integer_t nzc, VectorISUPPZ& isuppz,
            logical_t& tryrac, integer_t& info, optimal_workspace work ) {
        real_type opt_size_work;
        integer_t opt_size_iwork;
        detail::stemr( jobz, range, n, traits::vector_storage(d),
                traits::vector_storage(e), vl, vu, il, iu, m,
                traits::vector_storage(w), traits::matrix_storage(z),
                traits::leading_dimension(z), nzc,
                traits::vector_storage(isuppz), tryrac, &opt_size_work, -1,
                &opt_size_iwork, -1, info );
        traits::detail::array< real_type > tmp_work(
                traits::detail::to_int( opt_size_work ) );
        traits::detail::array< integer_t > tmp_iwork( opt_size_iwork );
        invoke( jobz, range, n, d, e, vl, vu, il, iu, m, w, z, nzc, isuppz,
                tryrac, info, workspace( tmp_work, tmp_iwork ) );
    }

    static integer_t min_size_work( const integer_t n, const char jobz ) {
        if ( jobz == 'V' ) {
            return std::max( 1, 18*n );
        } else {
            return std::max( 1, 12*n );
        }
    }

    static integer_t min_size_iwork( const integer_t n, const char jobz ) {
        if ( jobz == 'V' ) {
            return std::max( 1, 10*n );
        } else {
            return std::max( 1, 8*n );
        }
    }
};


// template function to call stemr
template< typename VectorD, typename VectorE, typename VectorW,
        typename MatrixZ, typename VectorISUPPZ, typename Workspace >
inline integer_t stemr( const char jobz, const char range,
        const integer_t n, VectorD& d, VectorE& e,
        const typename traits::type_traits< typename traits::matrix_traits<
        MatrixZ >::value_type >::real_type vl,
        const typename traits::type_traits< typename traits::matrix_traits<
        MatrixZ >::value_type >::real_type vu, const integer_t il,
        const integer_t iu, integer_t& m, VectorW& w, MatrixZ& z,
        const integer_t nzc, VectorISUPPZ& isuppz, logical_t& tryrac,
        Workspace work ) {
    typedef typename traits::matrix_traits< MatrixZ >::value_type value_type;
    integer_t info(0);
    stemr_impl< value_type >::invoke( jobz, range, n, d, e, vl, vu, il,
            iu, m, w, z, nzc, isuppz, tryrac, info, work );
    return info;
}

// template function to call stemr, default workspace type
template< typename VectorD, typename VectorE, typename VectorW,
        typename MatrixZ, typename VectorISUPPZ >
inline integer_t stemr( const char jobz, const char range,
        const integer_t n, VectorD& d, VectorE& e,
        const typename traits::type_traits< typename traits::matrix_traits<
        MatrixZ >::value_type >::real_type vl,
        const typename traits::type_traits< typename traits::matrix_traits<
        MatrixZ >::value_type >::real_type vu, const integer_t il,
        const integer_t iu, integer_t& m, VectorW& w, MatrixZ& z,
        const integer_t nzc, VectorISUPPZ& isuppz, logical_t& tryrac ) {
    typedef typename traits::matrix_traits< MatrixZ >::value_type value_type;
    integer_t info(0);
    stemr_impl< value_type >::invoke( jobz, range, n, d, e, vl, vu, il,
            iu, m, w, z, nzc, isuppz, tryrac, info, optimal_workspace() );
    return info;
}

}}}} // namespace boost::numeric::bindings::lapack

#endif
