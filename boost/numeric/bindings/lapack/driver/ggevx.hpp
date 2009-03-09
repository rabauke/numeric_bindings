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

#ifndef BOOST_NUMERIC_BINDINGS_LAPACK_GGEVX_HPP
#define BOOST_NUMERIC_BINDINGS_LAPACK_GGEVX_HPP

#include <boost/assert.hpp>
#include <boost/numeric/bindings/lapack/lapack.h>
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
    inline void ggevx( char const balanc, char const jobvl, char const jobvr,
            char const sense, integer_t const n, float* a,
            integer_t const lda, float* b, integer_t const ldb, float* alphar,
            float* alphai, float* beta, float* vl, integer_t const ldvl,
            float* vr, integer_t const ldvr, integer_t& ilo, integer_t& ihi,
            float* lscale, float* rscale, float& abnrm, float& bbnrm,
            float* rconde, float* rcondv, float* work, integer_t const lwork,
            integer_t* iwork, logical_t* bwork, integer_t& info ) {
        LAPACK_SGGEVX( &balanc, &jobvl, &jobvr, &sense, &n, a, &lda, b, &ldb,
                alphar, alphai, beta, vl, &ldvl, vr, &ldvr, &ilo, &ihi,
                lscale, rscale, &abnrm, &bbnrm, rconde, rcondv, work, &lwork,
                iwork, bwork, &info );
    }
    inline void ggevx( char const balanc, char const jobvl, char const jobvr,
            char const sense, integer_t const n, double* a,
            integer_t const lda, double* b, integer_t const ldb,
            double* alphar, double* alphai, double* beta, double* vl,
            integer_t const ldvl, double* vr, integer_t const ldvr,
            integer_t& ilo, integer_t& ihi, double* lscale, double* rscale,
            double& abnrm, double& bbnrm, double* rconde, double* rcondv,
            double* work, integer_t const lwork, integer_t* iwork,
            logical_t* bwork, integer_t& info ) {
        LAPACK_DGGEVX( &balanc, &jobvl, &jobvr, &sense, &n, a, &lda, b, &ldb,
                alphar, alphai, beta, vl, &ldvl, vr, &ldvr, &ilo, &ihi,
                lscale, rscale, &abnrm, &bbnrm, rconde, rcondv, work, &lwork,
                iwork, bwork, &info );
    }
    inline void ggevx( char const balanc, char const jobvl, char const jobvr,
            char const sense, integer_t const n, traits::complex_f* a,
            integer_t const lda, traits::complex_f* b, integer_t const ldb,
            traits::complex_f* alpha, traits::complex_f* beta,
            traits::complex_f* vl, integer_t const ldvl,
            traits::complex_f* vr, integer_t const ldvr, integer_t& ilo,
            integer_t& ihi, float* lscale, float* rscale, float& abnrm,
            float& bbnrm, float* rconde, float* rcondv,
            traits::complex_f* work, integer_t const lwork, float* rwork,
            integer_t* iwork, logical_t* bwork, integer_t& info ) {
        LAPACK_CGGEVX( &balanc, &jobvl, &jobvr, &sense, &n,
                traits::complex_ptr(a), &lda, traits::complex_ptr(b), &ldb,
                traits::complex_ptr(alpha), traits::complex_ptr(beta),
                traits::complex_ptr(vl), &ldvl, traits::complex_ptr(vr),
                &ldvr, &ilo, &ihi, lscale, rscale, &abnrm, &bbnrm, rconde,
                rcondv, traits::complex_ptr(work), &lwork, rwork, iwork,
                bwork, &info );
    }
    inline void ggevx( char const balanc, char const jobvl, char const jobvr,
            char const sense, integer_t const n, traits::complex_d* a,
            integer_t const lda, traits::complex_d* b, integer_t const ldb,
            traits::complex_d* alpha, traits::complex_d* beta,
            traits::complex_d* vl, integer_t const ldvl,
            traits::complex_d* vr, integer_t const ldvr, integer_t& ilo,
            integer_t& ihi, double* lscale, double* rscale, double& abnrm,
            double& bbnrm, double* rconde, double* rcondv,
            traits::complex_d* work, integer_t const lwork, double* rwork,
            integer_t* iwork, logical_t* bwork, integer_t& info ) {
        LAPACK_ZGGEVX( &balanc, &jobvl, &jobvr, &sense, &n,
                traits::complex_ptr(a), &lda, traits::complex_ptr(b), &ldb,
                traits::complex_ptr(alpha), traits::complex_ptr(beta),
                traits::complex_ptr(vl), &ldvl, traits::complex_ptr(vr),
                &ldvr, &ilo, &ihi, lscale, rscale, &abnrm, &bbnrm, rconde,
                rcondv, traits::complex_ptr(work), &lwork, rwork, iwork,
                bwork, &info );
    }
}

// value-type based template
template< typename ValueType, typename Enable = void >
struct ggevx_impl{};

// real specialization
template< typename ValueType >
struct ggevx_impl< ValueType, typename boost::enable_if< traits::is_real<ValueType> >::type > {

    typedef ValueType value_type;
    typedef typename traits::type_traits<ValueType>::real_type real_type;

    // user-defined workspace specialization
    template< typename MatrixA, typename MatrixB, typename VectorALPHAR,
            typename VectorALPHAI, typename VectorBETA, typename MatrixVL,
            typename MatrixVR, typename VectorLSCALE, typename VectorRSCALE,
            typename VectorRCONDE, typename VectorRCONDV, typename WORK,
            typename IWORK, typename BWORK >
    static void compute( char const balanc, char const jobvl,
            char const jobvr, char const sense, MatrixA& a, MatrixB& b,
            VectorALPHAR& alphar, VectorALPHAI& alphai, VectorBETA& beta,
            MatrixVL& vl, MatrixVR& vr, integer_t& ilo, integer_t& ihi,
            VectorLSCALE& lscale, VectorRSCALE& rscale, real_type& abnrm,
            real_type& bbnrm, VectorRCONDE& rconde, VectorRCONDV& rcondv,
            integer_t& info, detail::workspace3< WORK, IWORK, BWORK > work ) {
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixA >::value_type, typename traits::matrix_traits<
                MatrixB >::value_type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixA >::value_type, typename traits::vector_traits<
                VectorALPHAR >::value_type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixA >::value_type, typename traits::vector_traits<
                VectorALPHAI >::value_type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixA >::value_type, typename traits::vector_traits<
                VectorBETA >::value_type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixA >::value_type, typename traits::matrix_traits<
                MatrixVL >::value_type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixA >::value_type, typename traits::matrix_traits<
                MatrixVR >::value_type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixA >::value_type, typename traits::vector_traits<
                VectorLSCALE >::value_type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixA >::value_type, typename traits::vector_traits<
                VectorRSCALE >::value_type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixA >::value_type, typename traits::vector_traits<
                VectorRCONDE >::value_type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixA >::value_type, typename traits::vector_traits<
                VectorRCONDV >::value_type >::value) );
        BOOST_ASSERT( balanc == 'N' || balanc == 'P' || balanc == 'S' ||
                balanc == 'B' );
        BOOST_ASSERT( jobvl == 'N' || jobvl == 'V' );
        BOOST_ASSERT( jobvr == 'N' || jobvr == 'V' );
        BOOST_ASSERT( sense == 'N' || sense == 'E' || sense == 'V' ||
                sense == 'B' );
        BOOST_ASSERT( traits::matrix_num_columns(a) >= 0 );
        BOOST_ASSERT( traits::leading_dimension(a) >= std::max(1,
                traits::matrix_num_columns(a)) );
        BOOST_ASSERT( traits::leading_dimension(b) >= std::max(1,
                traits::matrix_num_columns(a)) );
        BOOST_ASSERT( traits::vector_size(alphar) >=
                traits::matrix_num_columns(a) );
        BOOST_ASSERT( traits::vector_size(alphai) >=
                traits::matrix_num_columns(a) );
        BOOST_ASSERT( traits::vector_size(work.select(real_type())) >=
                min_size_work( balanc, jobvl, jobvr, sense,
                traits::matrix_num_columns(a) ));
        BOOST_ASSERT( traits::vector_size(work.select(integer_t())) >=
                min_size_iwork( sense, traits::matrix_num_columns(a) ));
        BOOST_ASSERT( traits::vector_size(work.select(bool())) >=
                min_size_bwork( sense, traits::matrix_num_columns(a) ));
        detail::ggevx( balanc, jobvl, jobvr, sense,
                traits::matrix_num_columns(a), traits::matrix_storage(a),
                traits::leading_dimension(a), traits::matrix_storage(b),
                traits::leading_dimension(b), traits::vector_storage(alphar),
                traits::vector_storage(alphai), traits::vector_storage(beta),
                traits::matrix_storage(vl), traits::leading_dimension(vl),
                traits::matrix_storage(vr), traits::leading_dimension(vr),
                ilo, ihi, traits::vector_storage(lscale),
                traits::vector_storage(rscale), abnrm, bbnrm,
                traits::vector_storage(rconde),
                traits::vector_storage(rcondv),
                traits::vector_storage(work.select(real_type())),
                traits::vector_size(work.select(real_type())),
                traits::vector_storage(work.select(integer_t())),
                traits::vector_storage(work.select(bool())), info );
    }

    // minimal workspace specialization
    template< typename MatrixA, typename MatrixB, typename VectorALPHAR,
            typename VectorALPHAI, typename VectorBETA, typename MatrixVL,
            typename MatrixVR, typename VectorLSCALE, typename VectorRSCALE,
            typename VectorRCONDE, typename VectorRCONDV >
    static void compute( char const balanc, char const jobvl,
            char const jobvr, char const sense, MatrixA& a, MatrixB& b,
            VectorALPHAR& alphar, VectorALPHAI& alphai, VectorBETA& beta,
            MatrixVL& vl, MatrixVR& vr, integer_t& ilo, integer_t& ihi,
            VectorLSCALE& lscale, VectorRSCALE& rscale, real_type& abnrm,
            real_type& bbnrm, VectorRCONDE& rconde, VectorRCONDV& rcondv,
            integer_t& info, minimal_workspace work ) {
        traits::detail::array< real_type > tmp_work( min_size_work( balanc,
                jobvl, jobvr, sense, traits::matrix_num_columns(a) ) );
        traits::detail::array< integer_t > tmp_iwork( min_size_iwork( sense,
                traits::matrix_num_columns(a) ) );
        traits::detail::array< bool > tmp_bwork( min_size_bwork( sense,
                traits::matrix_num_columns(a) ) );
        compute( balanc, jobvl, jobvr, sense, a, b, alphar, alphai, beta, vl,
                vr, ilo, ihi, lscale, rscale, abnrm, bbnrm, rconde, rcondv,
                info, workspace( tmp_work, tmp_iwork, tmp_bwork ) );
    }

    // optimal workspace specialization
    template< typename MatrixA, typename MatrixB, typename VectorALPHAR,
            typename VectorALPHAI, typename VectorBETA, typename MatrixVL,
            typename MatrixVR, typename VectorLSCALE, typename VectorRSCALE,
            typename VectorRCONDE, typename VectorRCONDV >
    static void compute( char const balanc, char const jobvl,
            char const jobvr, char const sense, MatrixA& a, MatrixB& b,
            VectorALPHAR& alphar, VectorALPHAI& alphai, VectorBETA& beta,
            MatrixVL& vl, MatrixVR& vr, integer_t& ilo, integer_t& ihi,
            VectorLSCALE& lscale, VectorRSCALE& rscale, real_type& abnrm,
            real_type& bbnrm, VectorRCONDE& rconde, VectorRCONDV& rcondv,
            integer_t& info, optimal_workspace work ) {
        real_type opt_size_work;
        traits::detail::array< integer_t > tmp_iwork( min_size_iwork( sense,
                traits::matrix_num_columns(a) ) );
        traits::detail::array< bool > tmp_bwork( min_size_bwork( sense,
                traits::matrix_num_columns(a) ) );
        detail::ggevx( balanc, jobvl, jobvr, sense,
                traits::matrix_num_columns(a), traits::matrix_storage(a),
                traits::leading_dimension(a), traits::matrix_storage(b),
                traits::leading_dimension(b), traits::vector_storage(alphar),
                traits::vector_storage(alphai), traits::vector_storage(beta),
                traits::matrix_storage(vl), traits::leading_dimension(vl),
                traits::matrix_storage(vr), traits::leading_dimension(vr),
                ilo, ihi, traits::vector_storage(lscale),
                traits::vector_storage(rscale), abnrm, bbnrm,
                traits::vector_storage(rconde),
                traits::vector_storage(rcondv), &opt_size_work, -1,
                traits::vector_storage(tmp_iwork),
                traits::vector_storage(tmp_bwork), info );
        traits::detail::array< real_type > tmp_work(
                traits::detail::to_int( opt_size_work ) );
        compute( balanc, jobvl, jobvr, sense, a, b, alphar, alphai, beta, vl,
                vr, ilo, ihi, lscale, rscale, abnrm, bbnrm, rconde, rcondv,
                info, workspace( tmp_work, tmp_iwork, tmp_bwork ) );
    }

    static integer_t min_size_work( char const balanc, char const jobvl,
            char const jobvr, char const sense, integer_t const n ) {
        if ( balanc == 'S' || balanc == 'B' || jobvl == 'V' || jobvr == 'V' )
            return std::max( 1, 6*n );
        if ( sense == 'E' )
            return std::max( 1, 10*n );
        if ( sense == 'V' || sense == 'B' )
            return 2*n*n + 8*n + 16;
        return std::max( 1, 2*n );
    }

    static integer_t min_size_iwork( char const sense, integer_t const n ) {
        if ( sense == 'E' )
          return 0;
        else
          return n+6;
    }

    static integer_t min_size_bwork( char const sense, integer_t const n ) {
        if ( sense == 'N' )
          return 0;
        else
          return n;
    }
};

// complex specialization
template< typename ValueType >
struct ggevx_impl< ValueType, typename boost::enable_if< traits::is_complex<ValueType> >::type > {

    typedef ValueType value_type;
    typedef typename traits::type_traits<ValueType>::real_type real_type;

    // user-defined workspace specialization
    template< typename MatrixA, typename MatrixB, typename VectorALPHA,
            typename VectorBETA, typename MatrixVL, typename MatrixVR,
            typename VectorLSCALE, typename VectorRSCALE,
            typename VectorRCONDE, typename VectorRCONDV, typename WORK,
            typename RWORK, typename IWORK, typename BWORK >
    static void compute( char const balanc, char const jobvl,
            char const jobvr, char const sense, MatrixA& a, MatrixB& b,
            VectorALPHA& alpha, VectorBETA& beta, MatrixVL& vl, MatrixVR& vr,
            integer_t& ilo, integer_t& ihi, VectorLSCALE& lscale,
            VectorRSCALE& rscale, real_type& abnrm, real_type& bbnrm,
            VectorRCONDE& rconde, VectorRCONDV& rcondv, integer_t& info,
            detail::workspace4< WORK, RWORK, IWORK, BWORK > work ) {
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::vector_traits<
                VectorLSCALE >::value_type, typename traits::vector_traits<
                VectorRSCALE >::value_type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::vector_traits<
                VectorLSCALE >::value_type, typename traits::vector_traits<
                VectorRCONDE >::value_type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::vector_traits<
                VectorLSCALE >::value_type, typename traits::vector_traits<
                VectorRCONDV >::value_type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixA >::value_type, typename traits::matrix_traits<
                MatrixB >::value_type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixA >::value_type, typename traits::vector_traits<
                VectorALPHA >::value_type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixA >::value_type, typename traits::vector_traits<
                VectorBETA >::value_type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixA >::value_type, typename traits::matrix_traits<
                MatrixVL >::value_type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixA >::value_type, typename traits::matrix_traits<
                MatrixVR >::value_type >::value) );
        BOOST_ASSERT( balanc == 'N' || balanc == 'P' || balanc == 'S' ||
                balanc == 'B' );
        BOOST_ASSERT( jobvl == 'N' || jobvl == 'V' );
        BOOST_ASSERT( jobvr == 'N' || jobvr == 'V' );
        BOOST_ASSERT( sense == 'N' || sense == 'E' || sense == 'V' ||
                sense == 'B' );
        BOOST_ASSERT( traits::matrix_num_columns(a) >= 0 );
        BOOST_ASSERT( traits::leading_dimension(a) >= std::max(1,
                traits::matrix_num_columns(a)) );
        BOOST_ASSERT( traits::leading_dimension(b) >= std::max(1,
                traits::matrix_num_columns(a)) );
        BOOST_ASSERT( traits::vector_size(alpha) >=
                traits::matrix_num_columns(a) );
        BOOST_ASSERT( traits::vector_size(beta) >=
                traits::matrix_num_columns(a) );
        BOOST_ASSERT( traits::vector_size(work.select(value_type())) >=
                min_size_work( sense, traits::matrix_num_columns(a) ));
        BOOST_ASSERT( traits::vector_size(work.select(real_type())) >=
                min_size_rwork( balanc, traits::matrix_num_columns(a) ));
        BOOST_ASSERT( traits::vector_size(work.select(integer_t())) >=
                min_size_iwork( sense, traits::matrix_num_columns(a) ));
        BOOST_ASSERT( traits::vector_size(work.select(bool())) >=
                min_size_bwork( sense, traits::matrix_num_columns(a) ));
        detail::ggevx( balanc, jobvl, jobvr, sense,
                traits::matrix_num_columns(a), traits::matrix_storage(a),
                traits::leading_dimension(a), traits::matrix_storage(b),
                traits::leading_dimension(b), traits::vector_storage(alpha),
                traits::vector_storage(beta), traits::matrix_storage(vl),
                traits::leading_dimension(vl), traits::matrix_storage(vr),
                traits::leading_dimension(vr), ilo, ihi,
                traits::vector_storage(lscale),
                traits::vector_storage(rscale), abnrm, bbnrm,
                traits::vector_storage(rconde),
                traits::vector_storage(rcondv),
                traits::vector_storage(work.select(value_type())),
                traits::vector_size(work.select(value_type())),
                traits::vector_storage(work.select(real_type())),
                traits::vector_storage(work.select(integer_t())),
                traits::vector_storage(work.select(bool())), info );
    }

    // minimal workspace specialization
    template< typename MatrixA, typename MatrixB, typename VectorALPHA,
            typename VectorBETA, typename MatrixVL, typename MatrixVR,
            typename VectorLSCALE, typename VectorRSCALE,
            typename VectorRCONDE, typename VectorRCONDV >
    static void compute( char const balanc, char const jobvl,
            char const jobvr, char const sense, MatrixA& a, MatrixB& b,
            VectorALPHA& alpha, VectorBETA& beta, MatrixVL& vl, MatrixVR& vr,
            integer_t& ilo, integer_t& ihi, VectorLSCALE& lscale,
            VectorRSCALE& rscale, real_type& abnrm, real_type& bbnrm,
            VectorRCONDE& rconde, VectorRCONDV& rcondv, integer_t& info,
            minimal_workspace work ) {
        traits::detail::array< value_type > tmp_work( min_size_work( sense,
                traits::matrix_num_columns(a) ) );
        traits::detail::array< real_type > tmp_rwork( min_size_rwork( balanc,
                traits::matrix_num_columns(a) ) );
        traits::detail::array< integer_t > tmp_iwork( min_size_iwork( sense,
                traits::matrix_num_columns(a) ) );
        traits::detail::array< bool > tmp_bwork( min_size_bwork( sense,
                traits::matrix_num_columns(a) ) );
        compute( balanc, jobvl, jobvr, sense, a, b, alpha, beta, vl, vr, ilo,
                ihi, lscale, rscale, abnrm, bbnrm, rconde, rcondv, info,
                workspace( tmp_work, tmp_rwork, tmp_iwork, tmp_bwork ) );
    }

    // optimal workspace specialization
    template< typename MatrixA, typename MatrixB, typename VectorALPHA,
            typename VectorBETA, typename MatrixVL, typename MatrixVR,
            typename VectorLSCALE, typename VectorRSCALE,
            typename VectorRCONDE, typename VectorRCONDV >
    static void compute( char const balanc, char const jobvl,
            char const jobvr, char const sense, MatrixA& a, MatrixB& b,
            VectorALPHA& alpha, VectorBETA& beta, MatrixVL& vl, MatrixVR& vr,
            integer_t& ilo, integer_t& ihi, VectorLSCALE& lscale,
            VectorRSCALE& rscale, real_type& abnrm, real_type& bbnrm,
            VectorRCONDE& rconde, VectorRCONDV& rcondv, integer_t& info,
            optimal_workspace work ) {
        value_type opt_size_work;
        traits::detail::array< real_type > tmp_rwork( min_size_rwork( balanc,
                traits::matrix_num_columns(a) ) );
        traits::detail::array< integer_t > tmp_iwork( min_size_iwork( sense,
                traits::matrix_num_columns(a) ) );
        traits::detail::array< bool > tmp_bwork( min_size_bwork( sense,
                traits::matrix_num_columns(a) ) );
        detail::ggevx( balanc, jobvl, jobvr, sense,
                traits::matrix_num_columns(a), traits::matrix_storage(a),
                traits::leading_dimension(a), traits::matrix_storage(b),
                traits::leading_dimension(b), traits::vector_storage(alpha),
                traits::vector_storage(beta), traits::matrix_storage(vl),
                traits::leading_dimension(vl), traits::matrix_storage(vr),
                traits::leading_dimension(vr), ilo, ihi,
                traits::vector_storage(lscale),
                traits::vector_storage(rscale), abnrm, bbnrm,
                traits::vector_storage(rconde),
                traits::vector_storage(rcondv), &opt_size_work, -1,
                traits::vector_storage(tmp_rwork),
                traits::vector_storage(tmp_iwork),
                traits::vector_storage(tmp_bwork), info );
        traits::detail::array< value_type > tmp_work(
                traits::detail::to_int( opt_size_work ) );
        compute( balanc, jobvl, jobvr, sense, a, b, alpha, beta, vl, vr, ilo,
                ihi, lscale, rscale, abnrm, bbnrm, rconde, rcondv, info,
                workspace( tmp_work, tmp_rwork, tmp_iwork, tmp_bwork ) );
    }

    static integer_t min_size_work( char const sense, integer_t const n ) {
        if ( sense == 'N' )
            return std::max( 1, 2*n );
        else {
            if ( sense == 'E' )
                return std::max( 1, 4*n );
            else
                return std::max( 1, 2*n*n+2*n );
        }
    }

    static integer_t min_size_rwork( char const balanc, integer_t const n ) {
        if ( balanc == 'S' || balanc == 'B' )
            return std::max( 1, 6*n );
        else
            return std::max( 1, 2*n );
    }

    static integer_t min_size_iwork( char const sense, integer_t const n ) {
        if ( sense == 'E' )
          return 0;
        else
          return n+2;
    }

    static integer_t min_size_bwork( char const sense, integer_t const n ) {
        if ( sense == 'N' )
          return 0;
        else
          return n;
    }
};


// template function to call ggevx
template< typename MatrixA, typename MatrixB, typename VectorALPHAR,
        typename VectorALPHAI, typename VectorBETA, typename MatrixVL,
        typename MatrixVR, typename VectorLSCALE, typename VectorRSCALE,
        typename VectorRCONDE, typename VectorRCONDV, typename Workspace >
inline integer_t ggevx( char const balanc, char const jobvl,
        char const jobvr, char const sense, MatrixA& a, MatrixB& b,
        VectorALPHAR& alphar, VectorALPHAI& alphai, VectorBETA& beta,
        MatrixVL& vl, MatrixVR& vr, integer_t& ilo, integer_t& ihi,
        VectorLSCALE& lscale, VectorRSCALE& rscale,
        typename traits::matrix_traits< MatrixA >::value_type& abnrm,
        typename traits::matrix_traits< MatrixA >::value_type& bbnrm,
        VectorRCONDE& rconde, VectorRCONDV& rcondv,
        Workspace work = optimal_workspace() ) {
    typedef typename traits::matrix_traits< MatrixA >::value_type value_type;
    integer_t info(0);
    ggevx_impl< value_type >::compute( balanc, jobvl, jobvr, sense, a, b,
            alphar, alphai, beta, vl, vr, ilo, ihi, lscale, rscale, abnrm,
            bbnrm, rconde, rcondv, info, work );
    return info;
}
// template function to call ggevx
template< typename MatrixA, typename MatrixB, typename VectorALPHA,
        typename VectorBETA, typename MatrixVL, typename MatrixVR,
        typename VectorLSCALE, typename VectorRSCALE, typename VectorRCONDE,
        typename VectorRCONDV, typename Workspace >
inline integer_t ggevx( char const balanc, char const jobvl,
        char const jobvr, char const sense, MatrixA& a, MatrixB& b,
        VectorALPHA& alpha, VectorBETA& beta, MatrixVL& vl, MatrixVR& vr,
        integer_t& ilo, integer_t& ihi, VectorLSCALE& lscale,
        VectorRSCALE& rscale, typename traits::matrix_traits<
        MatrixA >::value_type& abnrm, typename traits::matrix_traits<
        MatrixA >::value_type& bbnrm, VectorRCONDE& rconde,
        VectorRCONDV& rcondv, Workspace work = optimal_workspace() ) {
    typedef typename traits::matrix_traits< MatrixA >::value_type value_type;
    integer_t info(0);
    ggevx_impl< value_type >::compute( balanc, jobvl, jobvr, sense, a, b,
            alpha, beta, vl, vr, ilo, ihi, lscale, rscale, abnrm, bbnrm,
            rconde, rcondv, info, work );
    return info;
}

}}}} // namespace boost::numeric::bindings::lapack

#endif
