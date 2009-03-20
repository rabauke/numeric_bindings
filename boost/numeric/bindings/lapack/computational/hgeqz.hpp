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

#ifndef BOOST_NUMERIC_BINDINGS_LAPACK_COMPUTATIONAL_HGEQZ_HPP
#define BOOST_NUMERIC_BINDINGS_LAPACK_COMPUTATIONAL_HGEQZ_HPP

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
    inline void hgeqz( char const job, char const compq, char const compz,
            integer_t const n, integer_t const ilo, integer_t const ihi,
            float* h, integer_t const ldh, float* t, integer_t const ldt,
            float* alphar, float* alphai, float* beta, float* q,
            integer_t const ldq, float* z, integer_t const ldz, float* work,
            integer_t const lwork, integer_t& info ) {
        LAPACK_SHGEQZ( &job, &compq, &compz, &n, &ilo, &ihi, h, &ldh, t, &ldt,
                alphar, alphai, beta, q, &ldq, z, &ldz, work, &lwork, &info );
    }
    inline void hgeqz( char const job, char const compq, char const compz,
            integer_t const n, integer_t const ilo, integer_t const ihi,
            double* h, integer_t const ldh, double* t, integer_t const ldt,
            double* alphar, double* alphai, double* beta, double* q,
            integer_t const ldq, double* z, integer_t const ldz, double* work,
            integer_t const lwork, integer_t& info ) {
        LAPACK_DHGEQZ( &job, &compq, &compz, &n, &ilo, &ihi, h, &ldh, t, &ldt,
                alphar, alphai, beta, q, &ldq, z, &ldz, work, &lwork, &info );
    }
    inline void hgeqz( char const job, char const compq, char const compz,
            integer_t const n, integer_t const ilo, integer_t const ihi,
            traits::complex_f* h, integer_t const ldh, traits::complex_f* t,
            integer_t const ldt, traits::complex_f* alpha,
            traits::complex_f* beta, traits::complex_f* q,
            integer_t const ldq, traits::complex_f* z, integer_t const ldz,
            traits::complex_f* work, integer_t const lwork, float* rwork,
            integer_t& info ) {
        LAPACK_CHGEQZ( &job, &compq, &compz, &n, &ilo, &ihi,
                traits::complex_ptr(h), &ldh, traits::complex_ptr(t), &ldt,
                traits::complex_ptr(alpha), traits::complex_ptr(beta),
                traits::complex_ptr(q), &ldq, traits::complex_ptr(z), &ldz,
                traits::complex_ptr(work), &lwork, rwork, &info );
    }
    inline void hgeqz( char const job, char const compq, char const compz,
            integer_t const n, integer_t const ilo, integer_t const ihi,
            traits::complex_d* h, integer_t const ldh, traits::complex_d* t,
            integer_t const ldt, traits::complex_d* alpha,
            traits::complex_d* beta, traits::complex_d* q,
            integer_t const ldq, traits::complex_d* z, integer_t const ldz,
            traits::complex_d* work, integer_t const lwork, double* rwork,
            integer_t& info ) {
        LAPACK_ZHGEQZ( &job, &compq, &compz, &n, &ilo, &ihi,
                traits::complex_ptr(h), &ldh, traits::complex_ptr(t), &ldt,
                traits::complex_ptr(alpha), traits::complex_ptr(beta),
                traits::complex_ptr(q), &ldq, traits::complex_ptr(z), &ldz,
                traits::complex_ptr(work), &lwork, rwork, &info );
    }
}

// value-type based template
template< typename ValueType, typename Enable = void >
struct hgeqz_impl{};

// real specialization
template< typename ValueType >
struct hgeqz_impl< ValueType, typename boost::enable_if< traits::is_real<ValueType> >::type > {

    typedef ValueType value_type;
    typedef typename traits::type_traits<ValueType>::real_type real_type;

    // user-defined workspace specialization
    template< typename MatrixH, typename MatrixT, typename VectorALPHAR,
            typename VectorALPHAI, typename VectorBETA, typename MatrixQ,
            typename MatrixZ, typename WORK >
    static void invoke( char const job, char const compq, char const compz,
            integer_t const ilo, MatrixH& h, MatrixT& t, VectorALPHAR& alphar,
            VectorALPHAI& alphai, VectorBETA& beta, MatrixQ& q, MatrixZ& z,
            integer_t& info, detail::workspace1< WORK > work ) {
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixH >::value_type, typename traits::matrix_traits<
                MatrixT >::value_type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixH >::value_type, typename traits::vector_traits<
                VectorALPHAR >::value_type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixH >::value_type, typename traits::vector_traits<
                VectorALPHAI >::value_type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixH >::value_type, typename traits::vector_traits<
                VectorBETA >::value_type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixH >::value_type, typename traits::matrix_traits<
                MatrixQ >::value_type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixH >::value_type, typename traits::matrix_traits<
                MatrixZ >::value_type >::value) );
        BOOST_ASSERT( job == 'E' || job == 'S' );
        BOOST_ASSERT( compq == 'N' || compq == 'I' || compq == 'V' );
        BOOST_ASSERT( compz == 'N' || compz == 'I' || compz == 'V' );
        BOOST_ASSERT( traits::matrix_num_columns(h) >= 0 );
        BOOST_ASSERT( traits::vector_size(alphar) >=
                traits::matrix_num_columns(h) );
        BOOST_ASSERT( traits::vector_size(beta) >=
                traits::matrix_num_columns(h) );
        BOOST_ASSERT( traits::vector_size(work.select(real_type())) >=
                min_size_work( traits::matrix_num_columns(h) ));
        detail::hgeqz( job, compq, compz, traits::matrix_num_columns(h), ilo,
                traits::matrix_num_columns(h), traits::matrix_storage(h),
                traits::leading_dimension(h), traits::matrix_storage(t),
                traits::leading_dimension(t), traits::vector_storage(alphar),
                traits::vector_storage(alphai), traits::vector_storage(beta),
                traits::matrix_storage(q), traits::leading_dimension(q),
                traits::matrix_storage(z), traits::leading_dimension(z),
                traits::vector_storage(work.select(real_type())),
                traits::vector_size(work.select(real_type())), info );
    }

    // minimal workspace specialization
    template< typename MatrixH, typename MatrixT, typename VectorALPHAR,
            typename VectorALPHAI, typename VectorBETA, typename MatrixQ,
            typename MatrixZ >
    static void invoke( char const job, char const compq, char const compz,
            integer_t const ilo, MatrixH& h, MatrixT& t, VectorALPHAR& alphar,
            VectorALPHAI& alphai, VectorBETA& beta, MatrixQ& q, MatrixZ& z,
            integer_t& info, minimal_workspace work ) {
        traits::detail::array< real_type > tmp_work( min_size_work(
                traits::matrix_num_columns(h) ) );
        invoke( job, compq, compz, ilo, h, t, alphar, alphai, beta, q, z,
                info, workspace( tmp_work ) );
    }

    // optimal workspace specialization
    template< typename MatrixH, typename MatrixT, typename VectorALPHAR,
            typename VectorALPHAI, typename VectorBETA, typename MatrixQ,
            typename MatrixZ >
    static void invoke( char const job, char const compq, char const compz,
            integer_t const ilo, MatrixH& h, MatrixT& t, VectorALPHAR& alphar,
            VectorALPHAI& alphai, VectorBETA& beta, MatrixQ& q, MatrixZ& z,
            integer_t& info, optimal_workspace work ) {
        real_type opt_size_work;
        detail::hgeqz( job, compq, compz, traits::matrix_num_columns(h),
                ilo, traits::matrix_num_columns(h), traits::matrix_storage(h),
                traits::leading_dimension(h), traits::matrix_storage(t),
                traits::leading_dimension(t), traits::vector_storage(alphar),
                traits::vector_storage(alphai), traits::vector_storage(beta),
                traits::matrix_storage(q), traits::leading_dimension(q),
                traits::matrix_storage(z), traits::leading_dimension(z),
                &opt_size_work, -1, info );
        traits::detail::array< real_type > tmp_work(
                traits::detail::to_int( opt_size_work ) );
        invoke( job, compq, compz, ilo, h, t, alphar, alphai, beta, q, z,
                info, workspace( tmp_work ) );
    }

    static integer_t min_size_work( integer_t const n ) {
        return std::max( 1, n );
    }
};

// complex specialization
template< typename ValueType >
struct hgeqz_impl< ValueType, typename boost::enable_if< traits::is_complex<ValueType> >::type > {

    typedef ValueType value_type;
    typedef typename traits::type_traits<ValueType>::real_type real_type;

    // user-defined workspace specialization
    template< typename MatrixH, typename MatrixT, typename VectorALPHA,
            typename VectorBETA, typename MatrixQ, typename MatrixZ,
            typename WORK, typename RWORK >
    static void invoke( char const job, char const compq, char const compz,
            integer_t const ilo, MatrixH& h, MatrixT& t, VectorALPHA& alpha,
            VectorBETA& beta, MatrixQ& q, MatrixZ& z, integer_t& info,
            detail::workspace2< WORK, RWORK > work ) {
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixH >::value_type, typename traits::matrix_traits<
                MatrixT >::value_type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixH >::value_type, typename traits::vector_traits<
                VectorALPHA >::value_type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixH >::value_type, typename traits::vector_traits<
                VectorBETA >::value_type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixH >::value_type, typename traits::matrix_traits<
                MatrixQ >::value_type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixH >::value_type, typename traits::matrix_traits<
                MatrixZ >::value_type >::value) );
        BOOST_ASSERT( job == 'E' || job == 'S' );
        BOOST_ASSERT( compq == 'N' || compq == 'I' || compq == 'V' );
        BOOST_ASSERT( compz == 'N' || compz == 'I' || compz == 'V' );
        BOOST_ASSERT( traits::matrix_num_columns(h) >= 0 );
        BOOST_ASSERT( traits::vector_size(alpha) >=
                traits::matrix_num_columns(h) );
        BOOST_ASSERT( traits::vector_size(beta) >=
                traits::matrix_num_columns(h) );
        BOOST_ASSERT( traits::vector_size(work.select(value_type())) >=
                min_size_work( traits::matrix_num_columns(h) ));
        BOOST_ASSERT( traits::vector_size(work.select(real_type())) >=
                min_size_rwork( traits::matrix_num_columns(h) ));
        detail::hgeqz( job, compq, compz, traits::matrix_num_columns(h), ilo,
                traits::matrix_num_columns(h), traits::matrix_storage(h),
                traits::leading_dimension(h), traits::matrix_storage(t),
                traits::leading_dimension(t), traits::vector_storage(alpha),
                traits::vector_storage(beta), traits::matrix_storage(q),
                traits::leading_dimension(q), traits::matrix_storage(z),
                traits::leading_dimension(z),
                traits::vector_storage(work.select(value_type())),
                traits::vector_size(work.select(value_type())),
                traits::vector_storage(work.select(real_type())), info );
    }

    // minimal workspace specialization
    template< typename MatrixH, typename MatrixT, typename VectorALPHA,
            typename VectorBETA, typename MatrixQ, typename MatrixZ >
    static void invoke( char const job, char const compq, char const compz,
            integer_t const ilo, MatrixH& h, MatrixT& t, VectorALPHA& alpha,
            VectorBETA& beta, MatrixQ& q, MatrixZ& z, integer_t& info,
            minimal_workspace work ) {
        traits::detail::array< value_type > tmp_work( min_size_work(
                traits::matrix_num_columns(h) ) );
        traits::detail::array< real_type > tmp_rwork( min_size_rwork(
                traits::matrix_num_columns(h) ) );
        invoke( job, compq, compz, ilo, h, t, alpha, beta, q, z, info,
                workspace( tmp_work, tmp_rwork ) );
    }

    // optimal workspace specialization
    template< typename MatrixH, typename MatrixT, typename VectorALPHA,
            typename VectorBETA, typename MatrixQ, typename MatrixZ >
    static void invoke( char const job, char const compq, char const compz,
            integer_t const ilo, MatrixH& h, MatrixT& t, VectorALPHA& alpha,
            VectorBETA& beta, MatrixQ& q, MatrixZ& z, integer_t& info,
            optimal_workspace work ) {
        value_type opt_size_work;
        traits::detail::array< real_type > tmp_rwork( min_size_rwork(
                traits::matrix_num_columns(h) ) );
        detail::hgeqz( job, compq, compz, traits::matrix_num_columns(h),
                ilo, traits::matrix_num_columns(h), traits::matrix_storage(h),
                traits::leading_dimension(h), traits::matrix_storage(t),
                traits::leading_dimension(t), traits::vector_storage(alpha),
                traits::vector_storage(beta), traits::matrix_storage(q),
                traits::leading_dimension(q), traits::matrix_storage(z),
                traits::leading_dimension(z), &opt_size_work, -1,
                traits::vector_storage(tmp_rwork), info );
        traits::detail::array< value_type > tmp_work(
                traits::detail::to_int( opt_size_work ) );
        invoke( job, compq, compz, ilo, h, t, alpha, beta, q, z, info,
                workspace( tmp_work, tmp_rwork ) );
    }

    static integer_t min_size_work( integer_t const n ) {
        return std::max( 1, n );
    }

    static integer_t min_size_rwork( integer_t const n ) {
        return n;
    }
};


// template function to call hgeqz
template< typename MatrixH, typename MatrixT, typename VectorALPHAR,
        typename VectorALPHAI, typename VectorBETA, typename MatrixQ,
        typename MatrixZ, typename Workspace >
inline integer_t hgeqz( char const job, char const compq,
        char const compz, integer_t const ilo, MatrixH& h, MatrixT& t,
        VectorALPHAR& alphar, VectorALPHAI& alphai, VectorBETA& beta,
        MatrixQ& q, MatrixZ& z, Workspace work ) {
    typedef typename traits::matrix_traits< MatrixH >::value_type value_type;
    integer_t info(0);
    hgeqz_impl< value_type >::invoke( job, compq, compz, ilo, h, t,
            alphar, alphai, beta, q, z, info, work );
    return info;
}

// template function to call hgeqz, default workspace type
template< typename MatrixH, typename MatrixT, typename VectorALPHAR,
        typename VectorALPHAI, typename VectorBETA, typename MatrixQ,
        typename MatrixZ >
inline integer_t hgeqz( char const job, char const compq,
        char const compz, integer_t const ilo, MatrixH& h, MatrixT& t,
        VectorALPHAR& alphar, VectorALPHAI& alphai, VectorBETA& beta,
        MatrixQ& q, MatrixZ& z ) {
    typedef typename traits::matrix_traits< MatrixH >::value_type value_type;
    integer_t info(0);
    hgeqz_impl< value_type >::invoke( job, compq, compz, ilo, h, t,
            alphar, alphai, beta, q, z, info, optimal_workspace() );
    return info;
}
// template function to call hgeqz
template< typename MatrixH, typename MatrixT, typename VectorALPHA,
        typename VectorBETA, typename MatrixQ, typename MatrixZ,
        typename Workspace >
inline integer_t hgeqz( char const job, char const compq,
        char const compz, integer_t const ilo, MatrixH& h, MatrixT& t,
        VectorALPHA& alpha, VectorBETA& beta, MatrixQ& q, MatrixZ& z,
        Workspace work ) {
    typedef typename traits::matrix_traits< MatrixH >::value_type value_type;
    integer_t info(0);
    hgeqz_impl< value_type >::invoke( job, compq, compz, ilo, h, t,
            alpha, beta, q, z, info, work );
    return info;
}

// template function to call hgeqz, default workspace type
template< typename MatrixH, typename MatrixT, typename VectorALPHA,
        typename VectorBETA, typename MatrixQ, typename MatrixZ >
inline integer_t hgeqz( char const job, char const compq,
        char const compz, integer_t const ilo, MatrixH& h, MatrixT& t,
        VectorALPHA& alpha, VectorBETA& beta, MatrixQ& q, MatrixZ& z ) {
    typedef typename traits::matrix_traits< MatrixH >::value_type value_type;
    integer_t info(0);
    hgeqz_impl< value_type >::invoke( job, compq, compz, ilo, h, t,
            alpha, beta, q, z, info, optimal_workspace() );
    return info;
}

}}}} // namespace boost::numeric::bindings::lapack

#endif
