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

#ifndef BOOST_NUMERIC_BINDINGS_LAPACK_COMPUTATIONAL_TGSYL_HPP
#define BOOST_NUMERIC_BINDINGS_LAPACK_COMPUTATIONAL_TGSYL_HPP

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
    inline void tgsyl( char const trans, integer_t const ijob,
            integer_t const m, integer_t const n, float* a,
            integer_t const lda, float* b, integer_t const ldb, float* c,
            integer_t const ldc, float* d, integer_t const ldd, float* e,
            integer_t const lde, float* f, integer_t const ldf, float& scale,
            float& dif, float* work, integer_t const lwork, integer_t* iwork,
            integer_t& info ) {
        LAPACK_STGSYL( &trans, &ijob, &m, &n, a, &lda, b, &ldb, c, &ldc, d,
                &ldd, e, &lde, f, &ldf, &scale, &dif, work, &lwork, iwork,
                &info );
    }
    inline void tgsyl( char const trans, integer_t const ijob,
            integer_t const m, integer_t const n, double* a,
            integer_t const lda, double* b, integer_t const ldb, double* c,
            integer_t const ldc, double* d, integer_t const ldd, double* e,
            integer_t const lde, double* f, integer_t const ldf,
            double& scale, double& dif, double* work, integer_t const lwork,
            integer_t* iwork, integer_t& info ) {
        LAPACK_DTGSYL( &trans, &ijob, &m, &n, a, &lda, b, &ldb, c, &ldc, d,
                &ldd, e, &lde, f, &ldf, &scale, &dif, work, &lwork, iwork,
                &info );
    }
    inline void tgsyl( char const trans, integer_t const ijob,
            integer_t const m, integer_t const n, traits::complex_f* a,
            integer_t const lda, traits::complex_f* b, integer_t const ldb,
            traits::complex_f* c, integer_t const ldc, traits::complex_f* d,
            integer_t const ldd, traits::complex_f* e, integer_t const lde,
            traits::complex_f* f, integer_t const ldf, float& scale,
            float& dif, traits::complex_f* work, integer_t const lwork,
            integer_t* iwork, integer_t& info ) {
        LAPACK_CTGSYL( &trans, &ijob, &m, &n, traits::complex_ptr(a), &lda,
                traits::complex_ptr(b), &ldb, traits::complex_ptr(c), &ldc,
                traits::complex_ptr(d), &ldd, traits::complex_ptr(e), &lde,
                traits::complex_ptr(f), &ldf, &scale, &dif,
                traits::complex_ptr(work), &lwork, iwork, &info );
    }
    inline void tgsyl( char const trans, integer_t const ijob,
            integer_t const m, integer_t const n, traits::complex_d* a,
            integer_t const lda, traits::complex_d* b, integer_t const ldb,
            traits::complex_d* c, integer_t const ldc, traits::complex_d* d,
            integer_t const ldd, traits::complex_d* e, integer_t const lde,
            traits::complex_d* f, integer_t const ldf, double& scale,
            double& dif, traits::complex_d* work, integer_t const lwork,
            integer_t* iwork, integer_t& info ) {
        LAPACK_ZTGSYL( &trans, &ijob, &m, &n, traits::complex_ptr(a), &lda,
                traits::complex_ptr(b), &ldb, traits::complex_ptr(c), &ldc,
                traits::complex_ptr(d), &ldd, traits::complex_ptr(e), &lde,
                traits::complex_ptr(f), &ldf, &scale, &dif,
                traits::complex_ptr(work), &lwork, iwork, &info );
    }
}

// value-type based template
template< typename ValueType, typename Enable = void >
struct tgsyl_impl{};

// real specialization
template< typename ValueType >
struct tgsyl_impl< ValueType, typename boost::enable_if< traits::is_real<ValueType> >::type > {

    typedef ValueType value_type;
    typedef typename traits::type_traits<ValueType>::real_type real_type;

    // user-defined workspace specialization
    template< typename MatrixA, typename MatrixB, typename MatrixC,
            typename MatrixD, typename MatrixE, typename MatrixF,
            typename WORK, typename IWORK >
    static void compute( char const trans, integer_t const ijob,
            integer_t const m, integer_t const n, MatrixA& a, MatrixB& b,
            MatrixC& c, MatrixD& d, MatrixE& e, MatrixF& f, real_type& scale,
            real_type& dif, integer_t& info, detail::workspace2< WORK,
            IWORK > work ) {
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixA >::value_type, typename traits::matrix_traits<
                MatrixB >::value_type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixA >::value_type, typename traits::matrix_traits<
                MatrixC >::value_type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixA >::value_type, typename traits::matrix_traits<
                MatrixD >::value_type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixA >::value_type, typename traits::matrix_traits<
                MatrixE >::value_type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixA >::value_type, typename traits::matrix_traits<
                MatrixF >::value_type >::value) );
        BOOST_ASSERT( trans == 'N' || trans == 'T' );
        BOOST_ASSERT( traits::vector_size(work.select(real_type())) >=
                min_size_work( $CALL_MIN_SIZE ));
        BOOST_ASSERT( traits::vector_size(work.select(integer_t())) >=
                min_size_iwork( m, n ));
        detail::tgsyl( trans, ijob, m, n, traits::matrix_storage(a),
                traits::leading_dimension(a), traits::matrix_storage(b),
                traits::leading_dimension(b), traits::matrix_storage(c),
                traits::leading_dimension(c), traits::matrix_storage(d),
                traits::leading_dimension(d), traits::matrix_storage(e),
                traits::leading_dimension(e), traits::matrix_storage(f),
                traits::leading_dimension(f), scale, dif,
                traits::vector_storage(work.select(real_type())),
                traits::vector_size(work.select(real_type())),
                traits::vector_storage(work.select(integer_t())), info );
    }

    // minimal workspace specialization
    template< typename MatrixA, typename MatrixB, typename MatrixC,
            typename MatrixD, typename MatrixE, typename MatrixF >
    static void compute( char const trans, integer_t const ijob,
            integer_t const m, integer_t const n, MatrixA& a, MatrixB& b,
            MatrixC& c, MatrixD& d, MatrixE& e, MatrixF& f, real_type& scale,
            real_type& dif, integer_t& info, minimal_workspace work ) {
        traits::detail::array< real_type > tmp_work( min_size_work(
                $CALL_MIN_SIZE ) );
        traits::detail::array< integer_t > tmp_iwork( min_size_iwork( m, n ) );
        compute( trans, ijob, m, n, a, b, c, d, e, f, scale, dif, info,
                workspace( tmp_work, tmp_iwork ) );
    }

    // optimal workspace specialization
    template< typename MatrixA, typename MatrixB, typename MatrixC,
            typename MatrixD, typename MatrixE, typename MatrixF >
    static void compute( char const trans, integer_t const ijob,
            integer_t const m, integer_t const n, MatrixA& a, MatrixB& b,
            MatrixC& c, MatrixD& d, MatrixE& e, MatrixF& f, real_type& scale,
            real_type& dif, integer_t& info, optimal_workspace work ) {
        real_type opt_size_work;
        traits::detail::array< integer_t > tmp_iwork( min_size_iwork( m, n ) );
        detail::tgsyl( trans, ijob, m, n, traits::matrix_storage(a),
                traits::leading_dimension(a), traits::matrix_storage(b),
                traits::leading_dimension(b), traits::matrix_storage(c),
                traits::leading_dimension(c), traits::matrix_storage(d),
                traits::leading_dimension(d), traits::matrix_storage(e),
                traits::leading_dimension(e), traits::matrix_storage(f),
                traits::leading_dimension(f), scale, dif, &opt_size_work, -1,
                traits::vector_storage(tmp_iwork), info );
        traits::detail::array< real_type > tmp_work(
                traits::detail::to_int( opt_size_work ) );
        compute( trans, ijob, m, n, a, b, c, d, e, f, scale, dif, info,
                workspace( tmp_work, tmp_iwork ) );
    }

    static integer_t min_size_work( $ARGUMENTS ) {
        $MIN_SIZE
    }

    static integer_t min_size_iwork( integer_t const m, integer_t const n ) {
        return m+n+6;
    }
};

// complex specialization
template< typename ValueType >
struct tgsyl_impl< ValueType, typename boost::enable_if< traits::is_complex<ValueType> >::type > {

    typedef ValueType value_type;
    typedef typename traits::type_traits<ValueType>::real_type real_type;

    // user-defined workspace specialization
    template< typename MatrixA, typename MatrixB, typename MatrixC,
            typename MatrixD, typename MatrixE, typename MatrixF,
            typename WORK, typename IWORK >
    static void compute( char const trans, integer_t const ijob,
            integer_t const m, integer_t const n, MatrixA& a, MatrixB& b,
            MatrixC& c, MatrixD& d, MatrixE& e, MatrixF& f, real_type& scale,
            real_type& dif, integer_t& info, detail::workspace2< WORK,
            IWORK > work ) {
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixA >::value_type, typename traits::matrix_traits<
                MatrixB >::value_type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixA >::value_type, typename traits::matrix_traits<
                MatrixC >::value_type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixA >::value_type, typename traits::matrix_traits<
                MatrixD >::value_type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixA >::value_type, typename traits::matrix_traits<
                MatrixE >::value_type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixA >::value_type, typename traits::matrix_traits<
                MatrixF >::value_type >::value) );
        BOOST_ASSERT( trans == 'N' || trans == 'C' );
        BOOST_ASSERT( traits::vector_size(work.select(value_type())) >=
                min_size_work( $CALL_MIN_SIZE ));
        BOOST_ASSERT( traits::vector_size(work.select(integer_t())) >=
                min_size_iwork( m, n ));
        detail::tgsyl( trans, ijob, m, n, traits::matrix_storage(a),
                traits::leading_dimension(a), traits::matrix_storage(b),
                traits::leading_dimension(b), traits::matrix_storage(c),
                traits::leading_dimension(c), traits::matrix_storage(d),
                traits::leading_dimension(d), traits::matrix_storage(e),
                traits::leading_dimension(e), traits::matrix_storage(f),
                traits::leading_dimension(f), scale, dif,
                traits::vector_storage(work.select(value_type())),
                traits::vector_size(work.select(value_type())),
                traits::vector_storage(work.select(integer_t())), info );
    }

    // minimal workspace specialization
    template< typename MatrixA, typename MatrixB, typename MatrixC,
            typename MatrixD, typename MatrixE, typename MatrixF >
    static void compute( char const trans, integer_t const ijob,
            integer_t const m, integer_t const n, MatrixA& a, MatrixB& b,
            MatrixC& c, MatrixD& d, MatrixE& e, MatrixF& f, real_type& scale,
            real_type& dif, integer_t& info, minimal_workspace work ) {
        traits::detail::array< value_type > tmp_work( min_size_work(
                $CALL_MIN_SIZE ) );
        traits::detail::array< integer_t > tmp_iwork( min_size_iwork( m, n ) );
        compute( trans, ijob, m, n, a, b, c, d, e, f, scale, dif, info,
                workspace( tmp_work, tmp_iwork ) );
    }

    // optimal workspace specialization
    template< typename MatrixA, typename MatrixB, typename MatrixC,
            typename MatrixD, typename MatrixE, typename MatrixF >
    static void compute( char const trans, integer_t const ijob,
            integer_t const m, integer_t const n, MatrixA& a, MatrixB& b,
            MatrixC& c, MatrixD& d, MatrixE& e, MatrixF& f, real_type& scale,
            real_type& dif, integer_t& info, optimal_workspace work ) {
        value_type opt_size_work;
        traits::detail::array< integer_t > tmp_iwork( min_size_iwork( m, n ) );
        detail::tgsyl( trans, ijob, m, n, traits::matrix_storage(a),
                traits::leading_dimension(a), traits::matrix_storage(b),
                traits::leading_dimension(b), traits::matrix_storage(c),
                traits::leading_dimension(c), traits::matrix_storage(d),
                traits::leading_dimension(d), traits::matrix_storage(e),
                traits::leading_dimension(e), traits::matrix_storage(f),
                traits::leading_dimension(f), scale, dif, &opt_size_work, -1,
                traits::vector_storage(tmp_iwork), info );
        traits::detail::array< value_type > tmp_work(
                traits::detail::to_int( opt_size_work ) );
        compute( trans, ijob, m, n, a, b, c, d, e, f, scale, dif, info,
                workspace( tmp_work, tmp_iwork ) );
    }

    static integer_t min_size_work( $ARGUMENTS ) {
        $MIN_SIZE
    }

    static integer_t min_size_iwork( integer_t const m, integer_t const n ) {
        return m+n+2;
    }
};


// template function to call tgsyl
template< typename MatrixA, typename MatrixB, typename MatrixC,
        typename MatrixD, typename MatrixE, typename MatrixF,
        typename Workspace >
inline integer_t tgsyl( char const trans, integer_t const ijob,
        integer_t const m, integer_t const n, MatrixA& a, MatrixB& b,
        MatrixC& c, MatrixD& d, MatrixE& e, MatrixF& f,
        typename traits::matrix_traits< MatrixA >::value_type& scale,
        typename traits::matrix_traits< MatrixA >::value_type& dif,
        Workspace work = optimal_workspace() ) {
    typedef typename traits::matrix_traits< MatrixA >::value_type value_type;
    integer_t info(0);
    tgsyl_impl< value_type >::compute( trans, ijob, m, n, a, b, c, d, e,
            f, scale, dif, info, work );
    return info;
}

}}}} // namespace boost::numeric::bindings::lapack

#endif
