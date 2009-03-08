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

#ifndef BOOST_NUMERIC_BINDINGS_LAPACK_PPSVX_HPP
#define BOOST_NUMERIC_BINDINGS_LAPACK_PPSVX_HPP

#include <boost/numeric/bindings/lapack/lapack.h>
#include <boost/numeric/bindings/lapack/workspace.hpp>
#include <boost/numeric/bindings/traits/detail/array.hpp>
#include <boost/numeric/bindings/traits/is_complex.hpp>
#include <boost/numeric/bindings/traits/is_real.hpp>
#include <boost/numeric/bindings/traits/traits.hpp>
#include <boost/numeric/bindings/traits/type_traits.hpp>
#include <boost/static_assert.hpp>
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
    inline void ppsvx( char const fact, char const uplo, integer_t const n,
            integer_t const nrhs, float* ap, float* afp, char& equed,
            float* s, float* b, integer_t const ldb, float* x,
            integer_t const ldx, float& rcond, float* ferr, float* berr,
            float* work, integer_t* iwork, integer_t& info ) {
        LAPACK_SPPSVX( &fact, &uplo, &n, &nrhs, ap, afp, &equed, s, b, &ldb,
                x, &ldx, &rcond, ferr, berr, work, iwork, &info );
    }
    inline void ppsvx( char const fact, char const uplo, integer_t const n,
            integer_t const nrhs, double* ap, double* afp, char& equed,
            double* s, double* b, integer_t const ldb, double* x,
            integer_t const ldx, double& rcond, double* ferr, double* berr,
            double* work, integer_t* iwork, integer_t& info ) {
        LAPACK_DPPSVX( &fact, &uplo, &n, &nrhs, ap, afp, &equed, s, b, &ldb,
                x, &ldx, &rcond, ferr, berr, work, iwork, &info );
    }
    inline void ppsvx( char const fact, char const uplo, integer_t const n,
            integer_t const nrhs, traits::complex_f* ap,
            traits::complex_f* afp, char& equed, float* s,
            traits::complex_f* b, integer_t const ldb, traits::complex_f* x,
            integer_t const ldx, float& rcond, float* ferr, float* berr,
            traits::complex_f* work, float* rwork, integer_t& info ) {
        LAPACK_CPPSVX( &fact, &uplo, &n, &nrhs, traits::complex_ptr(ap),
                traits::complex_ptr(afp), &equed, s, traits::complex_ptr(b),
                &ldb, traits::complex_ptr(x), &ldx, &rcond, ferr, berr,
                traits::complex_ptr(work), rwork, &info );
    }
    inline void ppsvx( char const fact, char const uplo, integer_t const n,
            integer_t const nrhs, traits::complex_d* ap,
            traits::complex_d* afp, char& equed, double* s,
            traits::complex_d* b, integer_t const ldb, traits::complex_d* x,
            integer_t const ldx, double& rcond, double* ferr, double* berr,
            traits::complex_d* work, double* rwork, integer_t& info ) {
        LAPACK_ZPPSVX( &fact, &uplo, &n, &nrhs, traits::complex_ptr(ap),
                traits::complex_ptr(afp), &equed, s, traits::complex_ptr(b),
                &ldb, traits::complex_ptr(x), &ldx, &rcond, ferr, berr,
                traits::complex_ptr(work), rwork, &info );
    }
}

// value-type based template
template< typename ValueType, typename Enable = void >
struct ppsvx_impl{};

// real specialization
template< typename ValueType >
struct ppsvx_impl< ValueType, typename boost::enable_if< traits::is_real<ValueType> >::type > {

    typedef ValueType value_type;
    typedef typename traits::type_traits<ValueType>::real_type real_type;

    // user-defined workspace specialization
    template< typename MatrixAP, typename VectorAFP, typename VectorS,
            typename MatrixB, typename MatrixX, typename VectorFERR,
            typename VectorBERR, typename WORK, typename IWORK >
    static void compute( char const fact, MatrixAP& ap, VectorAFP& afp,
            char& equed, VectorS& s, MatrixB& b, MatrixX& x, real_type& rcond,
            VectorFERR& ferr, VectorBERR& berr, integer_t& info,
            detail::workspace2< WORK, IWORK > work ) {
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixAP >::value_type, typename traits::vector_traits<
                VectorAFP >::value_type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixAP >::value_type, typename traits::vector_traits<
                VectorS >::value_type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixAP >::value_type, typename traits::matrix_traits<
                MatrixB >::value_type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixAP >::value_type, typename traits::matrix_traits<
                MatrixX >::value_type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixAP >::value_type, typename traits::vector_traits<
                VectorFERR >::value_type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixAP >::value_type, typename traits::vector_traits<
                VectorBERR >::value_type >::value) );
#ifndef NDEBUG
        assert( fact == 'F' || fact == 'Y' || fact == 'N' || fact == 'E' );
        assert( traits::matrix_uplo_tag(ap) == 'U' ||
                traits::matrix_uplo_tag(ap) == 'L' );
        assert( traits::matrix_num_columns(ap) >= 0 );
        assert( traits::matrix_num_columns(x) >= 0 );
        assert( equed == 'N' || equed == 'Y' );
        assert( traits::leading_dimension(b) >= std::max(1,
                traits::matrix_num_columns(ap)) );
        assert( traits::leading_dimension(x) >= std::max(1,
                traits::matrix_num_columns(ap)) );
        assert( traits::vector_size(berr) >= traits::matrix_num_columns(x) );
        assert( traits::vector_size(work.select(real_type())) >=
                min_size_work( traits::matrix_num_columns(ap) ));
        assert( traits::vector_size(work.select(integer_t())) >=
                min_size_iwork( traits::matrix_num_columns(ap) ));
#endif
        detail::ppsvx( fact, traits::matrix_uplo_tag(ap),
                traits::matrix_num_columns(ap), traits::matrix_num_columns(x),
                traits::matrix_storage(ap), traits::vector_storage(afp),
                equed, traits::vector_storage(s), traits::matrix_storage(b),
                traits::leading_dimension(b), traits::matrix_storage(x),
                traits::leading_dimension(x), rcond,
                traits::vector_storage(ferr), traits::vector_storage(berr),
                traits::vector_storage(work.select(real_type())),
                traits::vector_storage(work.select(integer_t())), info );
    }

    // minimal workspace specialization
    template< typename MatrixAP, typename VectorAFP, typename VectorS,
            typename MatrixB, typename MatrixX, typename VectorFERR,
            typename VectorBERR >
    static void compute( char const fact, MatrixAP& ap, VectorAFP& afp,
            char& equed, VectorS& s, MatrixB& b, MatrixX& x, real_type& rcond,
            VectorFERR& ferr, VectorBERR& berr, integer_t& info,
            minimal_workspace work ) {
        traits::detail::array< real_type > tmp_work( min_size_work(
                traits::matrix_num_columns(ap) ) );
        traits::detail::array< integer_t > tmp_iwork( min_size_iwork(
                traits::matrix_num_columns(ap) ) );
        compute( fact, ap, afp, equed, s, b, x, rcond, ferr, berr, info,
                workspace( tmp_work, tmp_iwork ) );
    }

    // optimal workspace specialization
    template< typename MatrixAP, typename VectorAFP, typename VectorS,
            typename MatrixB, typename MatrixX, typename VectorFERR,
            typename VectorBERR >
    static void compute( char const fact, MatrixAP& ap, VectorAFP& afp,
            char& equed, VectorS& s, MatrixB& b, MatrixX& x, real_type& rcond,
            VectorFERR& ferr, VectorBERR& berr, integer_t& info,
            optimal_workspace work ) {
        compute( fact, ap, afp, equed, s, b, x, rcond, ferr, berr, info,
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
struct ppsvx_impl< ValueType, typename boost::enable_if< traits::is_complex<ValueType> >::type > {

    typedef ValueType value_type;
    typedef typename traits::type_traits<ValueType>::real_type real_type;

    // user-defined workspace specialization
    template< typename MatrixAP, typename VectorAFP, typename VectorS,
            typename MatrixB, typename MatrixX, typename VectorFERR,
            typename VectorBERR, typename WORK, typename RWORK >
    static void compute( char const fact, MatrixAP& ap, VectorAFP& afp,
            char& equed, VectorS& s, MatrixB& b, MatrixX& x, real_type& rcond,
            VectorFERR& ferr, VectorBERR& berr, integer_t& info,
            detail::workspace2< WORK, RWORK > work ) {
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::vector_traits<
                VectorS >::value_type, typename traits::vector_traits<
                VectorFERR >::value_type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::vector_traits<
                VectorS >::value_type, typename traits::vector_traits<
                VectorBERR >::value_type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixAP >::value_type, typename traits::vector_traits<
                VectorAFP >::value_type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixAP >::value_type, typename traits::matrix_traits<
                MatrixB >::value_type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixAP >::value_type, typename traits::matrix_traits<
                MatrixX >::value_type >::value) );
#ifndef NDEBUG
        assert( fact == 'F' || fact == 'Y' || fact == 'N' || fact == 'E' );
        assert( traits::matrix_uplo_tag(ap) == 'U' ||
                traits::matrix_uplo_tag(ap) == 'L' );
        assert( traits::matrix_num_columns(ap) >= 0 );
        assert( traits::matrix_num_columns(x) >= 0 );
        assert( equed == 'N' || equed == 'Y' );
        assert( traits::leading_dimension(b) >= std::max(1,
                traits::matrix_num_columns(ap)) );
        assert( traits::leading_dimension(x) >= std::max(1,
                traits::matrix_num_columns(ap)) );
        assert( traits::vector_size(berr) >= traits::matrix_num_columns(x) );
        assert( traits::vector_size(work.select(value_type())) >=
                min_size_work( traits::matrix_num_columns(ap) ));
        assert( traits::vector_size(work.select(real_type())) >=
                min_size_rwork( traits::matrix_num_columns(ap) ));
#endif
        detail::ppsvx( fact, traits::matrix_uplo_tag(ap),
                traits::matrix_num_columns(ap), traits::matrix_num_columns(x),
                traits::matrix_storage(ap), traits::vector_storage(afp),
                equed, traits::vector_storage(s), traits::matrix_storage(b),
                traits::leading_dimension(b), traits::matrix_storage(x),
                traits::leading_dimension(x), rcond,
                traits::vector_storage(ferr), traits::vector_storage(berr),
                traits::vector_storage(work.select(value_type())),
                traits::vector_storage(work.select(real_type())), info );
    }

    // minimal workspace specialization
    template< typename MatrixAP, typename VectorAFP, typename VectorS,
            typename MatrixB, typename MatrixX, typename VectorFERR,
            typename VectorBERR >
    static void compute( char const fact, MatrixAP& ap, VectorAFP& afp,
            char& equed, VectorS& s, MatrixB& b, MatrixX& x, real_type& rcond,
            VectorFERR& ferr, VectorBERR& berr, integer_t& info,
            minimal_workspace work ) {
        traits::detail::array< value_type > tmp_work( min_size_work(
                traits::matrix_num_columns(ap) ) );
        traits::detail::array< real_type > tmp_rwork( min_size_rwork(
                traits::matrix_num_columns(ap) ) );
        compute( fact, ap, afp, equed, s, b, x, rcond, ferr, berr, info,
                workspace( tmp_work, tmp_rwork ) );
    }

    // optimal workspace specialization
    template< typename MatrixAP, typename VectorAFP, typename VectorS,
            typename MatrixB, typename MatrixX, typename VectorFERR,
            typename VectorBERR >
    static void compute( char const fact, MatrixAP& ap, VectorAFP& afp,
            char& equed, VectorS& s, MatrixB& b, MatrixX& x, real_type& rcond,
            VectorFERR& ferr, VectorBERR& berr, integer_t& info,
            optimal_workspace work ) {
        compute( fact, ap, afp, equed, s, b, x, rcond, ferr, berr, info,
                minimal_workspace() );
    }

    static integer_t min_size_work( integer_t const n ) {
        return 2*n;
    }

    static integer_t min_size_rwork( integer_t const n ) {
        return n;
    }
};


// template function to call ppsvx
template< typename MatrixAP, typename VectorAFP, typename VectorS,
        typename MatrixB, typename MatrixX, typename VectorFERR,
        typename VectorBERR, typename Workspace >
inline integer_t ppsvx( char const fact, MatrixAP& ap, VectorAFP& afp,
        char& equed, VectorS& s, MatrixB& b, MatrixX& x,
        typename traits::matrix_traits< MatrixAP >::value_type& rcond,
        VectorFERR& ferr, VectorBERR& berr,
        Workspace work = optimal_workspace() ) {
    typedef typename traits::matrix_traits< MatrixAP >::value_type value_type;
    integer_t info(0);
    ppsvx_impl< value_type >::compute( fact, ap, afp, equed, s, b, x,
            rcond, ferr, berr, info, work );
    return info;
}

}}}} // namespace boost::numeric::bindings::lapack

#endif
