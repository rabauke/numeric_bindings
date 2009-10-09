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

#ifndef BOOST_NUMERIC_BINDINGS_LAPACK_COMPUTATIONAL_TGSNA_HPP
#define BOOST_NUMERIC_BINDINGS_LAPACK_COMPUTATIONAL_TGSNA_HPP

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

inline void tgsna( const char job, const char howmny, const logical_t* select,
        const integer_t n, const float* a, const integer_t lda,
        const float* b, const integer_t ldb, const float* vl,
        const integer_t ldvl, const float* vr, const integer_t ldvr, float* s,
        float* dif, const integer_t mm, integer_t& m, float* work,
        const integer_t lwork, integer_t* iwork, integer_t& info ) {
    LAPACK_STGSNA( &job, &howmny, select, &n, a, &lda, b, &ldb, vl, &ldvl, vr,
            &ldvr, s, dif, &mm, &m, work, &lwork, iwork, &info );
}
inline void tgsna( const char job, const char howmny, const logical_t* select,
        const integer_t n, const double* a, const integer_t lda,
        const double* b, const integer_t ldb, const double* vl,
        const integer_t ldvl, const double* vr, const integer_t ldvr,
        double* s, double* dif, const integer_t mm, integer_t& m,
        double* work, const integer_t lwork, integer_t* iwork,
        integer_t& info ) {
    LAPACK_DTGSNA( &job, &howmny, select, &n, a, &lda, b, &ldb, vl, &ldvl, vr,
            &ldvr, s, dif, &mm, &m, work, &lwork, iwork, &info );
}
inline void tgsna( const char job, const char howmny, const logical_t* select,
        const integer_t n, const traits::complex_f* a, const integer_t lda,
        const traits::complex_f* b, const integer_t ldb,
        const traits::complex_f* vl, const integer_t ldvl,
        const traits::complex_f* vr, const integer_t ldvr, float* s,
        float* dif, const integer_t mm, integer_t& m, traits::complex_f* work,
        const integer_t lwork, integer_t* iwork, integer_t& info ) {
    LAPACK_CTGSNA( &job, &howmny, select, &n, traits::complex_ptr(a), &lda,
            traits::complex_ptr(b), &ldb, traits::complex_ptr(vl), &ldvl,
            traits::complex_ptr(vr), &ldvr, s, dif, &mm, &m,
            traits::complex_ptr(work), &lwork, iwork, &info );
}
inline void tgsna( const char job, const char howmny, const logical_t* select,
        const integer_t n, const traits::complex_d* a, const integer_t lda,
        const traits::complex_d* b, const integer_t ldb,
        const traits::complex_d* vl, const integer_t ldvl,
        const traits::complex_d* vr, const integer_t ldvr, double* s,
        double* dif, const integer_t mm, integer_t& m,
        traits::complex_d* work, const integer_t lwork, integer_t* iwork,
        integer_t& info ) {
    LAPACK_ZTGSNA( &job, &howmny, select, &n, traits::complex_ptr(a), &lda,
            traits::complex_ptr(b), &ldb, traits::complex_ptr(vl), &ldvl,
            traits::complex_ptr(vr), &ldvr, s, dif, &mm, &m,
            traits::complex_ptr(work), &lwork, iwork, &info );
}
} // namespace detail

// value-type based template
template< typename ValueType, typename Enable = void >
struct tgsna_impl{};

// real specialization
template< typename ValueType >
struct tgsna_impl< ValueType, typename boost::enable_if< traits::is_real<ValueType> >::type > {

    typedef ValueType value_type;
    typedef typename traits::type_traits<ValueType>::real_type real_type;

    // user-defined workspace specialization
    template< typename VectorSELECT, typename MatrixA, typename MatrixB,
            typename MatrixVL, typename MatrixVR, typename VectorS,
            typename VectorDIF, typename WORK, typename IWORK >
    static void invoke( const char job, const char howmny,
            const VectorSELECT& select, const integer_t n, const MatrixA& a,
            const MatrixB& b, const MatrixVL& vl, const MatrixVR& vr,
            VectorS& s, VectorDIF& dif, const integer_t mm, integer_t& m,
            integer_t& info, detail::workspace2< WORK, IWORK > work ) {
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixA >::value_type, typename traits::matrix_traits<
                MatrixB >::value_type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixA >::value_type, typename traits::matrix_traits<
                MatrixVL >::value_type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixA >::value_type, typename traits::matrix_traits<
                MatrixVR >::value_type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixA >::value_type, typename traits::vector_traits<
                VectorS >::value_type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixA >::value_type, typename traits::vector_traits<
                VectorDIF >::value_type >::value) );
        BOOST_ASSERT( job == 'E' || job == 'V' || job == 'B' );
        BOOST_ASSERT( howmny == 'A' || howmny == 'S' );
        BOOST_ASSERT( n >= 0 );
        BOOST_ASSERT( traits::leading_dimension(a) >= std::max<
                std::ptrdiff_t >(1,n) );
        BOOST_ASSERT( traits::leading_dimension(b) >= std::max<
                std::ptrdiff_t >(1,n) );
        BOOST_ASSERT( mm >= m );
        BOOST_ASSERT( traits::vector_size(work.select(real_type())) >=
                min_size_work( $CALL_MIN_SIZE ));
        BOOST_ASSERT( traits::vector_size(work.select(integer_t())) >=
                min_size_iwork( $CALL_MIN_SIZE ));
        detail::tgsna( job, howmny, traits::vector_storage(select), n,
                traits::matrix_storage(a), traits::leading_dimension(a),
                traits::matrix_storage(b), traits::leading_dimension(b),
                traits::matrix_storage(vl), traits::leading_dimension(vl),
                traits::matrix_storage(vr), traits::leading_dimension(vr),
                traits::vector_storage(s), traits::vector_storage(dif), mm, m,
                traits::vector_storage(work.select(real_type())),
                traits::vector_size(work.select(real_type())),
                traits::vector_storage(work.select(integer_t())), info );
    }

    // minimal workspace specialization
    template< typename VectorSELECT, typename MatrixA, typename MatrixB,
            typename MatrixVL, typename MatrixVR, typename VectorS,
            typename VectorDIF >
    static void invoke( const char job, const char howmny,
            const VectorSELECT& select, const integer_t n, const MatrixA& a,
            const MatrixB& b, const MatrixVL& vl, const MatrixVR& vr,
            VectorS& s, VectorDIF& dif, const integer_t mm, integer_t& m,
            integer_t& info, minimal_workspace work ) {
        traits::detail::array< real_type > tmp_work( min_size_work(
                $CALL_MIN_SIZE ) );
        traits::detail::array< integer_t > tmp_iwork( min_size_iwork(
                $CALL_MIN_SIZE ) );
        invoke( job, howmny, select, n, a, b, vl, vr, s, dif, mm, m, info,
                workspace( tmp_work, tmp_iwork ) );
    }

    // optimal workspace specialization
    template< typename VectorSELECT, typename MatrixA, typename MatrixB,
            typename MatrixVL, typename MatrixVR, typename VectorS,
            typename VectorDIF >
    static void invoke( const char job, const char howmny,
            const VectorSELECT& select, const integer_t n, const MatrixA& a,
            const MatrixB& b, const MatrixVL& vl, const MatrixVR& vr,
            VectorS& s, VectorDIF& dif, const integer_t mm, integer_t& m,
            integer_t& info, optimal_workspace work ) {
        real_type opt_size_work;
        traits::detail::array< integer_t > tmp_iwork( min_size_iwork(
                $CALL_MIN_SIZE ) );
        detail::tgsna( job, howmny, traits::vector_storage(select), n,
                traits::matrix_storage(a), traits::leading_dimension(a),
                traits::matrix_storage(b), traits::leading_dimension(b),
                traits::matrix_storage(vl), traits::leading_dimension(vl),
                traits::matrix_storage(vr), traits::leading_dimension(vr),
                traits::vector_storage(s), traits::vector_storage(dif), mm, m,
                &opt_size_work, -1, traits::vector_storage(tmp_iwork), info );
        traits::detail::array< real_type > tmp_work(
                traits::detail::to_int( opt_size_work ) );
        invoke( job, howmny, select, n, a, b, vl, vr, s, dif, mm, m, info,
                workspace( tmp_work, tmp_iwork ) );
    }

    static integer_t min_size_work( $ARGUMENTS ) {
        $MIN_SIZE
    }

    static integer_t min_size_iwork( $ARGUMENTS ) {
        $MIN_SIZE
    }
};

// complex specialization
template< typename ValueType >
struct tgsna_impl< ValueType, typename boost::enable_if< traits::is_complex<ValueType> >::type > {

    typedef ValueType value_type;
    typedef typename traits::type_traits<ValueType>::real_type real_type;

    // user-defined workspace specialization
    template< typename VectorSELECT, typename MatrixA, typename MatrixB,
            typename MatrixVL, typename MatrixVR, typename VectorS,
            typename VectorDIF, typename WORK, typename IWORK >
    static void invoke( const char job, const char howmny,
            const VectorSELECT& select, const integer_t n, const MatrixA& a,
            const MatrixB& b, const MatrixVL& vl, const MatrixVR& vr,
            VectorS& s, VectorDIF& dif, const integer_t mm, integer_t& m,
            integer_t& info, detail::workspace2< WORK, IWORK > work ) {
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::vector_traits<
                VectorS >::value_type, typename traits::vector_traits<
                VectorDIF >::value_type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixA >::value_type, typename traits::matrix_traits<
                MatrixB >::value_type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixA >::value_type, typename traits::matrix_traits<
                MatrixVL >::value_type >::value) );
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixA >::value_type, typename traits::matrix_traits<
                MatrixVR >::value_type >::value) );
        BOOST_ASSERT( job == 'E' || job == 'V' || job == 'B' );
        BOOST_ASSERT( howmny == 'A' || howmny == 'S' );
        BOOST_ASSERT( n >= 0 );
        BOOST_ASSERT( traits::leading_dimension(a) >= std::max<
                std::ptrdiff_t >(1,n) );
        BOOST_ASSERT( traits::leading_dimension(b) >= std::max<
                std::ptrdiff_t >(1,n) );
        BOOST_ASSERT( mm >= m );
        BOOST_ASSERT( traits::vector_size(work.select(value_type())) >=
                min_size_work( $CALL_MIN_SIZE ));
        BOOST_ASSERT( traits::vector_size(work.select(integer_t())) >=
                min_size_iwork( $CALL_MIN_SIZE ));
        detail::tgsna( job, howmny, traits::vector_storage(select), n,
                traits::matrix_storage(a), traits::leading_dimension(a),
                traits::matrix_storage(b), traits::leading_dimension(b),
                traits::matrix_storage(vl), traits::leading_dimension(vl),
                traits::matrix_storage(vr), traits::leading_dimension(vr),
                traits::vector_storage(s), traits::vector_storage(dif), mm, m,
                traits::vector_storage(work.select(value_type())),
                traits::vector_size(work.select(value_type())),
                traits::vector_storage(work.select(integer_t())), info );
    }

    // minimal workspace specialization
    template< typename VectorSELECT, typename MatrixA, typename MatrixB,
            typename MatrixVL, typename MatrixVR, typename VectorS,
            typename VectorDIF >
    static void invoke( const char job, const char howmny,
            const VectorSELECT& select, const integer_t n, const MatrixA& a,
            const MatrixB& b, const MatrixVL& vl, const MatrixVR& vr,
            VectorS& s, VectorDIF& dif, const integer_t mm, integer_t& m,
            integer_t& info, minimal_workspace work ) {
        traits::detail::array< value_type > tmp_work( min_size_work(
                $CALL_MIN_SIZE ) );
        traits::detail::array< integer_t > tmp_iwork( min_size_iwork(
                $CALL_MIN_SIZE ) );
        invoke( job, howmny, select, n, a, b, vl, vr, s, dif, mm, m, info,
                workspace( tmp_work, tmp_iwork ) );
    }

    // optimal workspace specialization
    template< typename VectorSELECT, typename MatrixA, typename MatrixB,
            typename MatrixVL, typename MatrixVR, typename VectorS,
            typename VectorDIF >
    static void invoke( const char job, const char howmny,
            const VectorSELECT& select, const integer_t n, const MatrixA& a,
            const MatrixB& b, const MatrixVL& vl, const MatrixVR& vr,
            VectorS& s, VectorDIF& dif, const integer_t mm, integer_t& m,
            integer_t& info, optimal_workspace work ) {
        invoke( job, howmny, select, n, a, b, vl, vr, s, dif, mm, m, info,
                minimal_workspace() );
    }

    static integer_t min_size_work( $ARGUMENTS ) {
        $MIN_SIZE
    }

    static integer_t min_size_iwork( $ARGUMENTS ) {
        $MIN_SIZE
    }
};


// template function to call tgsna
template< typename VectorSELECT, typename MatrixA, typename MatrixB,
        typename MatrixVL, typename MatrixVR, typename VectorS,
        typename VectorDIF, typename Workspace >
inline integer_t tgsna( const char job, const char howmny,
        const VectorSELECT& select, const integer_t n, const MatrixA& a,
        const MatrixB& b, const MatrixVL& vl, const MatrixVR& vr, VectorS& s,
        VectorDIF& dif, const integer_t mm, integer_t& m, Workspace work ) {
    typedef typename traits::matrix_traits< MatrixA >::value_type value_type;
    integer_t info(0);
    tgsna_impl< value_type >::invoke( job, howmny, select, n, a, b, vl,
            vr, s, dif, mm, m, info, work );
    return info;
}

// template function to call tgsna, default workspace type
template< typename VectorSELECT, typename MatrixA, typename MatrixB,
        typename MatrixVL, typename MatrixVR, typename VectorS,
        typename VectorDIF >
inline integer_t tgsna( const char job, const char howmny,
        const VectorSELECT& select, const integer_t n, const MatrixA& a,
        const MatrixB& b, const MatrixVL& vl, const MatrixVR& vr, VectorS& s,
        VectorDIF& dif, const integer_t mm, integer_t& m ) {
    typedef typename traits::matrix_traits< MatrixA >::value_type value_type;
    integer_t info(0);
    tgsna_impl< value_type >::invoke( job, howmny, select, n, a, b, vl,
            vr, s, dif, mm, m, info, optimal_workspace() );
    return info;
}

} // namespace lapack
} // namespace bindings
} // namespace numeric
} // namespace boost

#endif
