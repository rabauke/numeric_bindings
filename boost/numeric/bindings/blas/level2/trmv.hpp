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

#ifndef BOOST_NUMERIC_BINDINGS_BLAS_LEVEL2_TRMV_HPP
#define BOOST_NUMERIC_BINDINGS_BLAS_LEVEL2_TRMV_HPP

// Include header of configured BLAS interface
#if defined BOOST_NUMERIC_BINDINGS_BLAS_CBLAS
#include <boost/numeric/bindings/blas/detail/cblas.h>
#elif defined BOOST_NUMERIC_BINDINGS_BLAS_CUBLAS
#include <boost/numeric/bindings/blas/detail/cublas.h>
#else
#include <boost/numeric/bindings/blas/detail/blas.h>
#endif

#include <boost/mpl/bool.hpp>
#include <boost/numeric/bindings/traits/traits.hpp>
#include <boost/numeric/bindings/traits/type_traits.hpp>
#include <boost/static_assert.hpp>
#include <boost/type_traits/is_same.hpp>

namespace boost {
namespace numeric {
namespace bindings {
namespace blas {

// The detail namespace is used for overloads on value type,
// and to dispatch to the right routine

namespace detail {

inline void trmv( const char uplo, const char trans, const char diag,
        const integer_t n, const float* a, const integer_t lda, float* x,
        const integer_t incx ) {
#if defined BOOST_NUMERIC_BINDINGS_BLAS_CBLAS
    cblas_strmv( CblasColMajor, ( uplo == 'U' ? CblasUpper : CblasLower ),
            ( trans == 'N' ? CblasNoTrans : ( trans == 'T' ? CblasTrans : CblasConjTrans ) ),
            ( uplo == 'N' ? CblasNonUnit : CblasUnit ), n, a, lda, x, incx );
#elif defined BOOST_NUMERIC_BINDINGS_BLAS_CUBLAS
    cublasStrmv( uplo, trans, diag, n, a, lda, x, incx );
#else
    BLAS_STRMV( &uplo, &trans, &diag, &n, a, &lda, x, &incx );
#endif
}

inline void trmv( const char uplo, const char trans, const char diag,
        const integer_t n, const double* a, const integer_t lda, double* x,
        const integer_t incx ) {
#if defined BOOST_NUMERIC_BINDINGS_BLAS_CBLAS
    cblas_dtrmv( CblasColMajor, ( uplo == 'U' ? CblasUpper : CblasLower ),
            ( trans == 'N' ? CblasNoTrans : ( trans == 'T' ? CblasTrans : CblasConjTrans ) ),
            ( uplo == 'N' ? CblasNonUnit : CblasUnit ), n, a, lda, x, incx );
#elif defined BOOST_NUMERIC_BINDINGS_BLAS_CUBLAS
    // NOT FOUND();
#else
    BLAS_DTRMV( &uplo, &trans, &diag, &n, a, &lda, x, &incx );
#endif
}

inline void trmv( const char uplo, const char trans, const char diag,
        const integer_t n, const traits::complex_f* a, const integer_t lda,
        traits::complex_f* x, const integer_t incx ) {
#if defined BOOST_NUMERIC_BINDINGS_BLAS_CBLAS
    cblas_ctrmv( CblasColMajor, ( uplo == 'U' ? CblasUpper : CblasLower ),
            ( trans == 'N' ? CblasNoTrans : ( trans == 'T' ? CblasTrans : CblasConjTrans ) ),
            ( uplo == 'N' ? CblasNonUnit : CblasUnit ), n,
            traits::void_ptr(a), lda, traits::void_ptr(x), incx );
#elif defined BOOST_NUMERIC_BINDINGS_BLAS_CUBLAS
    // NOT FOUND();
#else
    BLAS_CTRMV( &uplo, &trans, &diag, &n, traits::complex_ptr(a), &lda,
            traits::complex_ptr(x), &incx );
#endif
}

inline void trmv( const char uplo, const char trans, const char diag,
        const integer_t n, const traits::complex_d* a, const integer_t lda,
        traits::complex_d* x, const integer_t incx ) {
#if defined BOOST_NUMERIC_BINDINGS_BLAS_CBLAS
    cblas_ztrmv( CblasColMajor, ( uplo == 'U' ? CblasUpper : CblasLower ),
            ( trans == 'N' ? CblasNoTrans : ( trans == 'T' ? CblasTrans : CblasConjTrans ) ),
            ( uplo == 'N' ? CblasNonUnit : CblasUnit ), n,
            traits::void_ptr(a), lda, traits::void_ptr(x), incx );
#elif defined BOOST_NUMERIC_BINDINGS_BLAS_CUBLAS
    // NOT FOUND();
#else
    BLAS_ZTRMV( &uplo, &trans, &diag, &n, traits::complex_ptr(a), &lda,
            traits::complex_ptr(x), &incx );
#endif
}


} // namespace detail

// value-type based template
template< typename ValueType >
struct trmv_impl {

    typedef ValueType value_type;
    typedef typename traits::type_traits<ValueType>::real_type real_type;
    typedef void return_type;

    // static template member function
    template< typename MatrixA, typename VectorX >
    static return_type invoke( const char trans, const char diag,
            const MatrixA& a, VectorX& x ) {
        BOOST_STATIC_ASSERT( (boost::is_same< typename traits::matrix_traits<
                MatrixA >::value_type, typename traits::vector_traits<
                VectorX >::value_type >::value) );
        detail::trmv( traits::matrix_uplo_tag(a), trans, diag,
                traits::matrix_num_columns(a), traits::matrix_storage(a),
                traits::leading_dimension(a), traits::vector_storage(x),
                traits::vector_stride(x) );
    }
};

// generic template function to call trmv
template< typename MatrixA, typename VectorX >
inline typename trmv_impl< typename traits::matrix_traits<
        MatrixA >::value_type >::return_type
trmv( const char trans, const char diag, const MatrixA& a, VectorX& x ) {
    typedef typename traits::matrix_traits< MatrixA >::value_type value_type;
    trmv_impl< value_type >::invoke( trans, diag, a, x );
}

} // namespace blas
} // namespace bindings
} // namespace numeric
} // namespace boost

#endif
