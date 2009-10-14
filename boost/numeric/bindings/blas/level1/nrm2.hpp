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

#ifndef BOOST_NUMERIC_BINDINGS_BLAS_LEVEL1_NRM2_HPP
#define BOOST_NUMERIC_BINDINGS_BLAS_LEVEL1_NRM2_HPP

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

inline float nrm2( const integer_t n, const float* x, const integer_t incx ) {
#if defined BOOST_NUMERIC_BINDINGS_BLAS_CBLAS
    return cblas_snrm2( n, x, incx );
#elif defined BOOST_NUMERIC_BINDINGS_BLAS_CUBLAS
    return cublasSnrm2( n, x, incx );
#else
    return BLAS_SNRM2( &n, x, &incx );
#endif
}

inline double nrm2( const integer_t n, const double* x,
        const integer_t incx ) {
#if defined BOOST_NUMERIC_BINDINGS_BLAS_CBLAS
    return cblas_dnrm2( n, x, incx );
#elif defined BOOST_NUMERIC_BINDINGS_BLAS_CUBLAS
    return cublasDnrm2( n, x, incx );
#else
    return BLAS_DNRM2( &n, x, &incx );
#endif
}


} // namespace detail

// value-type based template
template< typename ValueType >
struct nrm2_impl {

    typedef ValueType value_type;
    typedef typename traits::type_traits<ValueType>::real_type real_type;
    typedef value_type return_type;

    // static template member function
    template< typename VectorX >
    static return_type invoke( const VectorX& x ) {
        return detail::nrm2( traits::vector_size(x),
                traits::vector_storage(x), traits::vector_stride(x) );
    }
};

// generic template function to call nrm2
template< typename VectorX >
inline typename nrm2_impl< typename traits::vector_traits<
        VectorX >::value_type >::return_type
nrm2( const VectorX& x ) {
    typedef typename traits::vector_traits< VectorX >::value_type value_type;
    return nrm2_impl< value_type >::invoke( x );
}

} // namespace blas
} // namespace bindings
} // namespace numeric
} // namespace boost

#endif
