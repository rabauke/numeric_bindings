//
// Copyright (c) 2002--2010
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

#ifndef BOOST_NUMERIC_BINDINGS_LAPACK_COMPUTATIONAL_POEQU_HPP
#define BOOST_NUMERIC_BINDINGS_LAPACK_COMPUTATIONAL_POEQU_HPP

#include <boost/assert.hpp>
#include <boost/numeric/bindings/begin.hpp>
#include <boost/numeric/bindings/is_complex.hpp>
#include <boost/numeric/bindings/is_mutable.hpp>
#include <boost/numeric/bindings/is_real.hpp>
#include <boost/numeric/bindings/remove_imaginary.hpp>
#include <boost/numeric/bindings/size.hpp>
#include <boost/numeric/bindings/stride.hpp>
#include <boost/numeric/bindings/value.hpp>
#include <boost/static_assert.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/type_traits/remove_const.hpp>
#include <boost/utility/enable_if.hpp>

//
// The LAPACK-backend for poequ is the netlib-compatible backend.
//
#include <boost/numeric/bindings/lapack/detail/lapack.h>
#include <boost/numeric/bindings/lapack/detail/lapack_option.hpp>

namespace boost {
namespace numeric {
namespace bindings {
namespace lapack {

//
// The detail namespace contains value-type-overloaded functions that
// dispatch to the appropriate back-end LAPACK-routine.
//
namespace detail {

//
// Overloaded function for dispatching to
// * netlib-compatible LAPACK backend (the default), and
// * float value-type.
//
inline std::ptrdiff_t poequ( fortran_int_t n, const float* a,
        fortran_int_t lda, float* s, float& scond, float& amax ) {
    fortran_int_t info(0);
    LAPACK_SPOEQU( &n, a, &lda, s, &scond, &amax, &info );
    return info;
}

//
// Overloaded function for dispatching to
// * netlib-compatible LAPACK backend (the default), and
// * double value-type.
//
inline std::ptrdiff_t poequ( fortran_int_t n, const double* a,
        fortran_int_t lda, double* s, double& scond, double& amax ) {
    fortran_int_t info(0);
    LAPACK_DPOEQU( &n, a, &lda, s, &scond, &amax, &info );
    return info;
}

//
// Overloaded function for dispatching to
// * netlib-compatible LAPACK backend (the default), and
// * complex<float> value-type.
//
inline std::ptrdiff_t poequ( fortran_int_t n, const std::complex<float>* a,
        fortran_int_t lda, float* s, float& scond, float& amax ) {
    fortran_int_t info(0);
    LAPACK_CPOEQU( &n, a, &lda, s, &scond, &amax, &info );
    return info;
}

//
// Overloaded function for dispatching to
// * netlib-compatible LAPACK backend (the default), and
// * complex<double> value-type.
//
inline std::ptrdiff_t poequ( fortran_int_t n, const std::complex<double>* a,
        fortran_int_t lda, double* s, double& scond, double& amax ) {
    fortran_int_t info(0);
    LAPACK_ZPOEQU( &n, a, &lda, s, &scond, &amax, &info );
    return info;
}

} // namespace detail

//
// Value-type based template class. Use this class if you need a type
// for dispatching to poequ.
//
template< typename Value, typename Enable = void >
struct poequ_impl {};

//
// This implementation is enabled if Value is a real type.
//
template< typename Value >
struct poequ_impl< Value, typename boost::enable_if< is_real< Value > >::type > {

    typedef Value value_type;
    typedef typename remove_imaginary< Value >::type real_type;
    typedef tag::column_major order;

    //
    // Static member function, that
    // * Deduces the required arguments for dispatching to LAPACK, and
    // * Asserts that most arguments make sense.
    //
    template< typename MatrixA, typename VectorS >
    static std::ptrdiff_t invoke( const MatrixA& a, VectorS& s,
            real_type& scond, real_type& amax ) {
        BOOST_STATIC_ASSERT( (boost::is_same< typename remove_const<
                typename value< MatrixA >::type >::type,
                typename remove_const< typename value<
                VectorS >::type >::type >::value) );
        BOOST_STATIC_ASSERT( (is_mutable< VectorS >::value) );
        BOOST_ASSERT( size_column(a) >= 0 );
        BOOST_ASSERT( size_minor(a) == 1 || stride_minor(a) == 1 );
        BOOST_ASSERT( stride_major(a) >= std::max< std::ptrdiff_t >(1,
                size_column(a)) );
        return detail::poequ( size_column(a), begin_value(a), stride_major(a),
                begin_value(s), scond, amax );
    }

};

//
// This implementation is enabled if Value is a complex type.
//
template< typename Value >
struct poequ_impl< Value, typename boost::enable_if< is_complex< Value > >::type > {

    typedef Value value_type;
    typedef typename remove_imaginary< Value >::type real_type;
    typedef tag::column_major order;

    //
    // Static member function, that
    // * Deduces the required arguments for dispatching to LAPACK, and
    // * Asserts that most arguments make sense.
    //
    template< typename MatrixA, typename VectorS >
    static std::ptrdiff_t invoke( const MatrixA& a, VectorS& s,
            real_type& scond, real_type& amax ) {
        BOOST_STATIC_ASSERT( (is_mutable< VectorS >::value) );
        BOOST_ASSERT( size_column(a) >= 0 );
        BOOST_ASSERT( size_minor(a) == 1 || stride_minor(a) == 1 );
        BOOST_ASSERT( stride_major(a) >= std::max< std::ptrdiff_t >(1,
                size_column(a)) );
        return detail::poequ( size_column(a), begin_value(a), stride_major(a),
                begin_value(s), scond, amax );
    }

};


//
// Functions for direct use. These functions are overloaded for temporaries,
// so that wrapped types can still be passed and used for write-access. In
// addition, if applicable, they are overloaded for user-defined workspaces.
// Calls to these functions are passed to the poequ_impl classes. In the 
// documentation, most overloads are collapsed to avoid a large number of
// prototypes which are very similar.
//

//
// Overloaded function for poequ. Its overload differs for
// * VectorS&
//
template< typename MatrixA, typename VectorS >
inline std::ptrdiff_t poequ( const MatrixA& a, VectorS& s,
        typename remove_imaginary< typename value<
        MatrixA >::type >::type& scond, typename remove_imaginary<
        typename value< MatrixA >::type >::type& amax ) {
    return poequ_impl< typename value< MatrixA >::type >::invoke( a, s,
            scond, amax );
}

//
// Overloaded function for poequ. Its overload differs for
// * const VectorS&
//
template< typename MatrixA, typename VectorS >
inline std::ptrdiff_t poequ( const MatrixA& a, const VectorS& s,
        typename remove_imaginary< typename value<
        MatrixA >::type >::type& scond, typename remove_imaginary<
        typename value< MatrixA >::type >::type& amax ) {
    return poequ_impl< typename value< MatrixA >::type >::invoke( a, s,
            scond, amax );
}

} // namespace lapack
} // namespace bindings
} // namespace numeric
} // namespace boost

#endif
