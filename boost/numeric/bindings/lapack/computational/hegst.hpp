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

#ifndef BOOST_NUMERIC_BINDINGS_LAPACK_COMPUTATIONAL_HEGST_HPP
#define BOOST_NUMERIC_BINDINGS_LAPACK_COMPUTATIONAL_HEGST_HPP

#include <boost/assert.hpp>
#include <boost/numeric/bindings/begin.hpp>
#include <boost/numeric/bindings/data_side.hpp>
#include <boost/numeric/bindings/is_mutable.hpp>
#include <boost/numeric/bindings/lapack/detail/lapack.h>
#include <boost/numeric/bindings/lapack/detail/lapack_option.hpp>
#include <boost/numeric/bindings/remove_imaginary.hpp>
#include <boost/numeric/bindings/size.hpp>
#include <boost/numeric/bindings/stride.hpp>
#include <boost/numeric/bindings/value.hpp>
#include <boost/static_assert.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/type_traits/remove_const.hpp>

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
// Overloaded function for dispatching to complex<float> value-type.
//
template< typename UpLo >
inline void hegst( fortran_int_t itype, UpLo, fortran_int_t n,
        std::complex<float>* a, fortran_int_t lda,
        const std::complex<float>* b, fortran_int_t ldb,
        fortran_int_t& info ) {
    LAPACK_CHEGST( &itype, &lapack_option< UpLo >::value, &n, a, &lda, b,
            &ldb, &info );
}

//
// Overloaded function for dispatching to complex<double> value-type.
//
template< typename UpLo >
inline void hegst( fortran_int_t itype, UpLo, fortran_int_t n,
        std::complex<double>* a, fortran_int_t lda,
        const std::complex<double>* b, fortran_int_t ldb,
        fortran_int_t& info ) {
    LAPACK_ZHEGST( &itype, &lapack_option< UpLo >::value, &n, a, &lda, b,
            &ldb, &info );
}

} // namespace detail

//
// Value-type based template class. Use this class if you need a type
// for dispatching to hegst.
//
template< typename Value >
struct hegst_impl {

    typedef Value value_type;
    typedef typename remove_imaginary< Value >::type real_type;
    typedef tag::column_major order;

    //
    // Static member function, that
    // * Deduces the required arguments for dispatching to LAPACK, and
    // * Asserts that most arguments make sense.
    //
    template< typename MatrixA, typename MatrixB >
    static void invoke( const fortran_int_t itype,
            const fortran_int_t n, MatrixA& a, const MatrixB& b,
            fortran_int_t& info ) {
        typedef typename result_of::data_side< MatrixA >::type uplo;
        BOOST_STATIC_ASSERT( (boost::is_same< typename remove_const<
                typename value< MatrixA >::type >::type,
                typename remove_const< typename value<
                MatrixB >::type >::type >::value) );
        BOOST_STATIC_ASSERT( (is_mutable< MatrixA >::value) );
        BOOST_ASSERT( n >= 0 );
        BOOST_ASSERT( size_minor(a) == 1 || stride_minor(a) == 1 );
        BOOST_ASSERT( size_minor(b) == 1 || stride_minor(b) == 1 );
        BOOST_ASSERT( stride_major(a) >= std::max< std::ptrdiff_t >(1,n) );
        BOOST_ASSERT( stride_major(b) >= std::max< std::ptrdiff_t >(1,n) );
        detail::hegst( itype, uplo(), n, begin_value(a), stride_major(a),
                begin_value(b), stride_major(b), info );
    }

};


//
// Functions for direct use. These functions are overloaded for temporaries,
// so that wrapped types can still be passed and used for write-access. In
// addition, if applicable, they are overloaded for user-defined workspaces.
// Calls to these functions are passed to the hegst_impl classes. In the 
// documentation, most overloads are collapsed to avoid a large number of
// prototypes which are very similar.
//

//
// Overloaded function for hegst. Its overload differs for
// * MatrixA&
//
template< typename MatrixA, typename MatrixB >
inline std::ptrdiff_t hegst( const fortran_int_t itype,
        const fortran_int_t n, MatrixA& a, const MatrixB& b ) {
    fortran_int_t info(0);
    hegst_impl< typename value< MatrixA >::type >::invoke( itype, n, a,
            b, info );
    return info;
}

//
// Overloaded function for hegst. Its overload differs for
// * const MatrixA&
//
template< typename MatrixA, typename MatrixB >
inline std::ptrdiff_t hegst( const fortran_int_t itype,
        const fortran_int_t n, const MatrixA& a, const MatrixB& b ) {
    fortran_int_t info(0);
    hegst_impl< typename value< MatrixA >::type >::invoke( itype, n, a,
            b, info );
    return info;
}

} // namespace lapack
} // namespace bindings
} // namespace numeric
} // namespace boost

#endif
