//
// Copyright (c) 2009 Rutger ter Borg
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef BOOST_NUMERIC_BINDINGS_UPPER_HPP
#define BOOST_NUMERIC_BINDINGS_UPPER_HPP

#include <boost/numeric/bindings/detail/basic_wrapper.hpp>
#include <boost/numeric/bindings/tag.hpp>

namespace boost {
namespace numeric {
namespace bindings {
namespace result_of {

template< typename T >
struct upper {
    typedef detail::basic_wrapper<
        T,
        mpl::pair< tag::matrix_type, tag::triangular >,
        mpl::pair< tag::data_side, tag::upper >
    > type;
};

} // namespace result_of

template< typename T >
const typename result_of::upper< T >::type upper( T& underlying ) {
    return typename result_of::upper< T >::type( underlying );
}

template< typename T >
const typename result_of::upper< const T >::type upper( const T& underlying ) {
    return typename result_of::upper< const T >::type( underlying );
}

namespace blas {

using boost::numeric::bindings::upper;

} // namespace blas

namespace lapack {

using boost::numeric::bindings::upper;

} // namespace lapack

} // namespace bindings
} // namespace numeric
} // namespace boost

#endif
