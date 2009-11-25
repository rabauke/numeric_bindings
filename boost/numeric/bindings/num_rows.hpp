//
// Copyright (c) 2009 Rutger ter Borg
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef BOOST_NUMERIC_BINDINGS_NUM_ROWS_HPP
#define BOOST_NUMERIC_BINDINGS_NUM_ROWS_HPP

#include <boost/numeric/bindings/size.hpp>

namespace boost {
namespace numeric {
namespace bindings {
namespace result_of {

template< typename T >
struct num_rows {
    typedef typename result_of::size<T,1>::type type;
};

} // namespace result_of

template< typename T >
inline typename result_of::num_rows<T>::type num_rows( T const& t ) {
    return size<1>( t );
}

} // namespace bindings
} // namespace numeric
} // namespace boost

#endif
