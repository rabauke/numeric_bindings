//
// Copyright (c) 2009 by Rutger ter Borg
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef BOOST_NUMERIC_BINDINGS_TRAITS_IS_REAL_HPP
#define BOOST_NUMERIC_BINDINGS_TRAITS_IS_REAL_HPP

#include <boost/mpl/bool.hpp>

namespace boost {
namespace numeric {
namespace bindings {
namespace traits {

template< typename T >
struct is_real: boost::mpl::bool_<false> {};

template<>
struct is_real< float >: boost::mpl::bool_<true> {};

template<>
struct is_real< double >: boost::mpl::bool_<true> {};

template<>
struct is_real< long double >: boost::mpl::bool_<true> {};

}}}}

#endif
