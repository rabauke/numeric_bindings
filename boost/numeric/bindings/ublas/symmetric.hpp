//
// Copyright (c) 2009 Rutger ter Borg
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef BOOST_NUMERIC_BINDINGS_UBLAS_SYMMETRIC_HPP
#define BOOST_NUMERIC_BINDINGS_UBLAS_SYMMETRIC_HPP

#include <boost/numeric/bindings/detail/adaptor.hpp>
#include <boost/numeric/bindings/ublas/detail/convert_to.hpp>
#include <boost/numeric/ublas/symmetric.hpp>

namespace boost {
namespace numeric {
namespace bindings {
namespace detail {

template< typename T, typename F1, typename F2, typename A, typename Id, typename Enable >
struct adaptor< ublas::symmetric_matrix< T, F1, F2, A >, Id, Enable > {

    typedef typename copy_const< Id, T >::type value_type;
    typedef mpl::map<
        mpl::pair< tag::value_type, value_type >,
        mpl::pair< tag::entity, tag::matrix >,
        mpl::pair< tag::size_type<1>, std::ptrdiff_t >,
        mpl::pair< tag::size_type<2>, std::ptrdiff_t >,
        mpl::pair< tag::matrix_type, tag::symmetric >,
        mpl::pair< tag::data_structure, tag::triangular_array >,
        mpl::pair< tag::data_side, typename convert_to< tag::data_side, F1 >::type >,
        mpl::pair< tag::data_order, typename convert_to< tag::data_order, F2 >::type >
    > property_map;

    static std::ptrdiff_t size1( const Id& t ) {
        return t.size1();
    }

    static std::ptrdiff_t size2( const Id& t ) {
        return t.size2();
    }

    static value_type* begin_value( Id& t ) {
        return bindings::begin_value( t.data() );
    }

    static value_type* end_value( Id& t ) {
        return bindings::end_value( t.data() );
    }

};

} // namespace detail
} // namespace bindings
} // namespace numeric
} // namespace boost

#endif
