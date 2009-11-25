//
// Copyright (c) 2009 Rutger ter Borg
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef BOOST_NUMERIC_BINDINGS_ROW_HPP
#define BOOST_NUMERIC_BINDINGS_ROW_HPP

#include <boost/numeric/bindings/data.hpp>
#include <boost/numeric/bindings/detail/adaptable_type.hpp>
#include <boost/numeric/bindings/size.hpp>
#include <boost/numeric/bindings/stride.hpp>
#include <boost/numeric/bindings/value_type.hpp>
#include <boost/ref.hpp>

namespace boost {
namespace numeric {
namespace bindings {
namespace detail {

template< typename T >
struct row_wrapper:
        adaptable_type< row_wrapper<T> >,
        reference_wrapper<T> {

    row_wrapper( T& t, std::size_t index ):
        reference_wrapper<T>(t),
        m_index( index ) {}

    std::size_t m_index;
};

template< typename T, typename Id, typename Enable >
struct adaptor< row_wrapper<T>, Id, Enable > {

    typedef typename value_type<T>::type value_type;
    typedef mpl::map<
        mpl::pair< tag::value_type, value_type >,
        mpl::pair< tag::entity, tag::vector >,
        mpl::pair< tag::size_type<1>, typename result_of::size<T,2>::type >,
        mpl::pair< tag::data_structure, tag::linear_array >,
        mpl::pair< tag::stride_type<1>, typename result_of::stride<T,2>::type >
    > property_map;

    static typename result_of::size<T,2>::type size1( Id const& id ) {
        return size<2>( id.get() );
    }

    static typename result_of::data<T>::type data( Id& id ) {
        return bindings::data( id.get() ) + id.m_index * stride<1>( id.get() );
    }

    static typename result_of::stride<T,2>::type stride1( Id const& id ) {
        return stride<2>( id.get() );
    }

};

} // namespace detail

namespace result_of {

template< typename T >
struct row {
    typedef detail::row_wrapper<T> type;
};

} // namespace result_of

template< typename T >
detail::row_wrapper<T> row( T& underlying, std::size_t index ) {
    return detail::row_wrapper<T>( underlying, index );
}

template< typename T >
detail::row_wrapper<T const> row( T const& underlying, std::size_t index ) {
    return detail::row_wrapper<T const>( underlying, index );
}

template< int N, typename T >
void row( T const& underlying ) {

}

} // namespace bindings
} // namespace numeric
} // namespace boost

#endif
