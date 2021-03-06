//
// Copyright (c) 2016 Heiko Bauke
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef BOOST_NUMERIC_BINDINGS_EIGEN_VECTOR_HPP
#define BOOST_NUMERIC_BINDINGS_EIGEN_VECTOR_HPP

#include <boost/numeric/bindings/detail/adaptor.hpp>
#include <boost/numeric/bindings/begin.hpp>
#include <boost/numeric/bindings/end.hpp>
#include <Eigen/Core>

namespace boost {
namespace numeric {
namespace bindings {
namespace detail {

template< typename T, int S, int Options, typename Id, typename Enable >
struct adaptor< Eigen::Matrix< T, S, 1, Options >, Id, Enable > {

    typedef typename copy_const< Id, T >::type value_type;
    typedef mpl::map<
        mpl::pair< tag::value_type, value_type >,
        mpl::pair< tag::entity, tag::vector >,
        mpl::pair< tag::size_type<1>, std::ptrdiff_t >,
        mpl::pair< tag::data_structure, tag::linear_array >,
        mpl::pair< tag::stride_type<1>, tag::contiguous >
    > property_map;

    static std::ptrdiff_t size1( const Id& id ) {
        return id.rows();
    }

    static value_type* begin_value( Id& id ) {
      return id.data();
    }

    static value_type* end_value( Id& id ) {
      return id.data() + id.size();
    }

};


template< typename T, int S, int Options, typename Id, typename Enable >
struct adaptor< Eigen::Matrix< T, 1, S, Options >, Id, Enable > {

    typedef typename copy_const< Id, T >::type value_type;
    typedef mpl::map<
        mpl::pair< tag::value_type, value_type >,
        mpl::pair< tag::entity, tag::vector >,
        mpl::pair< tag::size_type<1>, std::ptrdiff_t >,
        mpl::pair< tag::data_structure, tag::linear_array >,
        mpl::pair< tag::stride_type<1>, tag::contiguous >
    > property_map;

    static std::ptrdiff_t size1( const Id& id ) {
        return id.cols();
    }

    static value_type* begin_value( Id& id ) {
      return id.data();
    }

    static value_type* end_value( Id& id ) {
      return id.data() + id.size();
    }

};


template< typename T, int S, bool q, typename Id, typename Enable >
struct adaptor< Eigen::Block< T, 1, S, q >, Id, Enable > {

    typedef typename copy_const< Id, typename T::value_type >::type value_type;
    typedef mpl::map<
        mpl::pair< tag::value_type, value_type >,
        mpl::pair< tag::entity, tag::vector >,
        mpl::pair< tag::size_type<1>, std::ptrdiff_t >,
        mpl::pair< tag::data_structure, tag::linear_array >,
        mpl::pair< tag::stride_type<1>, std::ptrdiff_t >
    > property_map;

    static std::ptrdiff_t size1( const Id& id ) {
        return id.size();
    }

    static value_type* begin_value( Id& id ) {
      return id.data();
    }

    static value_type* end_value( Id& id ) {
      return id.data() + size1(id)*stride1(id);
    }

    static std::ptrdiff_t stride1( const Id& id ) {
      return id.colStride();
    }

};


template< typename T, int S, bool q, typename Id, typename Enable >
struct adaptor< Eigen::Block< T, S, 1, q >, Id, Enable > {

    typedef typename copy_const< Id, typename T::value_type >::type value_type;
    typedef mpl::map<
        mpl::pair< tag::value_type, value_type >,
        mpl::pair< tag::entity, tag::vector >,
        mpl::pair< tag::size_type<1>, std::ptrdiff_t >,
        mpl::pair< tag::data_structure, tag::linear_array >,
        mpl::pair< tag::stride_type<1>, std::ptrdiff_t >
    > property_map;

    static std::ptrdiff_t size1( const Id& id ) {
        return id.size();
    }

    static value_type* begin_value( Id& id ) {
      return id.data();
    }

    static value_type* end_value( Id& id ) {
      return id.data() + size1(id)*stride1(id);
    }

    static std::ptrdiff_t stride1( const Id& id ) {
      return id.rowStride();
    }

};

} // namespace detail
} // namespace bindings
} // namespace numeric
} // namespace boost

#endif
