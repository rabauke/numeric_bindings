/*
 * 
 * Copyright (c) Kresimir Fresl and Toon Knapen 2002, 2003 
 *
 * Distributed under the Boost Software License, Version 1.0.
 * (See accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 * First author acknowledges the support of the Faculty of Civil Engineering, 
 * University of Zagreb, Croatia.
 *
 */

#ifndef BOOST_NUMERIC_BINDINGS_TRAITS_TYPE_TRAITS_HPP
#define BOOST_NUMERIC_BINDINGS_TRAITS_TYPE_TRAITS_HPP

#include <complex>

namespace boost { namespace numeric { namespace bindings { namespace traits {

  template <typename Real>
  struct type_traits {
  };

  template<>
  struct type_traits<float> {
    typedef float type;
    typedef float real_type;
  };

  template<> 
  struct type_traits<double> {
    typedef double type;
    typedef double real_type;
  };

  template<> 
  struct type_traits<std::complex<float> > {
    typedef std::complex<float> type; 
    typedef float real_type; 
  };

  template<> 
  struct type_traits<std::complex<double> > { 
    typedef std::complex<double> type; 
    typedef double real_type; 
  };


  template<typename T>
  inline T real(const std::complex<T> &x) {
    return x.real();
  }

  template<typename T>
  inline T imag(const std::complex<T> &x) {
    return x.imag();
  }


}}}}

#endif // BOOST_NUMERIC_BINDINGS_TRAITS_TYPE_TRAITS_HPP

