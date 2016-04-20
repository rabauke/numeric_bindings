#ifndef boost_numeric_bindings_type_hpp
#define boost_numeric_bindings_type_hpp

// This header provides typedefs to float complex and double complex.
// This makes it possible to redefine the complex class being used.


namespace boost { namespace numeric { namespace bindings { namespace traits {

  /* The types for single and double precision complex numbers.
   * You can use your own types if you define
   * BOOST_NUMERIC_BINDINGS_USE_CUSTOM_COMPLEX_TYPE.
   * Note that these types must have the same memory layout as the
   * corresponding FORTRAN types.
   * For that reason you can even use a different type in each translation
   * unit and the resulting binary will still work!
   */
#ifndef BOOST_NUMERIC_BINDINGS_USE_CUSTOM_COMPLEX_TYPE
#include <complex>
  typedef std::complex< float >  complex_f ;
  typedef std::complex< double > complex_d ;

  inline float  real (complex_f const& c) { return std::real (c); }
  inline float  imag (complex_f const& c) { return std::imag (c); }
  inline double real (complex_d const& c) { return std::real (c); }
  inline double imag (complex_d const& c) { return std::imag (c); }

#endif

  struct null_t {};

}}}}

#endif // boost_numeric_bindings_type_hpp
