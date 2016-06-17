#ifndef PRINT_HPP
#define PRINT_HPP

#include <iostream>
#include <iomanip>

template<typename Vec>
class printable_vec_cl {
  const Vec &v;
public:
  explicit printable_vec_cl(const Vec &v) : v(v) {
  }
  const Vec & get() const {
    return v;
  }
};

template<typename Vec>
printable_vec_cl<Vec> print_vec(const Vec &v) {
  return printable_vec_cl<Vec>(v);
}

template<typename CharT, typename Traits, typename Vec>
std::basic_ostream<CharT, Traits> & operator<<(std::basic_ostream<CharT, Traits>& out,
					       const printable_vec_cl<Vec> &v) {
  namespace bindings=boost::numeric::bindings;
  auto i=bindings::begin(v.get());
  auto i_end=bindings::end(v.get());
  for (; i!=i_end; ++i)
    out << std::setw(12) << (*i);
  return out;
}

#endif
