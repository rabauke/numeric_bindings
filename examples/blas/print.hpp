#ifndef PRINT_HPP
#define PRINT_HPP

#include <iostream>
#include <iomanip>
#include <boost/numeric/bindings/begin.hpp>
#include <boost/numeric/bindings/end.hpp>
#include <boost/numeric/bindings/size.hpp>
#include <boost/numeric/bindings/trans_tag.hpp>
#include <boost/numeric/bindings/data_order.hpp>
#include <boost/numeric/bindings/blas/detail/default_order.hpp>


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

template<typename Mat>
class printable_mat_cl {
  const Mat &v;
public:
  explicit printable_mat_cl(const Mat &v) : v(v) {
  }
  const Mat & get() const {
    return v;
  }
};

template<typename Mat>
printable_mat_cl<Mat> print_mat(const Mat &v) {
  return printable_mat_cl<Mat>(v);
}

template<typename CharT, typename Traits, typename Vec>
std::basic_ostream<CharT, Traits> & operator<<(std::basic_ostream<CharT, Traits>& out,
					       const printable_vec_cl<Vec> &v) {
  namespace bindings=boost::numeric::bindings;
  auto i=bindings::begin(v.get());
  auto i_end=bindings::end(v.get());
  for (; i!=i_end; ++i)
    out << std::setprecision(3) << std::fixed << (*i) << "  ";
  return out;
}

template<typename CharT, typename Traits, typename Mat>
std::basic_ostream<CharT, Traits> & operator<<(std::basic_ostream<CharT, Traits>& out,
					       const printable_mat_cl<Mat> &M) {
  namespace bindings=boost::numeric::bindings;
  typedef typename bindings::blas::detail::default_order< Mat >::type order;
  typedef typename bindings::result_of::trans_tag< Mat, order >::type trans;
  for (int i=0; i<bindings::size_row_op(M.get(), trans()); ++i) {
    for (int j=0; j<bindings::size_column_op(M.get(), trans()); ++j) 
      out << std::setprecision(3) << std::fixed << M.get()(i, j) << "  ";
    out << '\n';
  }
  return out;
}

#endif
