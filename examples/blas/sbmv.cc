#include <cstdlib>
#include <iostream>
#include <complex>
#include <type_traits>
#include <algorithm>
#define BOOST_UBLAS_NO_ELEMENT_PROXIES
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/banded.hpp>
#include <boost/numeric/ublas/symmetric.hpp>
#include <boost/numeric/bindings/ublas/vector.hpp>
#include <boost/numeric/bindings/ublas/banded.hpp>
#include <boost/numeric/bindings/ublas/symmetric.hpp>
#include <boost/numeric/bindings/blas/level2.hpp>
#include "print.hpp"
#include "random.hpp"

namespace ublas=boost::numeric::ublas;
namespace blas=boost::numeric::bindings::blas;

int main(int argc, char *argv[]) {
  {
    typedef ublas::vector<double> vector;
    typedef ublas::banded_matrix<double, ublas::column_major> matrix;
    typedef typename std::make_signed<vector::size_type>::type size_type;
    rand_normal<double>::reset();
    size_type n=8;
    matrix A(n, n, 2, 2);
    matrix A_u(n, n, 0, 2), A_l(n, n, 2, 0);
    for (size_type j=0; j<n; ++j) {
      A(j, j)=rand_normal<double>::get();
      A_u(j, j)=A(j, j);
      A_l(j, j)=A(j, j);
      for (size_type i=std::max(size_type(0), j-2); i<j; ++i) {
	A(i, j)=rand_normal<double>::get();
	A(j, i)=A(i, j);
	A_u(i, j)=A(i, j);
	A_l(j, i)=A(j, i);
      }
    }
    ublas::symmetric_adaptor<matrix, ublas::upper> B_u(A_u);
    ublas::symmetric_adaptor<matrix, ublas::lower> B_l(A_l);
    vector x(n);
    for (size_type i=0; i<n; ++i)
      x(i)=rand_normal<double>::get();
    vector y(n);
    for (size_type i=0; i<n; ++i)
      y(i)=rand_normal<double>::get();
    double alpha(rand_normal<double>::get());
    double beta(rand_normal<double>::get());
    vector y1(alpha*ublas::prod(A, x)+beta*y);
    vector y2(y);
    blas::sbmv(alpha, B_l, x, beta, y2);
    vector y3(y);
    blas::sbmv(alpha, B_u, x, beta, y3);
    vector y4(y);
    blas::hbmv(alpha, B_l, x, beta, y4);
    vector y5(y);
    blas::hbmv(alpha, B_u, x, beta, y5);
    std::cout << "testing boost::ublas containers\n"
    	      << "using ublas       : " << print_vec(y1) << '\n'
	      << "using blas sbmv (lower): " << print_vec(y2) << '\n'
	      << "using blas sbmv (upper): " << print_vec(y3) << '\n'
	      << "using blas hbmv (lower): " << print_vec(y4) << '\n'
	      << "using blas hbmv (upper): " << print_vec(y5) << '\n'
    	      << '\n';
  }
  return EXIT_SUCCESS;
}
