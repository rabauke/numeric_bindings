#include <cstdlib>
#include <iostream>
#include <complex>
#include <type_traits>
#include <algorithm>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/banded.hpp>
#include <boost/numeric/bindings/trans.hpp>
#include <boost/numeric/bindings/ublas/vector.hpp>
#include <boost/numeric/bindings/ublas/banded.hpp>
#include <boost/numeric/bindings/blas/level2.hpp>
#include "print.hpp"
#include "random.hpp"

namespace ublas=boost::numeric::ublas;
namespace blas=boost::numeric::bindings::blas;

int main(int argc, char *argv[]) {
  {
    typedef std::complex<double> complex;
    typedef ublas::vector<complex> vector;
    typedef ublas::banded_matrix<complex, ublas::column_major> matrix;
    typedef typename std::make_signed<vector::size_type>::type size_type;
    rand_normal<complex>::reset();
    size_type m=6, n=8;
    matrix A(m, n, 1, 2);
    for (size_type j=0; j<n; ++j)
      for (size_type i=std::max(j-2, size_type(0)); i<std::min(j+1+1, m); ++i)
 	A(i, j)=rand_normal<complex>::get();
    // A banded ublas matrux and its transpose have very different
    // memory layouts, thus the following line does not work.
    // matrix A_t(ublas::trans(A));
    matrix A_t(n, m, 2, 1);
    for (size_type j=0; j<m; ++j)
      for (size_type i=std::max(j-1, size_type(0)); i<std::min(j+2+1, n); ++i)
    	A_t(i, j)=A(j, i);
    vector x(n);
    for (size_type i=0; i<n; ++i)
      x(i)=rand_normal<complex>::get();
    vector y(m);
    for (size_type i=0; i<m; ++i)
      y(i)=rand_normal<complex>::get();
    complex alpha(rand_normal<complex>::get());
    complex beta(rand_normal<complex>::get());
    vector y1(alpha*ublas::prod(A, x)+beta*y);
    vector y2(y);
    blas::gbmv(alpha, A, x, beta, y2);
    vector y3(y);
    blas::gbmv(alpha, blas::trans(A_t), x, beta, y3);
    std::cout << "testing boost::ublas containers\n"
	      << "using ublas           : " << print_vec(y1) << '\n'
	      << "using blas            : " << print_vec(y2) << '\n'
	      << "using blas (tranposed): " << print_vec(y3) << '\n'
	      << '\n';
  }
  return EXIT_SUCCESS;
}
