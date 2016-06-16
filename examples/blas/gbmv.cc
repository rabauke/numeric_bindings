#include <cstdlib>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/banded.hpp>
#include <Eigen/Core>
#include <boost/numeric/bindings/trans.hpp>
#include <boost/numeric/bindings/ublas/vector.hpp>
#include <boost/numeric/bindings/ublas/banded.hpp>
#include <boost/numeric/bindings/blas/level2.hpp>
#include <iostream>
#include <type_traits>
#include <algorithm>

namespace ublas=boost::numeric::ublas;
namespace blas=boost::numeric::bindings::blas;
namespace bindings=boost::numeric::bindings;

int main(int argc, char *argv[]) {
 {
    typedef ublas::vector<std::complex<double>> vector;
    typedef ublas::banded_matrix<std::complex<double>, ublas::column_major> matrix;
    typedef typename std::make_signed<vector::size_type>::type size_type;
    size_type m=6, n=8;
    matrix A(m, n, 1, 2);
    for (size_type j=0; j<n; ++j)
      for (size_type i=std::max(j-2, size_type(0)); i<std::min(j+1+1, m); ++i)
 	A(i, j)=std::complex<double>(i, j);
    vector x(n);
    for (size_type i=0; i<n; ++i)
      x(i)=i;
    vector y(m);
    for (size_type i=0; i<m; ++i)
      y(i)=std::complex<double>(0, i);
    std::complex<double> alpha(0, 1);
    std::complex<double> beta(2, 0);
    vector y2(alpha*ublas::prod(A, x)+beta*y);
    blas::gbmv(alpha, A, x, beta, y);
    std::cout << y << '\n'
	      << y2 << '\n';
  }
  {
    typedef ublas::vector<std::complex<double>> vector;
    typedef ublas::banded_matrix<std::complex<double>, ublas::column_major> matrix;
    typedef typename std::make_signed<vector::size_type>::type size_type;
    size_type m=6, n=8;
    matrix A(n, m, 2, 1);
    for (size_type j=0; j<m; ++j)
      for (size_type i=std::max(j-1, size_type(0)); i<std::min(j+2+1, n); ++i)
 	A(i, j)=std::complex<double>(j, i);
    vector x(n);
    for (size_type i=0; i<n; ++i)
      x(i)=i;
    vector y(m);
    for (size_type i=0; i<m; ++i)
      y(i)=std::complex<double>(0, i);
    std::complex<double> alpha(0, 1);
    std::complex<double> beta(2, 0);
    vector y2(alpha*ublas::prod(ublas::trans(A), x)+beta*y);
    blas::gbmv(alpha, bindings::trans(A), x, beta, y);
    std::cout << y << '\n'
	      << y2 << '\n';
  }
  return EXIT_SUCCESS;
}
