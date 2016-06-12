#include <cstdlib>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <Eigen/Core>
#include <boost/numeric/bindings/ublas/vector.hpp>
#include <boost/numeric/bindings/ublas/matrix_proxy.hpp>
#include <boost/numeric/bindings/ublas/matrix_expression.hpp>
#include <boost/numeric/bindings/eigen/vector.hpp>
#include <boost/numeric/bindings/blas/level1.hpp>
#include <iostream>

namespace ublas=boost::numeric::ublas;
namespace blas=boost::numeric::bindings::blas;

int main(int argc, char *argv[]) {
  {
    typedef ublas::vector<std::complex<double>> vector;
    typedef ublas::matrix<std::complex<double>> matrix;
    typedef vector::size_type size_type;
    size_type n=8;
    vector v(n);
    for (size_type i=0; i<n; ++i)
      v(i)=i;
    std::cout << blas::asum(v) << '\n';
    matrix M(n, n);
    for (size_type j=0; j<n; ++j)
      for (size_type i=0; i<n; ++i) 
     	M(i, j)=std::complex<double>(i, j);
    ublas::matrix_column<matrix> mc(M, 2);
    ublas::matrix_row<matrix> mr(M, 3);
    std::cout << blas::asum(mc) << '\n'
	      << blas::asum(mr) << '\n';
  }
  {
    typedef Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1> vector;
    typedef Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> matrix;
    typedef int size_type;
    size_type n=8;
    vector v(n);
    for (size_type i=0; i<n; ++i)
      v(i)=i;
    std::cout << blas::asum(v) << '\n';
    matrix M(n, n);
    for (size_type j=0; j<n; ++j)
      for (size_type i=0; i<n; ++i) 
     	M(i, j)=std::complex<double>(i, j);
    auto mc=M.col(2);
    auto mr=M.row(3);
    std::cout << blas::asum(mc) << '\n'
	      << blas::asum(mr) << '\n';
  }
  return EXIT_SUCCESS;
}
