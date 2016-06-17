#include <cstdlib>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <Eigen/Core>
#include <boost/numeric/bindings/trans.hpp>
#include <boost/numeric/bindings/ublas/vector.hpp>
#include <boost/numeric/bindings/ublas/matrix.hpp>
#include <boost/numeric/bindings/eigen/vector.hpp>
#include <boost/numeric/bindings/eigen/matrix.hpp>
#include <boost/numeric/bindings/blas/level2.hpp>
#include <iostream>
#include "print.hpp"

namespace ublas=boost::numeric::ublas;
namespace blas=boost::numeric::bindings::blas;

int main(int argc, char *argv[]) {
  {
    typedef ublas::vector<std::complex<double>> vector;
    typedef ublas::matrix<std::complex<double>, ublas::column_major> matrix;
    typedef vector::size_type size_type;
    size_type m=6, n=8;
    matrix A(m, n);
    for (size_type j=0; j<n; ++j)
      for (size_type i=0; i<m; ++i) 
 	A(i, j)=std::complex<double>(i, j);
    matrix A_t(ublas::trans(A));
    vector x(n);
    for (size_type i=0; i<n; ++i)
      x(i)=i;
    vector y(m);
    for (size_type i=0; i<m; ++i)
      y(i)=std::complex<double>(0, i);
    std::complex<double> alpha(0, 1);
    std::complex<double> beta(2, 0);
    vector y1(alpha*ublas::prod(A, x)+beta*y);
    vector y2(y);
    blas::gemv(alpha, A, x, beta, y2);
    vector y3(y);
    blas::gemv(alpha, blas::trans(A_t), x, beta, y3);
    std::cout << "testing boost::ublas containers\n"
	      << "using ublas           : " << print_vec(y1) << '\n'
	      << "using blas            : " << print_vec(y2) << '\n'
	      << "using blas (tranposed): " << print_vec(y3) << '\n'
	      << '\n';
  }
  {
    typedef Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1> vector;
    typedef Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> matrix;
    typedef int size_type;
    size_type m=6, n=8;
    matrix A(m, n);
    for (size_type j=0; j<n; ++j)
      for (size_type i=0; i<m; ++i) 
  	A(i, j)=std::complex<double>(i, j);
    matrix A_t(A.transpose());
    vector x(n);
    for (size_type i=0; i<n; ++i)
      x(i)=i;
    vector y(m);
    for (size_type i=0; i<m; ++i)
      y(i)=std::complex<double>(0, i);
    std::complex<double> alpha(0, 1);
    std::complex<double> beta(2, 0);
    vector y1(alpha*A*x+beta*y);
    vector y2(y);
    blas::gemv(alpha, A, x, beta, y2);
    vector y3(y);
    blas::gemv(alpha, blas::trans(A_t), x, beta, y3);
    std::cout << "testing eigen++ containers\n"
	      << "using eigen++         : " << print_vec(y1) << '\n'
	      << "using blas            : " << print_vec(y2) << '\n'
	      << "using blas (tranposed): " << print_vec(y3) << '\n'
	      << '\n';
  }
  return EXIT_SUCCESS;
}
