#include <cstdlib>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <Eigen/Core>
#include <boost/numeric/bindings/ublas/vector.hpp>
#include <boost/numeric/bindings/ublas/matrix.hpp>
#include <boost/numeric/bindings/eigen/vector.hpp>
#include <boost/numeric/bindings/eigen/matrix.hpp>
#include <boost/numeric/bindings/blas/level2.hpp>
#include <iostream>

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
    vector x(n);
    for (size_type i=0; i<n; ++i)
      x(i)=i;
    vector y(m);
    for (size_type i=0; i<m; ++i)
      y(i)=std::complex<double>(0, i);
    std::complex<double> alpha(0, 1);
    std::complex<double> beta(2, 0);
    vector y2(alpha*ublas::prod(A, x)+beta*y);
    blas::gemv(alpha, A, x, beta, y);
    std::cout << y << '\n'
	      << y2 << '\n';
  }
  {
    typedef ublas::vector<std::complex<double>> vector;
    typedef ublas::matrix<std::complex<double>, ublas::column_major> matrix;
    typedef vector::size_type size_type;
    size_type m=6, n=8;
    matrix A(n, m);
    for (size_type j=0; j<m; ++j)
      for (size_type i=0; i<n; ++i) 
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
    blas::gemv(alpha, ublas::trans(A), x, beta, y);
    std::cout << y << '\n'
	      << y2 << '\n';
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
    vector x(n);
    for (size_type i=0; i<n; ++i)
      x(i)=i;
    vector y(m);
    for (size_type i=0; i<m; ++i)
      y(i)=std::complex<double>(0, i);
    std::complex<double> alpha(0, 1);
    std::complex<double> beta(2, 0);
    vector y2(alpha*A*x+beta*y);
    blas::gemv(alpha, A, x, beta, y);
    std::cout << y << '\n'
	      << y2 << '\n';
  }
  {
    typedef Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1> vector;
    typedef Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> matrix;
    typedef int size_type;
    size_type m=6, n=8;
    matrix A(n, m);
    for (size_type j=0; j<m; ++j)
      for (size_type i=0; i<n; ++i) 
 	A(i, j)=std::complex<double>(j, i);
    vector x(n);
    for (size_type i=0; i<n; ++i)
      x(i)=i;
    vector y(m);
    for (size_type i=0; i<m; ++i)
      y(i)=std::complex<double>(0, i);
    std::complex<double> alpha(0, 1);
    std::complex<double> beta(2, 0);
    const matrix &B(A);
    vector y2(alpha*B.transpose()*x+beta*y);
    blas::gemv(alpha, B.transpose(), x, beta, y);
    std::cout << y << '\n'
         << y2 << '\n';
  }
  return EXIT_SUCCESS;
}
