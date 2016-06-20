#include <cstdlib>
#include <iostream>
#include <complex>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <Eigen/Core>
#include <boost/numeric/bindings/trans.hpp>
#include <boost/numeric/bindings/ublas/vector.hpp>
#include <boost/numeric/bindings/ublas/matrix.hpp>
#include <boost/numeric/bindings/eigen/vector.hpp>
#include <boost/numeric/bindings/eigen/matrix.hpp>
#include <boost/numeric/bindings/blas/level2.hpp>
#include "print.hpp"
#include "random.hpp"

namespace ublas=boost::numeric::ublas;
namespace blas=boost::numeric::bindings::blas;

int main(int argc, char *argv[]) {
  {
    typedef std::complex<double> complex;
    typedef ublas::vector<complex> vector;
    typedef ublas::matrix<complex, ublas::column_major> matrix;
    typedef vector::size_type size_type;
    rand_normal<complex>::reset();
    size_type m=6, n=8;
    matrix A(m, n);
    for (size_type j=0; j<n; ++j)
      for (size_type i=0; i<m; ++i) 
 	A(i, j)=rand_normal<complex>::get();
    matrix A_t(ublas::trans(A));
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
    typedef std::complex<double> complex;
    typedef Eigen::Matrix<complex, Eigen::Dynamic, 1> vector;
    typedef Eigen::Matrix<complex, Eigen::Dynamic, Eigen::Dynamic> matrix;
    typedef int size_type;
    size_type m=6, n=8;
    rand_normal<complex>::reset();
    matrix A(m, n);
    for (size_type j=0; j<n; ++j)
      for (size_type i=0; i<m; ++i) 
  	A(i, j)=rand_normal<complex>::get();
    matrix A_t(A.transpose());
    vector x(n);
    for (size_type i=0; i<n; ++i)
      x(i)=rand_normal<complex>::get();
    vector y(m);
    for (size_type i=0; i<m; ++i)
      y(i)=rand_normal<complex>::get();
    complex alpha(rand_normal<complex>::get());
    complex beta(rand_normal<complex>::get());
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
