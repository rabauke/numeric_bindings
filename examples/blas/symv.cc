#include <cstdlib>
#include <iostream>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <Eigen/Core>
#include <boost/numeric/bindings/lower.hpp>
#include <boost/numeric/bindings/upper.hpp>
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
    typedef ublas::vector<double> vector;
    typedef ublas::matrix<double, ublas::column_major> matrix;
    typedef vector::size_type size_type;
    rand_normal<double>::reset();
    size_type n=8;
    matrix A(n, n), A_u(n, n), A_l(n, n);
    for (size_type j=0; j<n; ++j) {
      A(j, j)=rand_normal<double>::get();
      A_u(j, j)=A(j, j);
      A_l(j, j)=A(j, j);
      for (size_type i=0; i<j; ++i) {
	A(i, j)=rand_normal<double>::get();
	A(j, i)=A(i, j);
	A_u(i, j)=A(i, j);
	A_u(j, i)=0;
	A_l(i, j)=0;
	A_l(j, i)=A(j, i);
      }
    }
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
    blas::symv(alpha, blas::lower(A_l), x, beta, y2);
    vector y3(y);
    blas::symv(alpha, blas::upper(A_u), x, beta, y3);
    std::cout << "testing boost::ublas containers\n"
    	      << "using ublas       : " << print_vec(y1) << '\n'
    	      << "using blas (lower): " << print_vec(y2) << '\n'
    	      << "using blas (upper): " << print_vec(y3) << '\n'
    	      << '\n';
  }
  {
    typedef Eigen::Matrix<double, Eigen::Dynamic, 1> vector;
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> matrix;
    typedef int size_type;
    rand_normal<double>::reset();
    size_type n=8;
    matrix A(n, n), A_u(n, n), A_l(n, n);
    for (size_type j=0; j<n; ++j) {
      A(j, j)=rand_normal<double>::get();
      A_u(j, j)=A(j, j);
      A_l(j, j)=A(j, j);
      for (size_type i=0; i<j; ++i) {
    	A(i, j)=rand_normal<double>::get();
    	A(j, i)=A(i, j);
    	A_u(i, j)=A(i, j);
    	A_u(j, i)=0;
    	A_l(i, j)=0;
    	A_l(j, i)=A(j, i);
      }
    }
    vector x(n);
    for (size_type i=0; i<n; ++i)
      x(i)=rand_normal<double>::get();
    vector y(n);
    for (size_type i=0; i<n; ++i)
      y(i)=rand_normal<double>::get();
    double alpha(rand_normal<double>::get());
    double beta(rand_normal<double>::get());
    vector y1(alpha*A*x+beta*y);
    vector y2(y);
    blas::symv(alpha, blas::lower(A_l), x, beta, y2);
    vector y3(y);
    blas::symv(alpha, blas::upper(A_u), x, beta, y3);
    std::cout << "testing Eigen containers\n"
    	      << "using Eigen       : " << print_vec(y1) << '\n'
    	      << "using blas (lower): " << print_vec(y2) << '\n'
    	      << "using blas (upper): " << print_vec(y3) << '\n'
    	      << '\n';
  }
  return EXIT_SUCCESS;
}
