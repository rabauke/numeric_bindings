#include <cstdlib>
#include <iostream>
#include <complex>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <Eigen/Core>
#include <boost/numeric/bindings/ublas/vector.hpp>
#include <boost/numeric/bindings/ublas/matrix.hpp>
#include <boost/numeric/bindings/eigen/vector.hpp>
#include <boost/numeric/bindings/eigen/matrix.hpp>
#include <boost/numeric/bindings/blas/level3.hpp>
#include <boost/numeric/bindings/lower.hpp>
#include <boost/numeric/bindings/upper.hpp>
#include <boost/numeric/bindings/left.hpp>
#include <boost/numeric/bindings/right.hpp>
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
    {
      size_type m=6, n=8;
      matrix A(m, m);
      matrix B(m, n);
      matrix C(m, n);
      for (size_type j=0; j<m; ++j)
	for (size_type i=0; i<j; ++i) {
	  A(i, j)=rand_normal<complex>::get();
	  A(j, i)=A(i, j);
	}
      for (size_type j=0; j<n; ++j)
	for (size_type i=0; i<m; ++i) 
	  B(i, j)=rand_normal<complex>::get();
      for (size_type j=0; j<n; ++j)
	for (size_type i=0; i<m; ++i)
	  C(i, j)=rand_normal<complex>::get();
      complex alpha(rand_normal<complex>::get());
      complex beta(rand_normal<complex>::get());
      matrix C1(alpha*ublas::prod(A, B)+beta*C);
      matrix C2(C);
      blas::symm(blas::left(), alpha, blas::upper(A), B, beta, C2);
      std::cout << "testing boost::ublas containers\n"
		<< "using ublas (left multiply):\n" << print_mat(C1) << '\n'
		<< "using blas (left multiply):\n" << print_mat(C2) << '\n'
		<< '\n';
    }
    {
      size_type m=6, n=8;
      matrix A(n, n);
      matrix B(m, n);
      matrix C(m, n);
      for (size_type j=0; j<n; ++j)
	for (size_type i=0; i<j; ++i) {
	  A(i, j)=rand_normal<complex>::get();
	  A(j, i)=A(i, j);
	}
      for (size_type j=0; j<n; ++j)
	for (size_type i=0; i<m; ++i) 
	  B(i, j)=rand_normal<complex>::get();
      for (size_type j=0; j<n; ++j)
	for (size_type i=0; i<m; ++i)
	  C(i, j)=rand_normal<complex>::get();
      complex alpha(rand_normal<complex>::get());
      complex beta(rand_normal<complex>::get());
      matrix C1(alpha*ublas::prod(B, A)+beta*C);
      matrix C2(C);
      blas::symm(blas::right(), alpha, blas::upper(A), B, beta, C2);
      std::cout << "testing boost::ublas containers\n"
		<< "using ublas (right multiply):\n" << print_mat(C1) << '\n'
		<< "using blas (right multiply):\n" << print_mat(C2) << '\n'
		<< '\n';
    }
  }
  {
    typedef std::complex<double> complex;
    typedef Eigen::Matrix<complex, Eigen::Dynamic, 1> vector;
    typedef Eigen::Matrix<complex, Eigen::Dynamic, Eigen::Dynamic> matrix;
    typedef int size_type;
    rand_normal<complex>::reset();
    {
      size_type m=6, n=8;
      matrix A(m, m);
      matrix B(m, n);
      matrix C(m, n);
      for (size_type j=0; j<m; ++j)
	for (size_type i=0; i<j; ++i) {
	  A(i, j)=rand_normal<complex>::get();
	  A(j, i)=A(i, j);
	}
      for (size_type j=0; j<n; ++j)
	for (size_type i=0; i<m; ++i) 
	  B(i, j)=rand_normal<complex>::get();
      for (size_type j=0; j<n; ++j)
	for (size_type i=0; i<m; ++i)
	  C(i, j)=rand_normal<complex>::get();
      complex alpha(rand_normal<complex>::get());
      complex beta(rand_normal<complex>::get());
      matrix C1(alpha*A*B+beta*C);
      matrix C2(C);
      blas::symm(blas::left(), alpha, blas::upper(A), B, beta, C2);
      std::cout << "testing Eigen containers\n"
		<< "using Eigen (left multiply):\n" << print_mat(C1) << '\n'
		<< "using blas (left multiply):\n" << print_mat(C2) << '\n'
		<< '\n';
    }
    {
      size_type m=6, n=8;
      matrix A(n, n);
      matrix B(m, n);
      matrix C(m, n);
      for (size_type j=0; j<n; ++j)
	for (size_type i=0; i<j; ++i) {
	  A(i, j)=rand_normal<complex>::get();
	  A(j, i)=A(i, j);
	}
      for (size_type j=0; j<n; ++j)
	for (size_type i=0; i<m; ++i) 
	  B(i, j)=rand_normal<complex>::get();
      for (size_type j=0; j<n; ++j)
	for (size_type i=0; i<m; ++i)
	  C(i, j)=rand_normal<complex>::get();
      complex alpha(rand_normal<complex>::get());
      complex beta(rand_normal<complex>::get());
      matrix C1(alpha*B*A+beta*C);
      matrix C2(C);
      blas::symm(blas::right(), alpha, blas::upper(A), B, beta, C2);
      std::cout << "testing Eigen containers\n"
		<< "using Eigen (right multiply):\n" << print_mat(C1) << '\n'
		<< "using blas (right multiply):\n" << print_mat(C2) << '\n'
		<< '\n';
    }
  }
  return EXIT_SUCCESS;
}
