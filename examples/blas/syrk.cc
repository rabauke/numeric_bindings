#include <cstdlib>
#include <iostream>
#include <complex>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/bindings/lower.hpp>
#include <boost/numeric/bindings/upper.hpp>
#include <boost/numeric/bindings/ublas/matrix.hpp>
#include <boost/numeric/bindings/blas/level3.hpp>
#include <boost/numeric/bindings/trans.hpp>
#include "print.hpp"
#include "random.hpp"

namespace ublas=boost::numeric::ublas;
namespace blas=boost::numeric::bindings::blas;

int main(int argc, char *argv[]) {
  {
    typedef std::complex<double> complex;
    typedef ublas::matrix<complex, ublas::column_major> matrix;
    typedef matrix::size_type size_type;
    rand_normal<complex>::reset();
    size_type n=8;
    matrix A(n, n);
    for (size_type j=0; j<n; ++j) {
      A(j, j)=rand_normal<complex>::get();
      for (size_type i=0; i<n; ++i) {
	A(i, j)=rand_normal<complex>::get();
      }
    }
    matrix C(n, n);
    for (size_type j=0; j<n; ++j) {
      for (size_type i=0; i<=j; ++i) {
	C(i, j)=rand_normal<complex>::get();
	C(j, i)=C(i, j);
      }
    }
    complex alpha(rand_normal<complex>::get());
    complex beta(rand_normal<complex>::get());
    {
      matrix C1(alpha*ublas::prod(A, ublas::trans(A))+beta*C);
      for (size_type j=0; j<n; ++j)
	for (size_type i=j+1; i<n; ++i)
	  C1(i, j)=C(i, j);
      matrix C2(C);
      blas::syrk(alpha, A, beta, blas::upper(C2));
      std::cout << "testing boost::ublas containers\n"
		<< "using ublas (A right trans, C upper):\n" << print_mat(C1) << '\n'
		<< "using blas (A right trans, C upper):\n" << print_mat(C2) << '\n'
		<< '\n';
    }
    {
      matrix C1(alpha*ublas::prod(A, ublas::trans(A))+beta*C);
      for (size_type j=0; j<n; ++j)
	for (size_type i=0; i<j; ++i)
	  C1(i, j)=C(i, j);
      matrix C2(C);
      blas::syrk(alpha, A, beta, blas::lower(C2));
      std::cout << "testing boost::ublas containers\n"
		<< "using ublas (A right trans, C lower):\n" << print_mat(C1) << '\n'
		<< "using blas (A right trans, C lower):\n" << print_mat(C2) << '\n'
		<< '\n';
    }
    {
      matrix C1(alpha*ublas::prod(ublas::trans(A), A)+beta*C);
      for (size_type j=0; j<n; ++j)
	for (size_type i=j+1; i<n; ++i)
	  C1(i, j)=C(i, j);
      matrix C2(C);
      blas::syrk(alpha, blas::trans(A), beta, blas::upper(C2));
      std::cout << "testing boost::ublas containers\n"
		<< "using ublas (A left trans, C upper):\n" << print_mat(C1) << '\n'
		<< "using blas (A left trans, C upper):\n" << print_mat(C2) << '\n'
		<< '\n';
    }
    {
      matrix C1(alpha*ublas::prod(ublas::trans(A), A)+beta*C);
      for (size_type j=0; j<n; ++j)
	for (size_type i=0; i<j; ++i)
	  C1(i, j)=C(i, j);
      matrix C2(C);
      blas::syrk(alpha, blas::trans(A), beta, blas::lower(C2));
      std::cout << "testing boost::ublas containers\n"
		<< "using ublas (A left trans, C lower):\n" << print_mat(C1) << '\n'
		<< "using blas (A left trans, C lower):\n" << print_mat(C2) << '\n'
		<< '\n';
    }
  }
  return EXIT_SUCCESS;
}
