#include <cstdlib>
#include <iostream>
#include <complex>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/bindings/lower.hpp>
#include <boost/numeric/bindings/upper.hpp>
#include <boost/numeric/bindings/ublas/matrix.hpp>
#include <boost/numeric/bindings/blas/level3.hpp>
#include <boost/numeric/bindings/conj.hpp>
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
    matrix A(n, n), B(n, n), C(n, n);
    for (size_type j=0; j<n; ++j)
      for (size_type i=0; i<n; ++i) {
	A(i, j)=rand_normal<complex>::get();
	B(i, j)=rand_normal<complex>::get();
      }
    for (size_type j=0; j<n; ++j) {
      for (size_type i=0; i<j; ++i) {
	C(i, j)=rand_normal<complex>::get();
	C(j, i)=std::conj(C(i, j));
      }
      C(j, j)=rand_normal<complex>::get().real();
    }
    complex alpha(rand_normal<complex>::get());
    double beta(rand_normal<complex>::get().real());
    {
      matrix C1(alpha*ublas::prod(A, ublas::trans(ublas::conj(B)))+
		std::conj(alpha)*ublas::prod(B, ublas::trans(ublas::conj(A)))+
		beta*C);
      for (size_type j=0; j<n; ++j)
	for (size_type i=j+1; i<n; ++i)
	  C1(i, j)=C(i, j);
      matrix C2(C);
      blas::her2k(alpha, A, B, beta, blas::upper(C2));
      std::cout << "testing boost::ublas containers\n"
		<< "using ublas (A right htrans, C upper):\n" << print_mat(C1) << '\n'
		<< "using blas (A right htrans, C upper):\n" << print_mat(C2) << '\n'
		<< '\n';
    }
    {
      matrix C1(alpha*ublas::prod(A, ublas::trans(ublas::conj(B)))+
		std::conj(alpha)*ublas::prod(B, ublas::trans(ublas::conj(A)))+
		beta*C);
      for (size_type j=0; j<n; ++j)
    	for (size_type i=0; i<j; ++i)
    	  C1(i, j)=C(i, j);
      matrix C2(C);
      blas::her2k(alpha, A, B, beta, blas::lower(C2));
      std::cout << "testing boost::ublas containers\n"
    		<< "using ublas (A right htrans, C lower):\n" << print_mat(C1) << '\n'
    		<< "using blas (A right htrans, C lower):\n" << print_mat(C2) << '\n'
    		<< '\n';
    }
    {
      matrix C1(alpha*ublas::prod(ublas::trans(ublas::conj(A)), B)+
		std::conj(alpha)*ublas::prod(ublas::trans(ublas::conj(B)), A)+
		beta*C);
      for (size_type j=0; j<n; ++j)
    	for (size_type i=j+1; i<n; ++i)
    	  C1(i, j)=C(i, j);
      matrix C2(C);
      blas::her2k(alpha, blas::conj(A), B, beta, blas::upper(C2));
      std::cout << "testing boost::ublas containers\n"
    		<< "using ublas (A left htrans, C upper):\n" << print_mat(C1) << '\n'
    		<< "using blas (A left htrans, C upper):\n" << print_mat(C2) << '\n'
    		<< '\n';
    }
    {
      matrix C1(alpha*ublas::prod(ublas::trans(ublas::conj(A)), B)+
		std::conj(alpha)*ublas::prod(ublas::trans(ublas::conj(B)), A)+
		beta*C);
      for (size_type j=0; j<n; ++j)
    	for (size_type i=0; i<j; ++i)
    	  C1(i, j)=C(i, j);
      matrix C2(C);
      blas::her2k(alpha, blas::conj(A), B, beta, blas::lower(C2));
      std::cout << "testing boost::ublas containers\n"
    		<< "using ublas (A left htrans, C lower):\n" << print_mat(C1) << '\n'
    		<< "using blas (A left htrans, C lower):\n" << print_mat(C2) << '\n'
    		<< '\n';
    }
  }
  return EXIT_SUCCESS;
}
