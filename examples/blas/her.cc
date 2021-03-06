#include <cstdlib>
#include <iostream>
#include <complex>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/bindings/lower.hpp>
#include <boost/numeric/bindings/upper.hpp>
#include <boost/numeric/bindings/ublas/vector.hpp>
#include <boost/numeric/bindings/ublas/matrix.hpp>
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
    size_type n=8;
    matrix A(n, n);
    for (size_type j=0; j<n; ++j) {
      A(j, j)=rand_normal<complex>::get().real();
      for (size_type i=0; i<j; ++i) {
	A(i, j)=rand_normal<complex>::get();
	A(j, i)=std::conj(A(i, j));
      }
    }
    vector x(n);
    for (size_type i=0; i<n; ++i)
      x(i)=rand_normal<complex>::get();
    double alpha(rand_normal<complex>::get().real());
    matrix P;
    {
      P=alpha*ublas::outer_prod(x, ublas::conj(x));
      for (size_type j=0; j<n; ++j) 
	for (size_type i=0; i<j; ++i) 
	  P(i, j)=0;
      matrix A1(P+A);
      matrix A2(A);
      blas::her(alpha, x, blas::lower(A2));
      std::cout << print_mat(A1) << '\n'
		<< print_mat(A2) << '\n';
    }
    {
      P=alpha*ublas::outer_prod(x, ublas::conj(x));
      for (size_type j=0; j<n; ++j) 
	for (size_type i=j+1; i<n; ++i) 
	  P(i, j)=0;
      matrix A1(P+A);
      matrix A2(A);
      blas::her(alpha, x, blas::upper(A2));
      std::cout << print_mat(A1) << '\n'
		<< print_mat(A2) << '\n';
    }
  }
  return EXIT_SUCCESS;
}
