#include <cstdlib>
#include <iostream>
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
    typedef ublas::vector<double> vector;
    typedef ublas::matrix<double, ublas::column_major> matrix;
    typedef vector::size_type size_type;
    rand_normal<double>::reset();
    size_type n=8;
    matrix A(n, n);
    for (size_type j=0; j<n; ++j) {
      A(j, j)=rand_normal<double>::get();
      for (size_type i=0; i<j; ++i) {
	A(i, j)=rand_normal<double>::get();
	A(j, i)=A(i, j);
      }
    }
    vector x(n);
    for (size_type i=0; i<n; ++i)
      x(i)=rand_normal<double>::get();
    vector y(n);
    for (size_type i=0; i<n; ++i)
      y(i)=rand_normal<double>::get();
    double alpha(rand_normal<double>::get());
    matrix P;
    {
      P=ublas::outer_prod(alpha*x, y);
      P+=ublas::outer_prod(y, alpha*x);
      for (size_type j=0; j<n; ++j) 
	for (size_type i=0; i<j; ++i) 
	  P(i, j)=0;
      matrix A1(P+A);
      matrix A2(A);
      blas::syr2(alpha, x, y, blas::lower(A2));
      std::cout << print_mat(A1) << '\n'
		<< print_mat(A2) << '\n';
    }
    {
      P=ublas::outer_prod(alpha*x, y);
      P+=ublas::outer_prod(y, alpha*x);
      for (size_type j=0; j<n; ++j) 
	for (size_type i=j+1; i<n; ++i) 
	  P(i, j)=0;
      matrix A1(P+A);
      matrix A2(A);
      blas::syr2(alpha, x, y, blas::upper(A2));
      std::cout << print_mat(A1) << '\n'
		<< print_mat(A2) << '\n';
    }
  }
  return EXIT_SUCCESS;
}
