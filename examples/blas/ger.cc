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
    size_type m=6, n=8;
    matrix A(m, n);
    for (size_type j=0; j<n; ++j)
      for (size_type i=0; i<m; ++i) 
 	A(i, j)=rand_normal<double>::get();
    vector x(m);
    for (size_type i=0; i<m; ++i)
      x(i)=rand_normal<double>::get();
    vector y(n);
    for (size_type i=0; i<n; ++i)
      y(i)=rand_normal<double>::get();
    double alpha(rand_normal<double>::get());
    matrix A1(alpha*ublas::outer_prod(x, y)+A);
    matrix A2(A);
    blas::ger(alpha, x, y, A2);
    std::cout << print_mat(A1) << '\n'
	      << print_mat(A2) << '\n';
  }
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
    vector x(m);
    for (size_type i=0; i<m; ++i)
      x(i)=rand_normal<complex>::get();
    vector y(n);
    for (size_type i=0; i<n; ++i)
      y(i)=rand_normal<complex>::get();
    complex alpha(rand_normal<complex>::get());
    matrix A1(alpha*ublas::outer_prod(x, y)+A);
    matrix A2(A);
    blas::ger(alpha, x, y, A2);
    std::cout << print_mat(A1) << '\n'
  	      << print_mat(A2) << '\n';
  }
  return EXIT_SUCCESS;
}
