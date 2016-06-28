#include <cstdlib>
#include <iostream>
#include <complex>
#define BOOST_UBLAS_NO_ELEMENT_PROXIES
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/bindings/ublas/matrix.hpp>
#include <boost/numeric/bindings/blas/level3.hpp>
#include <boost/numeric/bindings/lower.hpp>
#include <boost/numeric/bindings/upper.hpp>
#include <boost/numeric/bindings/unit_lower.hpp>
#include <boost/numeric/bindings/unit_upper.hpp>
#include <boost/numeric/bindings/left.hpp>
#include <boost/numeric/bindings/right.hpp>
#include <boost/numeric/bindings/trans.hpp>
#include <boost/numeric/bindings/conj.hpp>
#include "print.hpp"
#include "random.hpp"

namespace ublas=boost::numeric::ublas;
namespace blas=boost::numeric::bindings::blas;

int main(int argc, char *argv[]) {
  {
    typedef std::complex<double> complex;
    typedef ublas::matrix<complex, ublas::column_major> matrix;
    typedef typename matrix::size_type size_type;
    rand_normal<complex>::reset();
    size_type n=8, m=2;
    matrix A_l(n, n);
    matrix A_u(n, n);
    for (size_type j=0; j<n; ++j) {
      for (size_type i=0; i<j; ++i) {
    	A_u(i, j)=rand_normal<complex>::get();
	A_l(j, i)=rand_normal<complex>::get();
      }
      A_u(j, j)=rand_normal<complex>::get();
      A_l(j, j)=rand_normal<complex>::get();
      for (size_type i=j+1; i<n; ++i) {
    	A_u(i, j)=0;
	A_l(j, i)=0;
      }
    }
    matrix B(n, m);
    for (size_type j=0; j<m; ++j)
      for (size_type i=0; i<n; ++i)
	B(i, j)=rand_normal<complex>::get();
    complex alpha=rand_normal<complex>::get();
    {
      matrix B1(B);
      for (size_type j=0; j<m; ++j) {
	ublas::matrix_column<matrix> b(B1, j);
	ublas::inplace_solve(A_l, b, ublas::lower_tag());
	b*=alpha;
      }
      matrix B2(B);
      blas::trsm(blas::left(), alpha, blas::lower(A_l), B2);
      std::cout << "testing boost::ublas containers\n"
    		<< "using ublas (left, lower):\n" << print_mat(B1) << '\n'
    		<< "using blas  (left, lower):\n" << print_mat(B2) << '\n'
     		<< '\n';
    }
    {
      matrix B1(B);
      for (size_type j=0; j<m; ++j) {
	ublas::matrix_column<matrix> b(B1, j);
	ublas::inplace_solve(A_u, b, ublas::upper_tag());
	b*=alpha;
      }
      matrix B2(B);
      blas::trsm(blas::left(), alpha, blas::upper(A_u), B2);
      std::cout << "testing boost::ublas containers\n"
    		<< "using ublas (left, upper):\n" << print_mat(B1) << '\n'
    		<< "using blas  (left, upper):\n" << print_mat(B2) << '\n'
     		<< '\n';
    }
  }
  return EXIT_SUCCESS;
}
