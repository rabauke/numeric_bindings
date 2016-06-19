#include <cstdlib>
#include <iostream>
#include <complex>
#define BOOST_UBLAS_NO_ELEMENT_PROXIES
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/hermitian.hpp>
#include <boost/numeric/bindings/ublas/vector.hpp>
#include <boost/numeric/bindings/ublas/hermitian.hpp>
#include <boost/numeric/bindings/blas/level2.hpp>
#include "print.hpp"
#include "random.hpp"

namespace ublas=boost::numeric::ublas;
namespace blas=boost::numeric::bindings::blas;

int main(int argc, char *argv[]) {
  {
    typedef std::complex<double> complex;
    typedef ublas::vector<complex> vector;
    typedef ublas::hermitian_matrix<complex, ublas::lower, ublas::column_major> matrix_l;
    typedef ublas::hermitian_matrix<complex, ublas::upper, ublas::column_major> matrix_u;
    typedef typename vector::size_type size_type;
    rand_normal<complex>::reset();
    size_type n=8;
    matrix_l A_l(n);
    matrix_u A_u(n);
    for (size_type j=0; j<n; ++j) {
      A_u(j, j)=rand_normal<complex>::get().real();
      A_l(j, j)=A_u(j, j);
      for (size_type i=0; i<j; ++i) {
	complex a();
    	A_u(i, j)=rand_normal<complex>::get();
	A_l(j, i)=std::conj(A_u(i, j));
       }
    }
    vector x(n);
    for (size_type i=0; i<n; ++i)
      x(i)=rand_normal<complex>::get();
    vector y(n);
    for (size_type i=0; i<n; ++i)
      y(i)=rand_normal<complex>::get();
    complex alpha(rand_normal<complex>::get());
    complex beta(rand_normal<complex>::get());
    vector y1(alpha*ublas::prod(A_l, x)+beta*y);
    vector y2(y);
    blas::hpmv(alpha, A_l, x, beta, y2);
    vector y3(y);
    blas::hpmv(alpha, A_u, x, beta, y3);
    std::cout << "testing boost::ublas containers\n"
    	      << "using ublas       : " << print_vec(y1) << '\n'
    	      << "using blas (lower): " << print_vec(y2) << '\n'
    	      << "using blas (upper): " << print_vec(y3) << '\n'
    	      << '\n';
  }
  return EXIT_SUCCESS;
}
