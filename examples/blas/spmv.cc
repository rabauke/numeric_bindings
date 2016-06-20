#include <cstdlib>
#include <iostream>
#include <complex>
#define BOOST_UBLAS_NO_ELEMENT_PROXIES
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/symmetric.hpp>
#include <boost/numeric/bindings/ublas/vector.hpp>
#include <boost/numeric/bindings/ublas/symmetric.hpp>
#include <boost/numeric/bindings/blas/level2.hpp>
#include "print.hpp"
#include "random.hpp"

namespace ublas=boost::numeric::ublas;
namespace blas=boost::numeric::bindings::blas;

int main(int argc, char *argv[]) {
  {
    typedef ublas::vector<double> vector;
    typedef ublas::symmetric_matrix<double, ublas::lower, ublas::column_major> matrix_l;
    typedef ublas::symmetric_matrix<double, ublas::upper, ublas::column_major> matrix_u;
    typedef typename vector::size_type size_type;
    rand_normal<double>::reset();
    size_type n=8;
    matrix_l A_l(n);
    matrix_u A_u(n);
    for (size_type j=0; j<n; ++j) {
      A_u(j, j)=rand_normal<double>::get();
      A_l(j, j)=A_u(j, j);
      for (size_type i=0; i<j; ++i) {
	double a();
    	A_u(i, j)=rand_normal<double>::get();
	A_l(j, i)=A_u(i, j);
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
    vector y1(alpha*ublas::prod(A_l, x)+beta*y);
    vector y2(y);
    blas::spmv(alpha, A_l, x, beta, y2);
    vector y3(y);
    blas::spmv(alpha, A_u, x, beta, y3);
    vector y4(y);
    blas::spmv(alpha, A_l, x, beta, y4);
    vector y5(y);
    blas::spmv(alpha, A_u, x, beta, y5);
    std::cout << "testing boost::ublas containers\n"
    	      << "using ublas            : " << print_vec(y1) << '\n'
    	      << "using blas spmv (lower): " << print_vec(y2) << '\n'
    	      << "using blas spmv (upper): " << print_vec(y3) << '\n'
    	      << "using blas hpmv (lower): " << print_vec(y4) << '\n'
    	      << "using blas hpmv (upper): " << print_vec(y5) << '\n'
    	      << '\n';
  }
  return EXIT_SUCCESS;
}
