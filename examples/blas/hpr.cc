#include <cstdlib>
#include <iostream>
#include <complex>
#define BOOST_UBLAS_NO_ELEMENT_PROXIES
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/bindings/ublas/vector.hpp>
#include <boost/numeric/bindings/ublas/matrix.hpp>
#include <boost/numeric/bindings/ublas/triangular.hpp>
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
    typedef ublas::triangular_matrix<complex, ublas::lower, ublas::column_major> matrix_l;
    typedef ublas::triangular_matrix<complex, ublas::upper, ublas::column_major> matrix_u;
    typedef typename vector::size_type size_type;
    rand_normal<complex>::reset();
    size_type n=8;
    matrix_l A_l(n, n);
    matrix_u A_u(n, n);
    for (size_type j=0; j<n; ++j) {
      A_u(j, j)=rand_normal<complex>::get().real();
      A_l(j, j)=A_u(j, j);
      for (size_type i=0; i<j; ++i) {
        A_u(i, j)=rand_normal<complex>::get();
        A_l(j, i)=std::conj(A_u(i, j));
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
      matrix A1(P+A_l);
      matrix_l A2(A_l);
      blas::hpr(alpha, x, A2);
      std::cout << print_mat(A1) << '\n'
                << print_mat(A2) << '\n';
    }
    {
      P=alpha*ublas::outer_prod(x, ublas::conj(x));
      for (size_type j=0; j<n; ++j)
        for (size_type i=j+1; i<n; ++i)
          P(i, j)=0;
      matrix A1(P+A_u);
      matrix_u A2(A_u);
      blas::hpr(alpha, x, A2);
      std::cout << print_mat(A1) << '\n'
                << print_mat(A2) << '\n';
    }
  }
  return EXIT_SUCCESS;
}
