#include <cstdlib>
#include <iostream>
#include <complex>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/bindings/ublas/vector.hpp>
#include <boost/numeric/bindings/ublas/matrix.hpp>
#include <boost/numeric/bindings/ublas/triangular.hpp>
#include <boost/numeric/bindings/ublas/matrix_proxy.hpp>
#include <boost/numeric/bindings/blas/level1.hpp>
#include <boost/numeric/bindings/blas/level2.hpp>
#include <boost/numeric/bindings/conj.hpp>
#include <boost/numeric/bindings/lapack/driver.hpp>
#include "random.hpp"

namespace ublas=boost::numeric::ublas;
namespace blas=boost::numeric::bindings::blas;
namespace lapack=boost::numeric::bindings::lapack;

int main(int argc, char *argv[]) {
  typedef std::complex<double> complex;
  typedef ublas::vector<complex> vector;
  typedef ublas::vector<double> d_vector;
  typedef ublas::matrix<complex, ublas::column_major> matrix;
  typedef ublas::triangular_matrix<complex, ublas::upper, ublas::column_major> triangular_matrix;
  typedef typename vector::size_type size_type;

  rand_normal<complex>::reset();
  int n=128;
  triangular_matrix A(n, n);
  for (size_type j=0; j<n; ++j) {
    for (size_type i=0; i<j; ++i)
      A(i, j)=rand_normal<complex>::get();
    A(j, j)=rand_normal<complex>::get().real();
  }
  {
    d_vector lambda(n);
    matrix vr(n, n);
    triangular_matrix A_bak(A);
    int info=lapack::hpev('V', A, lambda, vr);
    if (info==0) {
      for (int i=0; i<n; ++i) {
  	// res <- A*vr(i) - lambda(i)*vr(i)
  	ublas::matrix_column<matrix> v(vr, i);
  	vector res(v);
   	blas::hpmv(complex(1), A_bak, v, -lambda(i), res);
  	std::cout << "norm of residual (right eigen vector " << i
  		  << " ): " << blas::nrm2(res) << '\n';
      }
    } else
      if (info>0)
  	std::cout << "unable to compute all eigen values\n";
      else 
  	std::cout << "illegal arguments\n";
  }
  return EXIT_SUCCESS;
}

