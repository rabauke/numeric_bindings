#include <cstdlib>
#include <iostream>
#include <complex>
#include <type_traits>
#include <algorithm>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/banded.hpp>
#include <boost/numeric/ublas/hermitian.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/bindings/ublas/vector.hpp>
#include <boost/numeric/bindings/ublas/matrix.hpp>
#include <boost/numeric/bindings/ublas/banded.hpp>
#include <boost/numeric/bindings/ublas/hermitian.hpp>
#include <boost/numeric/bindings/ublas/matrix_proxy.hpp>
#include <boost/numeric/bindings/blas/level1.hpp>
#include <boost/numeric/bindings/blas/level2.hpp>
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
  typedef ublas::banded_matrix<complex, ublas::column_major> banded_matrix;
  typedef typename std::make_signed<vector::size_type>::type size_type;

  rand_normal<complex>::reset();
  size_type n=128, k=8;
  banded_matrix A(n, n, 0, k);
  for (size_type j=0; j<n; ++j) {
    for (size_type i=std::max(j-k, size_type(0)); i<j; ++i)
      A(i, j)=rand_normal<complex>::get();
    A(j, j)=rand_normal<complex>::get().real();
  }
  {
    d_vector lambda(n);
    banded_matrix A_bak(A);
    matrix vr(n ,n);
    ublas::hermitian_adaptor<banded_matrix, ublas::upper> B(A);
    int info=lapack::hbev('V', B, lambda, vr);
    ublas::hermitian_adaptor<banded_matrix, ublas::upper> B_bak(A_bak);
    if (info==0) {
      for (int i=0; i<n; ++i) {
  	// res <- A*vr(i) - lambda(i)*vr(i)
  	ublas::matrix_column<matrix> v(vr, i);
  	vector res(ublas::prod(B_bak, v)-lambda(i)*v);
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
