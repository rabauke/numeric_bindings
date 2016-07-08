#include <cstdlib>
#include <iostream>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/bindings/ublas/vector.hpp>
#include <boost/numeric/bindings/ublas/matrix.hpp>
#include <boost/numeric/bindings/ublas/matrix_proxy.hpp>
#include <boost/numeric/bindings/blas/level1.hpp>
#include <boost/numeric/bindings/blas/level2.hpp>
#include <boost/numeric/bindings/upper.hpp>
#include <boost/numeric/bindings/lapack/driver.hpp>
#include "random.hpp"

namespace ublas=boost::numeric::ublas;
namespace blas=boost::numeric::bindings::blas;
namespace lapack=boost::numeric::bindings::lapack;

int main(int argc, char *argv[]) {
  typedef ublas::vector<double> vector;
  typedef ublas::matrix<double, ublas::column_major> matrix;
  typedef typename vector::size_type size_type;

  rand_normal<double>::reset();
  size_type n=128;
  matrix A(n, n);
  for (size_type j=0; j<n; ++j) {
    for (size_type i=0; i<=j; ++i) {
      A(i, j)=rand_normal<double>::get();
      A(j, i)=A(i, j);
    }
  }
  {
    vector lambda(n);
    matrix A_bak(A);
    int info=lapack::heev('V', lapack::upper(A), lambda);
    if (info==0) {
      for (int i=0; i<n; ++i) {
	// res <- A*vr(i) - lambda(i)*vr(i)
	ublas::matrix_column<matrix> v(A, i);
	vector res(v);
	blas::gemv(1., A_bak, v, -lambda(i), res);
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
