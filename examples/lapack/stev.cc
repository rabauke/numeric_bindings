#include <cstdlib>
#include <iostream>
#include <type_traits>
#include <algorithm>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/banded.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/bindings/ublas/vector.hpp>
#include <boost/numeric/bindings/ublas/matrix.hpp>
#include <boost/numeric/bindings/ublas/banded.hpp>
#include <boost/numeric/bindings/ublas/matrix_proxy.hpp>
#include <boost/numeric/bindings/blas/level1.hpp>
#include <boost/numeric/bindings/blas/level2.hpp>
#include <boost/numeric/bindings/lapack/driver.hpp>
#include "random.hpp"

namespace ublas=boost::numeric::ublas;
namespace blas=boost::numeric::bindings::blas;
namespace lapack=boost::numeric::bindings::lapack;

int main(int argc, char *argv[]) {
  typedef ublas::vector<double> vector;
  typedef ublas::matrix<double, ublas::column_major> matrix;
  typedef ublas::banded_matrix<double, ublas::column_major> banded_matrix;
  typedef typename std::make_signed<vector::size_type>::type size_type;

  rand_normal<double>::reset();
  size_type n=128, k=1;
  banded_matrix A(n, n, k, k);
  for (size_type j=0; j<n; ++j) {
    for (size_type i=std::max(j-k, size_type(0)); i<=j; ++i) {
      A(i, j)=rand_normal<double>::get();
      A(j, i)=A(i, j);
    }
  }
  {
    vector d(n), e(n-1);
    for (size_type j=0; j<n; ++j) {
      d(j)=A(j, j);
      if (j<n-1)
	e(j)=A(j+1, j);
    }
    matrix vr(n ,n);
    int info=lapack::stev('V', n, d, e, vr);
    if (info==0) {
      for (int i=0; i<n; ++i) {
    	// res <- A*vr(i) - lambda(i)*vr(i)
	ublas::matrix_column<matrix> v(vr, i);
	vector res(ublas::prod(A, v)-d(i)*v);
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
