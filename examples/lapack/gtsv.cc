#include <cstdlib>
#include <iostream>
#include <complex>
#include <type_traits>
#include <algorithm>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/banded.hpp>
#include <boost/numeric/bindings/ublas/vector.hpp>
#include <boost/numeric/bindings/ublas/banded.hpp>
#include <boost/numeric/bindings/blas/level1.hpp>
#include <boost/numeric/bindings/blas/level2.hpp>
#include <boost/numeric/bindings/lapack/driver.hpp>
#include "random.hpp"
#include "print.hpp"

namespace ublas=boost::numeric::ublas;
namespace blas=boost::numeric::bindings::blas;
namespace lapack=boost::numeric::bindings::lapack;

int main(int argc, char *argv[]) {
  typedef std::complex<double> complex;
  typedef ublas::vector<complex> vector;
  typedef ublas::vector<int> p_vector;
  typedef ublas::banded_matrix<complex, ublas::column_major> matrix;
  typedef typename std::make_signed<vector::size_type>::type size_type;

  rand_normal<complex>::reset();
  size_type n=1024;
  matrix A(n, n, 1, 1);
  for (size_type j=0; j<n; ++j)
    for (size_type i=std::max(size_type(0), j-1); i<std::min(n, j+1+1); ++i)
      A(i, j)=rand_normal<complex>::get();
  vector b(n);
  for (size_type i=0; i<n; ++i)
    b(i)=rand_normal<complex>::get();
  vector x(b);
  vector d(n), du(n-1), dl(n-1);
  for (size_type i=0; i<n; ++i)
    d(i)=A(i ,i);
  for (size_type i=0; i<n-1; ++i) {
    du(i)=A(i, i+1);
    dl(i)=A(i+1, i);
  }
  int info=lapack::gtsv(n, dl, d, du, x); // solve
  if (info==0) {
    // res <- A*x - b
    vector res(b);
    blas::gbmv(complex(1, 0), A, x, complex(-1, 0), res);
    std::cout << "norm of residual : " << blas::nrm2(res) << '\n';
  } else
    if (info>0)
      std::cout << "singular matrix\n";
    else 
      std::cout << "illegal arguments\n";
  return EXIT_SUCCESS;
}

