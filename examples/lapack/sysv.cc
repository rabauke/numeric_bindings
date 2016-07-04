#include <cstdlib>
#include <iostream>
#include <complex>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/bindings/ublas/vector.hpp>
#include <boost/numeric/bindings/ublas/matrix.hpp>
#include <boost/numeric/bindings/blas/level1.hpp>
#include <boost/numeric/bindings/blas/level2.hpp>
#include <boost/numeric/bindings/lapack/driver.hpp>
#include <boost/numeric/bindings/lower.hpp>
#include <boost/numeric/bindings/upper.hpp>
#include "random.hpp"

namespace ublas=boost::numeric::ublas;
namespace blas=boost::numeric::bindings::blas;
namespace lapack=boost::numeric::bindings::lapack;

int main(int argc, char *argv[]) {
  typedef std::complex<double> complex;
  typedef ublas::vector<complex> vector;
  typedef ublas::vector<int> p_vector;
  typedef ublas::matrix<complex, ublas::column_major> matrix;
  typedef typename vector::size_type size_type;

  rand_normal<complex>::reset();
  int n=128;
  matrix A(n, n);
  for (size_type j=0; j<n; ++j)
    for (size_type i=0; i<=j; ++i) {
      A(i, j)=rand_normal<complex>::get();
      A(j, i)=A(i, j);
    }
  vector b(n);
  for (size_type i=0; i<n; ++i)
    b(i)=rand_normal<complex>::get();
  matrix A_bak(A);
  vector x(b);
  p_vector p(n);  // pivots
  int info=lapack::sysv(lapack::upper(A), p, x); // solve
  if (info==0) {
    // res <- A*x - b
    vector res(b);
    blas::gemv(complex(1, 0), A_bak, x, complex(-1, 0), res);
    std::cout << "norm of residual : " << blas::nrm2(res) << '\n';
  } else
    if (info>0)
      std::cout << "singular matrix\n";
    else 
      std::cout << "illegal arguments\n";
  return EXIT_SUCCESS;
}

