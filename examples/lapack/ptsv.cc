#include <cstdlib>
#include <iostream>
#include <complex>
#include <type_traits>
#include <algorithm>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/banded.hpp>
#include <boost/numeric/bindings/ublas/vector.hpp>
#include <boost/numeric/bindings/ublas/banded.hpp>
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
  typedef ublas::banded_matrix<complex, ublas::column_major> matrix;
  typedef ublas::matrix<complex, ublas::column_major> dense_matrix;
  typedef typename std::make_signed<vector::size_type>::type size_type;

  rand_normal<complex>::reset();
  size_type n=1024;
  matrix A(n, n, 1, 1);
  for (size_type i=0; i<n; ++i)
    A(i, i)=std::abs(rand_normal<complex>::get())+1;
  for (size_type i=0; i<n-1; ++i) {
    A(i+1, i)=complex(0);
    A(i, i+1)=complex(0);
  }
  for (int k=0; k<n-1; ++k) {
    // generate a random 2x2 unitary matrix
    double phi(rand_uniform<double>::get(0, 1.5707963267948966192));
    double alpha(rand_uniform<double>::get(0, 6.2831853071795864770));
    double psi(rand_uniform<double>::get(0, 6.2831853071795864770));
    double chi(rand_uniform<double>::get(0, 6.2831853071795864770));
    dense_matrix u(2, 2);
    u(0, 0)=complex(std::cos(alpha+psi), std::sin(alpha+psi))*std::cos(phi);
    u(1, 0)=-complex(std::cos(alpha-chi), std::sin(alpha-chi))*std::sin(phi);
    u(0, 1)=complex(std::cos(alpha+chi), std::sin(alpha+chi))*std::sin(phi);
    u(1, 1)=complex(std::cos(alpha-psi), std::sin(alpha-psi))*std::cos(phi);
    dense_matrix a(2, 2);
    a(0, 0)=A(k, k);
    a(1, 0)=A(k+1, k);
    a(0, 1)=A(k, k+1);
     a(1, 1)=A(k+1, k+1);
    a=ublas::prod(ublas::trans(ublas::conj(u)), a);
    a=ublas::prod(a, u);
    A(k, k)=a(0, 0);
    A(k+1, k)=a(1, 0);
    A(k, k+1)=a(0, 1);
    A(k+1, k+1)=a(1, 1);
  }
  vector b(n);
  for (size_type i=0; i<n; ++i)
    b(i)=rand_normal<complex>::get();
  vector x(b);
  d_vector d(n);
  vector e(n-1);
  for (size_type i=0; i<n; ++i)
    d(i)=A(i ,i).real();
  for (size_type i=0; i<n-1; ++i)
    e(i)=A(i+1, i);
  int info=lapack::ptsv(d, e, x); // solve
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

