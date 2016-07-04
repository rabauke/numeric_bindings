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
#include "print.hpp"

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
  // generate a random unitary matrix
  matrix U(n, n);
  for (size_type j=0; j<n; ++j)
    for (size_type i=0; i<n; ++i)
      U(i, j)=i==j ? complex(1) : complex(0);
  for (int k=0; k<n*n; ++k) {
    // generate a random 2x2 unitary matrix
    double phi(rand_uniform<double>::get(0, 1.5707963267948966192));
    double alpha(rand_uniform<double>::get(0, 6.2831853071795864770));
    double psi(rand_uniform<double>::get(0, 6.2831853071795864770));
    double chi(rand_uniform<double>::get(0, 6.2831853071795864770));
    matrix u(2, 2);
    u(0, 0)=complex(std::cos(alpha+psi), std::sin(alpha+psi))*std::cos(phi);
    u(1, 0)=-complex(std::cos(alpha-chi), std::sin(alpha-chi))*std::sin(phi);
    u(0, 1)=complex(std::cos(alpha+chi), std::sin(alpha+chi))*std::sin(phi);
    u(1, 1)=complex(std::cos(alpha-psi), std::sin(alpha-psi))*std::cos(phi);
    int j0, j1;
    j0=static_cast<int>(rand_uniform<double>::get(0, n));
    do {
      j1=static_cast<int>(rand_uniform<double>::get(0, n));
    } while (j0==j1);
    std::cout << j0 << '\t' << j1 << '\n';
    for (size_type i=0; i<n; ++i) {
      vector Uc(2);
      Uc(0)=U(j0, i);
      Uc(1)=U(j1, i);
      Uc=ublas::prod(u, Uc);
      U(j0, i)=Uc(0);
      U(j1, i)=Uc(1);
    }
  }
  matrix R(ublas::prod(ublas::trans(ublas::conj(U)), U));
  // generate random positive definite hermitian matrix
  matrix A(n, n);
  for (size_type j=0; j<n; ++j)
    for (size_type i=0; i<n; ++i)
      if (i==j)
	// eigenvalues drawn from the positive half of the normal distribution
	do {
	  A(i, j)=std::abs(rand_normal<complex>::get().real());
	} while (A(i, j)==complex(0));
      else
	A(i, j)=complex(0);
  // apply a random unitary transform to A
  A=ublas::prod(ublas::trans(ublas::conj(U)), A);
  A=ublas::prod(A, U);
  matrix A_bak(A);
  vector b(n);
  for (size_type i=0; i<n; ++i)
    b(i)=rand_normal<complex>::get();
  vector x(b);
  int info=lapack::posv(lapack::upper(A), x); // solve
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

