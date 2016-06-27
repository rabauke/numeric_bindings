#include <cstdlib>
#include <iostream>
#include <complex>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <Eigen/Core>
#include <boost/numeric/bindings/ublas/vector.hpp>
#include <boost/numeric/bindings/ublas/matrix.hpp>
#include <boost/numeric/bindings/eigen/vector.hpp>
#include <boost/numeric/bindings/eigen/matrix.hpp>
#include <boost/numeric/bindings/blas/level3.hpp>
#include <boost/numeric/bindings/trans.hpp>
#include <boost/numeric/bindings/conj.hpp>
#include "print.hpp"
#include "random.hpp"

namespace ublas=boost::numeric::ublas;
namespace blas=boost::numeric::bindings::blas;

int main(int argc, char *argv[]) {
  {
    typedef std::complex<double> complex;
    typedef ublas::vector<complex> vector;
    typedef ublas::matrix<complex, ublas::column_major> matrix;
    typedef vector::size_type size_type;
    rand_normal<complex>::reset();
    size_type m=6, k=7, n=8;
    matrix A(m, k);
    matrix B(k, n);
    matrix C(m, n);
    for (size_type j=0; j<k; ++j)
      for (size_type i=0; i<m; ++i) 
 	A(i, j)=rand_normal<complex>::get();
    for (size_type j=0; j<n; ++j)
      for (size_type i=0; i<k; ++i) 
 	B(i, j)=rand_normal<complex>::get();
    for (size_type j=0; j<n; ++j)
      for (size_type i=0; i<m; ++i) 
 	C(i, j)=rand_normal<complex>::get();
    matrix A_t(ublas::trans(A));
    matrix B_t(ublas::trans(B));
    matrix A_h(ublas::conj(A_t));
    matrix B_h(ublas::conj(B_t));
    complex alpha(rand_normal<complex>::get());
    complex beta(rand_normal<complex>::get());
    matrix C1(alpha*ublas::prod(A, B)+beta*C);
    matrix C2(C);
    blas::gemm(alpha, A, B, beta, C2);
    matrix C3(C);
    blas::gemm(alpha, blas::trans(A_t), blas::trans(B_t), beta, C3);
    matrix C4(C);
    blas::gemm(alpha, blas::conj(A_h), blas::conj(B_h), beta, C4);
    std::cout << "testing boost::ublas containers\n"
     	      << "using ublas:\n" << print_mat(C1) << '\n'
     	      << "using blas:\n" << print_mat(C2) << '\n'
     	      << "using blas (transposed):\n" << print_mat(C3) << '\n'
     	      << "using blas (hermitian transposed):\n" << print_mat(C4) << '\n'
	      << '\n';
  }
  {
    typedef std::complex<double> complex;
    typedef Eigen::Matrix<complex, Eigen::Dynamic, 1> vector;
    typedef Eigen::Matrix<complex, Eigen::Dynamic, Eigen::Dynamic> matrix;
    typedef int size_type;
    rand_normal<complex>::reset();
    size_type m=6, k=7, n=8;
    matrix A(m, k);
    matrix B(k, n);
    matrix C(m, n);
    for (size_type j=0; j<k; ++j)
      for (size_type i=0; i<m; ++i) 
 	A(i, j)=rand_normal<complex>::get();
    for (size_type j=0; j<n; ++j)
      for (size_type i=0; i<k; ++i) 
 	B(i, j)=rand_normal<complex>::get();
    for (size_type j=0; j<n; ++j)
      for (size_type i=0; i<m; ++i) 
 	C(i, j)=rand_normal<complex>::get();
    matrix A_t(A.transpose());
    matrix B_t(B.transpose());
    matrix A_h(A.adjoint());
    matrix B_h(B.adjoint());
    complex alpha(rand_normal<complex>::get());
    complex beta(rand_normal<complex>::get());
    matrix C1(alpha*A*B+beta*C);
    matrix C2(C);
    blas::gemm(alpha, A, B, beta, C2);
    matrix C3(C);
    blas::gemm(alpha, blas::trans(A_t), blas::trans(B_t), beta, C3);
    matrix C4(C);
    blas::gemm(alpha, blas::conj(A_h), blas::conj(B_h), beta, C4);
    std::cout << "testing Eigen containers\n"
     	      << "using Eigen:\n" << print_mat(C1) << '\n'
     	      << "using blas:\n" << print_mat(C2) << '\n'
     	      << "using blas (transposed):\n" << print_mat(C3) << '\n'
     	      << "using blas (hermitian transposed):\n" << print_mat(C4) << '\n'
    	      << '\n';
  }
  return EXIT_SUCCESS;
}
