#include <cstdlib>
#include <iostream>
#include <complex>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <Eigen/Core>
#include <boost/numeric/bindings/ublas/vector.hpp>
#include <boost/numeric/bindings/ublas/matrix.hpp>
#include <boost/numeric/bindings/ublas/matrix_proxy.hpp>
#include <boost/numeric/bindings/eigen/vector.hpp>
#include <boost/numeric/bindings/blas/level1.hpp>
#include "random.hpp"

namespace ublas=boost::numeric::ublas;
namespace blas=boost::numeric::bindings::blas;

typedef double real;
typedef std::complex<real> complex;

int main(int argc, char *argv[]) {
  {
    typedef ublas::vector<real> vector;
    typedef ublas::matrix<real> matrix;
    typedef vector::size_type size_type;
    rand_normal<real>::reset();
    size_type n=8;
    vector v(n);
    for (size_type i=0; i<n; ++i)
      v(i)=rand_normal<real>::get();
    std::cout << "ublas using vectors : nrm2(v) = " << ublas::norm_2(v) << '\n'
	      << "blas using vectors  : nrm2(v) = " << blas::nrm2(v) << '\n';
    matrix M(n, n);
    for (size_type j=0; j<n; ++j)
      for (size_type i=0; i<n; ++i)
    	M(i, j)=0;
    ublas::matrix_column<matrix> mc(M, 2);
    ublas::matrix_row<matrix> mr(M, 3);
    mc=v;
    std::cout << "blas using cols     : nrm2(v) = " << blas::nrm2(mc) << '\n';
    mr=v;
    std::cout << "blas using rows     : nrm2(v) = " << blas::nrm2(mr) << '\n';
  }
  {
    typedef ublas::vector<complex> vector;
    typedef ublas::matrix<complex> matrix;
    typedef vector::size_type size_type;
    rand_normal<complex>::reset();
    size_type n=8;
    vector v(n);
    for (size_type i=0; i<n; ++i)
      v(i)=rand_normal<complex>::get();
    std::cout << "ublas using vectors : nrm2(v) = " << ublas::norm_2(v) << '\n'
	      << "blas using vectors  : nrm2(v) = " << blas::nrm2(v) << '\n';
    matrix M(n, n);
    for (size_type j=0; j<n; ++j)
      for (size_type i=0; i<n; ++i)
    	M(i, j)=0;
    ublas::matrix_column<matrix> mc(M, 2);
    ublas::matrix_row<matrix> mr(M, 3);
    mc=v;
    std::cout << "blas using cols     : nrm2(v) = " << blas::nrm2(mc) << '\n';
    mr=v;
    std::cout << "blas using rows     : nrm2(v) = " << blas::nrm2(mr) << '\n';
  }
  {
    typedef Eigen::Matrix<real, Eigen::Dynamic, 1> vector;
    typedef Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic> matrix;
    typedef int size_type;
    rand_normal<real>::reset();
    size_type n=8;
    vector v(n);
    for (size_type i=0; i<n; ++i)
      v(i)=rand_normal<real>::get();
    std::cout << "eigen using vectors : nrm2(v) = " << v.norm() << '\n'
	      << "blas using vectors  : nrm2(v) = " << blas::nrm2(v) << '\n';
    matrix M(n, n);
    for (size_type j=0; j<n; ++j)
      for (size_type i=0; i<n; ++i)
    	M(i, j)=0;
    auto mc=M.col(2);
    auto mr=M.row(3);
    mc=v;
    std::cout << "blas using cols     : nrm2(v) = " << blas::nrm2(mc) << '\n';
    mr=v;
    std::cout << "blas using rows     : nrm2(v) = " << blas::nrm2(mr) << '\n';
  }
  {
    typedef Eigen::Matrix<complex, Eigen::Dynamic, 1> vector;
    typedef Eigen::Matrix<complex, Eigen::Dynamic, Eigen::Dynamic> matrix;
    typedef int size_type;
    rand_normal<complex>::reset();
    size_type n=8;
    vector v(n);
    for (size_type i=0; i<n; ++i)
      v(i)=rand_normal<complex>::get();
    std::cout << "eigen using vectors : nrm2(v) = " << v.norm() << '\n'
	      << "blas using vectors  : nrm2(v) = " << blas::nrm2(v) << '\n';
    matrix M(n, n);
    for (size_type j=0; j<n; ++j)
      for (size_type i=0; i<n; ++i)
    	M(i, j)=0;
    auto mc=M.col(2);
    auto mr=M.row(3);
    mc=v;
    std::cout << "blas using cols     : nrm2(v) = " << blas::nrm2(mc) << '\n';
    mr=v;
    std::cout << "blas using rows     : nrm2(v) = " << blas::nrm2(mr) << '\n';
  }
  return EXIT_SUCCESS;
}
