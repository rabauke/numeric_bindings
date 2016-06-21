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
    vector v1(n), v2(n);
    for (size_type i=0; i<n; ++i)
      v1(i)=rand_normal<real>::get();
    for (size_type i=0; i<n; ++i)
      v2(i)=rand_normal<real>::get();
    matrix M1(n, n), M2(n, n);
    for (size_type j=0; j<n; ++j)
      for (size_type i=0; i<n; ++i) {
    	M1(i, j)=0;
    	M2(i, j)=0;
      }
    ublas::matrix_column<matrix> mc(M1, 2);
    ublas::matrix_row<matrix> mr(M2, 3);
    mc=v1;
    mr=v2;
    std::cout << "ublas using vectors    : dot(v1, v2) = " << ublas::inner_prod(v1, v2) << '\n'
	      << "blas using vectors     : dot(v1, v2) = " << blas::dot(v1, v2) << '\n'
	      << "blas using cols & rows : dot(v1, v2) = " << blas::dot(mc, mr) << '\n'
	      << "blas using vectors     : doth(v1, v2) = " << blas::doth(v1, v2) << '\n'
	      << "blas using cols & rows : doth(v1, v2) = " << blas::doth(mc, mr) << '\n';
  }
  {
    typedef ublas::vector<complex> vector;
    typedef ublas::matrix<complex> matrix;
    typedef vector::size_type size_type;
    rand_normal<complex>::reset();
    size_type n=8;
    vector v1(n), v2(n);
    for (size_type i=0; i<n; ++i)
      v1(i)=rand_normal<complex>::get();
    for (size_type i=0; i<n; ++i)
      v2(i)=rand_normal<complex>::get();
    matrix M1(n, n), M2(n, n);
    for (size_type j=0; j<n; ++j)
      for (size_type i=0; i<n; ++i) {
    	M1(i, j)=0;
    	M2(i, j)=0;
      }
    ublas::matrix_column<matrix> mc(M1, 2);
    ublas::matrix_row<matrix> mr(M2, 3);
    mc=v1;
    mr=v2;
    std::cout << "ublas using vectors    : doth(v1, v2) = " << ublas::inner_prod(ublas::conj(v1), v2) << '\n'
	      << "blas using vectors     : doth(v1, v2) = " << blas::doth(v1, v2) << '\n'
	      << "blas using cols & rows : doth(v1, v2) = " << blas::doth(mc, mr) << '\n'
	      << "blas using vectors     : dotc(v1, v2) = " << blas::dotc(v1, v2) << '\n'
	      << "blas using cols & rows : dotc(v1, v2) = " << blas::dotc(mc, mr) << '\n';
  }
  {
    typedef Eigen::Matrix<real, Eigen::Dynamic, 1> vector;
    typedef Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic> matrix;
    typedef int size_type;
    rand_normal<real>::reset();
    size_type n=8;
    vector v1(n), v2(n);
    for (size_type i=0; i<n; ++i)
      v1(i)=rand_normal<real>::get();
    for (size_type i=0; i<n; ++i)
      v2(i)=rand_normal<real>::get();
    matrix M1(n, n), M2(n, n);
    for (size_type j=0; j<n; ++j)
      for (size_type i=0; i<n; ++i) {
    	M1(i, j)=0;
    	M2(i, j)=0;
      }
    auto mc=M1.col(2);
    auto mr=M2.row(3);
    mc=v1;
    mr=v2;
    std::cout << "eigen using vectors    : dot(v1, v2) = " << v1.dot(v2) << '\n'
	      << "blas using vectors     : dot(v1, v2) = " << blas::dot(v1, v2) << '\n'
	      << "blas using cols & rows : dot(v1, v2) = " << blas::dot(mc, mr) << '\n'
	      << "blas using vectors     : doth(v1, v2) = " << blas::doth(v1, v2) << '\n'
	      << "blas using cols & rows : doth(v1, v2) = " << blas::doth(mc, mr) << '\n';
  }
  {
    typedef Eigen::Matrix<complex, Eigen::Dynamic, 1> vector;
    typedef Eigen::Matrix<complex, Eigen::Dynamic, Eigen::Dynamic> matrix;
    typedef int size_type;
    rand_normal<complex>::reset();
    size_type n=8;
    vector v1(n), v2(n);
    for (size_type i=0; i<n; ++i)
      v1(i)=rand_normal<complex>::get();
    for (size_type i=0; i<n; ++i)
      v2(i)=rand_normal<complex>::get();
    matrix M1(n, n), M2(n, n);
    for (size_type j=0; j<n; ++j)
      for (size_type i=0; i<n; ++i) {
    	M1(i, j)=0;
    	M2(i, j)=0;
      }
    auto mc=M1.col(2);
    auto mr=M2.row(3);
    mc=v1;
    mr=v2;
    std::cout << "eigen using vectors    : dot(v1, v2) = " << v1.dot(v2) << '\n'
	      << "blas using vectors     : doth(v1, v2) = " << blas::doth(v1, v2) << '\n'
	      << "blas using cols & rows : doth(v1, v2) = " << blas::doth(mc, mr) << '\n'
	      << "blas using vectors     : dotc(v1, v2) = " << blas::dotc(v1, v2) << '\n'
	      << "blas using cols & rows : dotc(v1, v2) = " << blas::dotc(mc, mr) << '\n';
  }
  return EXIT_SUCCESS;
}
