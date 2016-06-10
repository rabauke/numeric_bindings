#include <cstdlib>
#include <complex>
#include <random>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <Eigen/Core>
#include <boost/numeric/bindings/ublas/vector.hpp>
#include <boost/numeric/bindings/ublas/matrix_proxy.hpp>
#include <boost/numeric/bindings/ublas/matrix_expression.hpp>
#include <boost/numeric/bindings/eigen/vector.hpp>
#include <boost/numeric/bindings/blas/level1.hpp>

namespace ublas=boost::numeric::ublas;
namespace blas=boost::numeric::bindings::blas;

typedef double real;
typedef std::complex<real> complex;

int main(int argc, char *argv[]) {
  {
    typedef ublas::vector<complex> vector;
    typedef vector::size_type size_type;
    std::mt19937_64 R;
    std::normal_distribution<> N(0, 1);
    size_type dim=1024*1024;
    vector v1(dim), v2(dim);
    for (complex &x: v1) {
      real re(N(R)), im(N(R));
      x=complex(re, im);
    }
    for (complex &x: v2) {
      real re(N(R)), im(N(R));
      x=complex(re, im);
    }
    complex res=blas::dotu(v1, v2);
    std::cout << res << '\n';

    typedef ublas::matrix<complex, ublas::column_major> matrix;
    int n=4;
    matrix M1(n, n), M2(n, n);
    for (size_type j=0; j<n; ++j)
      for (size_type i=0; i<n; ++i) {
	real re(N(R)), im(N(R));
	M1(i, j)=complex(i+1, j+2);
      }
    for (size_type j=0; j<n; ++j) {
      ublas::matrix_column<matrix> mc(M1, j);
      for (size_type i=0; i<n; ++i) {
	ublas::matrix_row<matrix> mr(M1, i);
	M2(i, j)=blas::doth(mr, mc);
      }
    }
    std::cout << M1 << '\n';
    std::cout << M2 << '\n';
  }

  {
    typedef Eigen::Matrix<complex, Eigen::Dynamic, 1> vector;
    typedef int size_type;
    std::mt19937_64 R;
    std::normal_distribution<> N(0, 1);
    size_type dim=1024*1024;
    vector v1(dim), v2(dim);
    for (size_type i=0; i<dim; ++i) {
      real re(N(R)), im(N(R));
      v1(i)=complex(re, im);
    }
    for (size_type i=0; i<dim; ++i) {
      real re(N(R)), im(N(R));
      v2(i)=complex(re, im);
    }
    complex res=blas::dot(v1, v2);
    std::cout << res << '\n';

    typedef Eigen::Matrix<complex, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> matrix;
    int n=4;
    matrix M1(n, n), M2(n, n);
    for (size_type j=0; j<n; ++j)
      for (size_type i=0; i<n; ++i) {
	real re(N(R)), im(N(R));
	M1(i, j)=complex(i+1, j+2);
      }
    for (size_type j=0; j<n; ++j) {
      auto mc(M1.col(j));
      for (size_type i=0; i<n; ++i) {
    	auto mr(M1.row(i));
 	M2(i, j)=blas::doth(mr, mc);
      }
    }
    std::cout << M1 << '\n';
    std::cout << M2 << '\n';
  }
  return EXIT_SUCCESS;
}
