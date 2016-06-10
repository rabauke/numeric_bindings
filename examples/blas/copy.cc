#include <cstdlib>
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


int main(int argc, char *argv[]) {
  {
    typedef ublas::vector<double> vector;
    typedef ublas::matrix<double> matrix;
    typedef vector::size_type size_type;
    size_type n=8;
    vector v1(n), v2(n);
    for (auto &x: v1) 
      x=1;
    std::cout << v1 << '\n';
    for (auto &x: v2)
      x=0;
    blas::copy(v1, v2);
    std::cout << v2 << '\n';
    matrix M(n, n);
    for (size_type j=0; j<n; ++j)
      for (size_type i=0; i<n; ++i) 
     	M(i, j)=0;
    ublas::matrix_column<matrix> mc(M, 2);
    ublas::matrix_row<matrix> mr(M, 3);
    blas::copy(v1, mc);
    blas::copy(v1, mr);
    std::cout << M << '\n';
  }

  {
    typedef Eigen::Matrix<double, Eigen::Dynamic, 1> vector;
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> matrix;
    typedef int size_type;
    size_type n=8;
    vector v1(n), v2(n);
    for (size_type i=0; i<n; ++i)
      v1(i)=1;
    std::cout << v1 << '\n';
    for (size_type i=0; i<n; ++i)
      v2(i)=0;
    blas::copy(v1, v2);
    std::cout << v2 << '\n';
    matrix M(n, n);
    for (size_type j=0; j<n; ++j)
      for (size_type i=0; i<n; ++i) 
     	M(i, j)=0;
    auto mc=M.col(2);
    auto mr=M.row(3);
    blas::copy(v1, mc);
    blas::copy(v1, mr);
    std::cout << M << '\n';
  }
  return EXIT_SUCCESS;
}
