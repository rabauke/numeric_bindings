#include <cstdlib>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <Eigen/Core>
#include <boost/numeric/bindings/ublas/vector.hpp>
#include <boost/numeric/bindings/ublas/matrix.hpp>
#include <boost/numeric/bindings/ublas/matrix_proxy.hpp>
#include <boost/numeric/bindings/ublas/matrix_expression.hpp>
#include <boost/numeric/bindings/eigen/vector.hpp>
#include <boost/numeric/bindings/eigen/matrix.hpp>
#include <boost/numeric/bindings/blas/level1.hpp>
#include "random.hpp"
#include "print.hpp"

namespace ublas=boost::numeric::ublas;
namespace blas=boost::numeric::bindings::blas;


int main(int argc, char *argv[]) {
  {
    typedef ublas::vector<double> vector;
    typedef ublas::matrix<double> matrix;
    typedef vector::size_type size_type;
    rand_normal<double>::reset();
    size_type n=8;
    vector v1(n);
    for (size_type i=0; i<n; ++i)
      v1(i)=rand_normal<double>::get();
    vector v2(n);
    blas::copy(v1, v2);
    matrix M(n, n);
    for (size_type j=0; j<n; ++j)
      for (size_type i=0; i<n; ++i) 
     	M(i, j)=0;
    ublas::matrix_column<matrix> mc(M, 2);
    ublas::matrix_row<matrix> mr(M, 3);
    blas::copy(v1, mc);
    blas::copy(v1, mr);
    std::cout << "v1 : " << print_vec(v1) << '\n'
	      << "v2 : " << print_vec(v2) << '\n'
	      << "M :\n" 
	      << print_mat(M) << '\n';
  }
  {
    typedef Eigen::Matrix<double, Eigen::Dynamic, 1> vector;
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> matrix;
    typedef int size_type;
    rand_normal<double>::reset();
    size_type n=8;
    vector v1(n);
    for (size_type i=0; i<n; ++i)
      v1(i)=rand_normal<double>::get();
    vector v2(n);
    blas::copy(v1, v2);
    matrix M(n, n);
    for (size_type j=0; j<n; ++j)
      for (size_type i=0; i<n; ++i) 
     	M(i, j)=0;
    auto mc=M.col(2);
    auto mr=M.row(3);
    blas::copy(v1, mc);
    blas::copy(v1, mr);
    std::cout << "v1 : " << print_vec(v1) << '\n'
	      << "v2 : " << print_vec(v2) << '\n'
	      << "M :\n" 
	      << print_mat(M) << '\n';
  }
  return EXIT_SUCCESS;
}
