#include <cstdlib>
#include <iostream>
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
    vector x(n), y(n);
    for (size_type i=0; i<n; ++i)
      x(i)=rand_normal<double>::get();
    for (size_type i=0; i<n; ++i)
      y(i)=rand_normal<double>::get();
    double alpha=rand_normal<double>::get();
    vector y1(alpha*x+y);
    vector y2(y);
    blas::axpy(alpha, x, y2);
    std::cout << "y1 : " << print_vec(y1) << '\n'
	      << "y2 : " << print_vec(y2) << '\n';
    matrix M(n, n);
    for (size_type j=0; j<n; ++j)
      for (size_type i=0; i<n; ++i) 
     	M(i, j)=0;
    ublas::matrix_column<matrix> mc(M, 2);
    ublas::matrix_row<matrix> mr(M, 3);
    mc=y;
    blas::axpy(alpha, x, mc);
    std::cout << "M :\n" << print_mat(M) << '\n';
    mr=y;
    blas::axpy(alpha, x, mr);
    std::cout << "M :\n" << print_mat(M) << '\n';
  }
  {
    typedef Eigen::Matrix<double, Eigen::Dynamic, 1> vector;
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> matrix;
    typedef int size_type;
    rand_normal<double>::reset();
    size_type n=8;
    vector x(n), y(n);
    for (size_type i=0; i<n; ++i)
      x(i)=rand_normal<double>::get();
    for (size_type i=0; i<n; ++i)
      y(i)=rand_normal<double>::get();
    double alpha=rand_normal<double>::get();
    vector y1(alpha*x+y);
    vector y2(y);
    blas::axpy(alpha, x, y2);
    std::cout << "y1 : " << print_vec(y1) << '\n'
	      << "y2 : " << print_vec(y2) << '\n';
    matrix M(n, n);
    for (size_type j=0; j<n; ++j)
      for (size_type i=0; i<n; ++i) 
     	M(i, j)=0;
    auto mc=M.col(2);
    auto mr=M.row(3);
    mc=y;
    blas::axpy(alpha, x, mc);
    std::cout << "M :\n" << print_mat(M) << '\n';
    mr=y;
    blas::axpy(alpha, x, mr);
    std::cout << "M :\n" << print_mat(M) << '\n';
  }
  return EXIT_SUCCESS;
}
