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
    vector v(n);
    for (size_type i=0; i<n; ++i)
      v(i)=rand_normal<double>::get();
    double alpha=rand_normal<double>::get();
    vector v1(alpha*v);
    vector v2(v);
    blas::scal(alpha, v2);
    std::cout << "v1 : " << print_vec(v1) << '\n'
	      << "v2 : " << print_vec(v2) << '\n';
    matrix M(n, n);
    for (size_type j=0; j<n; ++j)
      for (size_type i=0; i<n; ++i) 
     	M(i, j)=1;
    ublas::matrix_column<matrix> mc(M, 2);
    ublas::matrix_row<matrix> mr(M, 3);
    mc=v;
    blas::scal(alpha, mc);
    std::cout << "mc : " << print_vec(mc) << '\n';
    mr=v;
    blas::scal(alpha, mr);
    std::cout << "mr : " << print_vec(mr) << '\n';
    std::cout << "M :\n" << print_mat(M) << '\n';
  }
  {
    typedef Eigen::Matrix<double, Eigen::Dynamic, 1> vector;
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> matrix;
    typedef int size_type;
    rand_normal<double>::reset();
    size_type n=8;
    vector v(n);
    for (size_type i=0; i<n; ++i)
      v(i)=rand_normal<double>::get();
    double alpha=rand_normal<double>::get();
    vector v1(alpha*v);
    vector v2(v);
    blas::scal(alpha, v2);
    std::cout << "v1 : " << print_vec(v1) << '\n'
    	      << "v2 : " << print_vec(v2) << '\n';
    matrix M(n, n);
    for (size_type j=0; j<n; ++j)
      for (size_type i=0; i<n; ++i) 
     	M(i, j)=1;
     auto mc=M.col(2);
    auto mr=M.row(3);
    mc=v;
    blas::scal(alpha, mc);
    std::cout << "mc : " << print_vec(mc) << '\n';
    mr=v;
    blas::scal(alpha, mr);
    std::cout << "mr : " << print_vec(mr) << '\n';
    std::cout << "M :\n" << print_mat(M) << '\n';
  }
  return EXIT_SUCCESS;
}
