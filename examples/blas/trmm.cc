#include <cstdlib>
#include <iostream>
#include <complex>
#define BOOST_UBLAS_NO_ELEMENT_PROXIES
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/bindings/ublas/matrix.hpp>
#include <boost/numeric/bindings/blas/level3.hpp>
#include <boost/numeric/bindings/lower.hpp>
#include <boost/numeric/bindings/upper.hpp>
#include <boost/numeric/bindings/unit_lower.hpp>
#include <boost/numeric/bindings/unit_upper.hpp>
#include <boost/numeric/bindings/trans.hpp>
#include <boost/numeric/bindings/conj.hpp>
#include <boost/numeric/bindings/left.hpp>
#include <boost/numeric/bindings/right.hpp>
#include "print.hpp"
#include "random.hpp"

namespace ublas=boost::numeric::ublas;
namespace blas=boost::numeric::bindings::blas;

int main(int argc, char *argv[]) {
  {
    typedef std::complex<double> complex;
    typedef ublas::vector<complex> vector;
    typedef ublas::matrix<complex, ublas::column_major> matrix;
    typedef typename vector::size_type size_type;
    rand_normal<complex>::reset();
    size_type n=8;
    matrix A_l(n, n), A_u(n, n), B(n, n);
    for (size_type j=0; j<n; ++j) {
      for (size_type i=0; i<j; ++i) {
    	A_u(i, j)=rand_normal<complex>::get();
	A_l(j, i)=rand_normal<complex>::get();
      }
      A_u(j, j)=rand_normal<complex>::get();
      A_l(j, j)=rand_normal<complex>::get();
      for (size_type i=j+1; i<n; ++i) {
    	A_u(i, j)=0;
	A_l(j, i)=0;
      }
    }
    for (size_type j=0; j<n; ++j)
      for (size_type i=0; i<n; ++i)
	B(j, i)=rand_normal<complex>::get();
    complex alpha=rand_normal<complex>::get();
    {
      matrix B1(alpha*ublas::prod(A_l, B));
      matrix B2(B);
      blas::trmm(blas::left(), alpha, blas::lower(A_l), B2);
      std::cout << "testing boost::ublas containers\n"
     		<< "using ublas (left multiply, lower):\n" << print_mat(B1) << '\n'
    		<< "using blas  (left multiply, lower):\n" << print_mat(B2) << '\n'
    		<< '\n';
    }
    {
      matrix B1(alpha*ublas::prod(A_u, B));
      matrix B2(B);
      blas::trmm(blas::left(), alpha, blas::upper(A_u), B2);
      std::cout << "testing boost::ublas containers\n"
     		<< "using ublas (left multiply, upper):\n" << print_mat(B1) << '\n'
    		<< "using blas  (left multiply, upper):\n" << print_mat(B2) << '\n'
    		<< '\n';
    }
    {
      matrix B1(alpha*ublas::prod(ublas::trans(A_l), B));
      matrix B2(B);
      blas::trmm(blas::left(), alpha, blas::trans(blas::lower(A_l)), B2);
      std::cout << "testing boost::ublas containers\n"
     		<< "using ublas (left multiply, lower):\n" << print_mat(B1) << '\n'
    		<< "using blas  (left multiply, lower):\n" << print_mat(B2) << '\n'
    		<< '\n';
    }
    {
      matrix B1(alpha*ublas::prod(ublas::trans(A_u), B));
      matrix B2(B);
      blas::trmm(blas::left(), alpha, blas::trans(blas::upper(A_u)), B2);
      std::cout << "testing boost::ublas containers\n"
     		<< "using ublas (left multiply, upper):\n" << print_mat(B1) << '\n'
    		<< "using blas  (left multiply, upper):\n" << print_mat(B2) << '\n'
    		<< '\n';
    }
    {
      matrix B1(alpha*ublas::prod(ublas::conj(ublas::trans(A_l)), B));
      matrix B2(B);
      blas::trmm(blas::left(), alpha, blas::conj(blas::lower(A_l)), B2);
      std::cout << "testing boost::ublas containers\n"
     		<< "using ublas (left multiply, lower):\n" << print_mat(B1) << '\n'
    		<< "using blas  (left multiply, lower):\n" << print_mat(B2) << '\n'
    		<< '\n';
    }
    {
      matrix B1(alpha*ublas::prod(ublas::conj(ublas::trans(A_u)), B));
      matrix B2(B);
      blas::trmm(blas::left(), alpha, blas::conj(blas::upper(A_u)), B2);
      std::cout << "testing boost::ublas containers\n"
     		<< "using ublas (left multiply, upper):\n" << print_mat(B1) << '\n'
    		<< "using blas  (left multiply, upper):\n" << print_mat(B2) << '\n'
    		<< '\n';
    }

    {
      matrix B1(alpha*ublas::prod(B, A_l));
      matrix B2(B);
      blas::trmm(blas::right(), alpha, blas::lower(A_l), B2);
      std::cout << "testing boost::ublas containers\n"
     		<< "using ublas (right multiply, lower):\n" << print_mat(B1) << '\n'
    		<< "using blas  (right multiply, lower):\n" << print_mat(B2) << '\n'
    		<< '\n';
    }
    {
      matrix B1(alpha*ublas::prod(B, A_u));
      matrix B2(B);
      blas::trmm(blas::right(), alpha, blas::upper(A_u), B2);
      std::cout << "testing boost::ublas containers\n"
     		<< "using ublas (right multiply, upper):\n" << print_mat(B1) << '\n'
    		<< "using blas  (right multiply, upper):\n" << print_mat(B2) << '\n'
    		<< '\n';
    }
    {
      matrix B1(alpha*ublas::prod(B, ublas::trans(A_l)));
      matrix B2(B);
      blas::trmm(blas::right(), alpha, blas::trans(blas::lower(A_l)), B2);
      std::cout << "testing boost::ublas containers\n"
     		<< "using ublas (right multiply, lower):\n" << print_mat(B1) << '\n'
    		<< "using blas  (right multiply, lower):\n" << print_mat(B2) << '\n'
    		<< '\n';
    }
    {
      matrix B1(alpha*ublas::prod(B, ublas::trans(A_u)));
      matrix B2(B);
      blas::trmm(blas::right(), alpha, blas::trans(blas::upper(A_u)), B2);
      std::cout << "testing boost::ublas containers\n"
     		<< "using ublas (right multiply, upper):\n" << print_mat(B1) << '\n'
    		<< "using blas  (right multiply, upper):\n" << print_mat(B2) << '\n'
    		<< '\n';
    }
    {
      matrix B1(alpha*ublas::prod(B, ublas::conj(ublas::trans(A_l))));
      matrix B2(B);
      blas::trmm(blas::right(), alpha, blas::conj(blas::lower(A_l)), B2);
      std::cout << "testing boost::ublas containers\n"
     		<< "using ublas (right multiply, lower):\n" << print_mat(B1) << '\n'
    		<< "using blas  (right multiply, lower):\n" << print_mat(B2) << '\n'
    		<< '\n';
    }
    {
      matrix B1(alpha*ublas::prod(B, ublas::conj(ublas::trans(A_u))));
      matrix B2(B);
      blas::trmm(blas::right(), alpha, blas::conj(blas::upper(A_u)), B2);
      std::cout << "testing boost::ublas containers\n"
     		<< "using ublas (right multiply, upper):\n" << print_mat(B1) << '\n'
    		<< "using blas  (right multiply, upper):\n" << print_mat(B2) << '\n'
    		<< '\n';
    }

    for (size_type i=0; i<n; ++i) {
      A_l(i, i)=1;
      A_u(i, i)=1;
    }

    {
      matrix B1(alpha*ublas::prod(A_l, B));
      matrix B2(B);
      blas::trmm(blas::left(), alpha, blas::unit_lower(A_l), B2);
      std::cout << "testing boost::ublas containers\n"
     		<< "using ublas (left multiply, unit_lower):\n" << print_mat(B1) << '\n'
    		<< "using blas  (left multiply, unit_lower):\n" << print_mat(B2) << '\n'
    		<< '\n';
    }
    {
      matrix B1(alpha*ublas::prod(A_u, B));
      matrix B2(B);
      blas::trmm(blas::left(), alpha, blas::unit_upper(A_u), B2);
      std::cout << "testing boost::ublas containers\n"
     		<< "using ublas (left multiply, unit_upper):\n" << print_mat(B1) << '\n'
    		<< "using blas  (left multiply, unit_upper):\n" << print_mat(B2) << '\n'
    		<< '\n';
    }
    {
      matrix B1(alpha*ublas::prod(ublas::trans(A_l), B));
      matrix B2(B);
      blas::trmm(blas::left(), alpha, blas::trans(blas::unit_lower(A_l)), B2);
      std::cout << "testing boost::ublas containers\n"
     		<< "using ublas (left multiply, unit_lower):\n" << print_mat(B1) << '\n'
    		<< "using blas  (left multiply, unit_lower):\n" << print_mat(B2) << '\n'
    		<< '\n';
    }
    {
      matrix B1(alpha*ublas::prod(ublas::trans(A_u), B));
      matrix B2(B);
      blas::trmm(blas::left(), alpha, blas::trans(blas::unit_upper(A_u)), B2);
      std::cout << "testing boost::ublas containers\n"
     		<< "using ublas (left multiply, unit_upper):\n" << print_mat(B1) << '\n'
    		<< "using blas  (left multiply, unit_upper):\n" << print_mat(B2) << '\n'
    		<< '\n';
    }
    {
      matrix B1(alpha*ublas::prod(ublas::conj(ublas::trans(A_l)), B));
      matrix B2(B);
      blas::trmm(blas::left(), alpha, blas::conj(blas::unit_lower(A_l)), B2);
      std::cout << "testing boost::ublas containers\n"
     		<< "using ublas (left multiply, unit_lower):\n" << print_mat(B1) << '\n'
    		<< "using blas  (left multiply, unit_lower):\n" << print_mat(B2) << '\n'
    		<< '\n';
    }
    {
      matrix B1(alpha*ublas::prod(ublas::conj(ublas::trans(A_u)), B));
      matrix B2(B);
      blas::trmm(blas::left(), alpha, blas::conj(blas::unit_upper(A_u)), B2);
      std::cout << "testing boost::ublas containers\n"
     		<< "using ublas (left multiply, unit_upper):\n" << print_mat(B1) << '\n'
    		<< "using blas  (left multiply, unit_upper):\n" << print_mat(B2) << '\n'
    		<< '\n';
    }

    {
      matrix B1(alpha*ublas::prod(B, A_l));
      matrix B2(B);
      blas::trmm(blas::right(), alpha, blas::unit_lower(A_l), B2);
      std::cout << "testing boost::ublas containers\n"
     		<< "using ublas (right multiply, unit_lower):\n" << print_mat(B1) << '\n'
    		<< "using blas  (right multiply, unit_lower):\n" << print_mat(B2) << '\n'
    		<< '\n';
    }
    {
      matrix B1(alpha*ublas::prod(B, A_u));
      matrix B2(B);
      blas::trmm(blas::right(), alpha, blas::unit_upper(A_u), B2);
      std::cout << "testing boost::ublas containers\n"
     		<< "using ublas (right multiply, unit_upper):\n" << print_mat(B1) << '\n'
    		<< "using blas  (right multiply, unit_upper):\n" << print_mat(B2) << '\n'
    		<< '\n';
    }
    {
      matrix B1(alpha*ublas::prod(B, ublas::trans(A_l)));
      matrix B2(B);
      blas::trmm(blas::right(), alpha, blas::trans(blas::unit_lower(A_l)), B2);
      std::cout << "testing boost::ublas containers\n"
     		<< "using ublas (right multiply, unit_lower):\n" << print_mat(B1) << '\n'
    		<< "using blas  (right multiply, unit_lower):\n" << print_mat(B2) << '\n'
    		<< '\n';
    }
    {
      matrix B1(alpha*ublas::prod(B, ublas::trans(A_u)));
      matrix B2(B);
      blas::trmm(blas::right(), alpha, blas::trans(blas::unit_upper(A_u)), B2);
      std::cout << "testing boost::ublas containers\n"
     		<< "using ublas (right multiply, unit_upper):\n" << print_mat(B1) << '\n'
    		<< "using blas  (right multiply, unit_upper):\n" << print_mat(B2) << '\n'
    		<< '\n';
    }
    {
      matrix B1(alpha*ublas::prod(B, ublas::conj(ublas::trans(A_l))));
      matrix B2(B);
      blas::trmm(blas::right(), alpha, blas::conj(blas::unit_lower(A_l)), B2);
      std::cout << "testing boost::ublas containers\n"
     		<< "using ublas (right multiply, unit_lower):\n" << print_mat(B1) << '\n'
    		<< "using blas  (right multiply, unit_lower):\n" << print_mat(B2) << '\n'
    		<< '\n';
    }
    {
      matrix B1(alpha*ublas::prod(B, ublas::conj(ublas::trans(A_u))));
      matrix B2(B);
      blas::trmm(blas::right(), alpha, blas::conj(blas::unit_upper(A_u)), B2);
      std::cout << "testing boost::ublas containers\n"
     		<< "using ublas (right multiply, unit_upper):\n" << print_mat(B1) << '\n'
    		<< "using blas  (right multiply, unit_upper):\n" << print_mat(B2) << '\n'
    		<< '\n';
    }
    
  }
  return EXIT_SUCCESS;
}
