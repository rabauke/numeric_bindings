#include <cstdlib>
#include <iostream>
#include <complex>
#define BOOST_UBLAS_NO_ELEMENT_PROXIES
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/bindings/ublas/vector.hpp>
#include <boost/numeric/bindings/ublas/triangular.hpp>
#include <boost/numeric/bindings/blas/level2.hpp>
#include <boost/numeric/bindings/lower.hpp>
#include <boost/numeric/bindings/upper.hpp>
#include <boost/numeric/bindings/unit_lower.hpp>
#include <boost/numeric/bindings/unit_upper.hpp>
#include <boost/numeric/bindings/trans.hpp>
#include <boost/numeric/bindings/herm.hpp>
#include <boost/numeric/bindings/conj.hpp>
#include "print.hpp"
#include "random.hpp"

namespace ublas=boost::numeric::ublas;
namespace blas=boost::numeric::bindings::blas;

int main(int argc, char *argv[]) {
  {
    typedef std::complex<double> complex;
    typedef ublas::vector<complex> vector;
    typedef ublas::triangular_matrix<complex, ublas::lower, ublas::column_major> matrix_l;
    typedef ublas::triangular_matrix<complex, ublas::upper, ublas::column_major> matrix_u;
    typedef typename vector::size_type size_type;
    rand_normal<complex>::reset();
    size_type n=8;
    matrix_l A_l(n, n);
    matrix_u A_u(n, n);
    for (size_type j=0; j<n; ++j) {
      A_u(j, j)=rand_normal<complex>::get();
      A_l(j, j)=rand_normal<complex>::get();
      for (size_type i=0; i<j; ++i) {
        A_u(i, j)=rand_normal<complex>::get();
        A_l(j, i)=rand_normal<complex>::get();
       }
    }
    vector x(n);
    for (size_type i=0; i<n; ++i)
      x(i)=rand_normal<complex>::get();
    {
      vector x1(ublas::prod(A_l, x));
      vector x2(x);
      blas::tpmv(A_l, x2);
      std::cout << "testing boost::ublas containers\n"
    		<< "using ublas (lower): " << print_vec(x1) << '\n'
    		<< "using blas  (lower): " << print_vec(x2) << '\n'
    		<< '\n';
    }
    {
      vector x1(ublas::prod(A_u, x));
      vector x2(x);
      blas::tpmv(A_u, x2);
      std::cout << "testing boost::ublas containers\n"
    		<< "using ublas (upper): " << print_vec(x1) << '\n'
    		<< "using blas  (upper): " << print_vec(x2) << '\n'
    		<< '\n';
    }
    {
      vector x1(ublas::prod(ublas::trans(A_l), x));
      vector x2(x);
      blas::tpmv(blas::trans(blas::lower(A_l)), x2);
      std::cout << "testing boost::ublas containers\n"
    		<< "using ublas (trans, lower): " << print_vec(x1) << '\n'
    		<< "using blas  (trans, lower): " << print_vec(x2) << '\n'
    		<< '\n';
    }
    {
      vector x1(ublas::prod(ublas::trans(A_u), x));
      vector x2(x);
      blas::tpmv(blas::trans(blas::upper(A_u)), x2);
      std::cout << "testing boost::ublas containers\n"
    		<< "using ublas (trans, upper): " << print_vec(x1) << '\n'
    		<< "using blas  (trans, upper): " << print_vec(x2) << '\n'
    		<< '\n';
    }
    {
      vector x1(ublas::prod(ublas::herm(A_l), x));
      vector x2(x);
      blas::tpmv(blas::conj(blas::lower(A_l)), x2);
      std::cout << "testing boost::ublas containers\n"
    		<< "using ublas (htrans, lower): " << print_vec(x1) << '\n'
    		<< "using blas  (htrans, lower): " << print_vec(x2) << '\n'
    		<< '\n';
    }
    {
      vector x1(ublas::prod(ublas::herm(A_u), x));
      vector x2(x);
      blas::tpmv(blas::conj(blas::upper(A_u)), x2);
      std::cout << "testing boost::ublas containers\n"
    		<< "using ublas (htrans, upper): " << print_vec(x1) << '\n'
    		<< "using blas  (htrans, upper): " << print_vec(x2) << '\n'
    		<< '\n';
    }

    for (size_type i=0; i<n; ++i) {
      A_l(i, i)=1;
      A_u(i, i)=1;
    }

    {
      vector x1(ublas::prod(A_l, x));
      vector x2(x);
      blas::tpmv(blas::unit_lower(A_l), x2);
      std::cout << "testing boost::ublas containers\n"
    		<< "using ublas (unit_lower): " << print_vec(x1) << '\n'
    		<< "using blas  (unit_lower): " << print_vec(x2) << '\n'
    		<< '\n';
    }
    {
      vector x1(ublas::prod(A_u, x));
      vector x2(x);
      blas::tpmv(blas::unit_upper(A_u), x2);
      std::cout << "testing boost::ublas containers\n"
    		<< "using ublas (unit_upper): " << print_vec(x1) << '\n'
    		<< "using blas  (unit_upper): " << print_vec(x2) << '\n'
    		<< '\n';
    }
    {
      vector x1(ublas::prod(ublas::trans(A_l), x));
      vector x2(x);
      blas::tpmv(blas::trans(blas::unit_lower(A_l)), x2);
      std::cout << "testing boost::ublas containers\n"
    		<< "using ublas (trans, unit_lower): " << print_vec(x1) << '\n'
    		<< "using blas  (trans, unit_lower): " << print_vec(x2) << '\n'
    		<< '\n';
    }
    {
      vector x1(ublas::prod(ublas::trans(A_u), x));
      vector x2(x);
      blas::tpmv(blas::trans(blas::unit_upper(A_u)), x2);
      std::cout << "testing boost::ublas containers\n"
    		<< "using ublas (trans, unit_upper): " << print_vec(x1) << '\n'
    		<< "using blas  (trans, unit_upper): " << print_vec(x2) << '\n'
    		<< '\n';
    }
    {
      vector x1(ublas::prod(ublas::herm(A_l), x));
      vector x2(x);
      blas::tpmv(blas::conj(blas::unit_lower(A_l)), x2);
      std::cout << "testing boost::ublas containers\n"
    		<< "using ublas (htrans, unit_lower): " << print_vec(x1) << '\n'
    		<< "using blas  (htrans, unit_lower): " << print_vec(x2) << '\n'
    		<< '\n';
    }
    {
      vector x1(ublas::prod(ublas::herm(A_u), x));
      vector x2(x);
      blas::tpmv(blas::conj(blas::unit_upper(A_u)), x2);
      std::cout << "testing boost::ublas containers\n"
    		<< "using ublas (htrans, unit_upper): " << print_vec(x1) << '\n'
    		<< "using blas  (htrans, unit_upper): " << print_vec(x2) << '\n'
    		<< '\n';
    }
  }
  return EXIT_SUCCESS;
}
